"""
VLN System - Core Module Implementations
==========================================
For: Bandwidth-Constrained Cooperative VLN (CoRL 2026)

Modules:
  1. CrossModalAttention   - attends over language tokens given visual query
  2. MessageAggregator     - attends over received teammate observations
  3. AgentStateGRU         - maintains per-agent recurrent state
  4. NavigationHead        - predicts navigation action logits
  5. CommunicationGating   - learned binary send/don't-send decision
  6. CooperativeVLNAgent   - full agent wrapping all of the above
  7. MultiAgentVLNSystem   - orchestrates N agents + bandwidth budget
  8. BandwidthConstrainedLoss - combined loss with penalty

Dependencies:
  pip install torch clip-by-openai
  import clip  (openai/CLIP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from typing import List, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# 1. Cross-Modal Attention
#    Attends over language token embeddings using the visual feature as query.
#    This is how the agent asks: "given what I see, which part of the
#    instruction is most relevant right now?"
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Single-head cross-attention:
        Query  = visual feature   (B, v_dim)
        Keys   = language tokens  (B, seq_len, l_dim)
        Values = language tokens  (B, seq_len, l_dim)

    Projects Q, K, V into a shared hidden_dim, computes scaled dot-product
    attention, and returns a context vector of shape (B, hidden_dim).

    Args:
        v_dim      : dimension of the visual feature vector (e.g. 512 for CLIP ViT-B/32)
        l_dim      : dimension of each language token embedding (e.g. 512 for CLIP)
        hidden_dim : projection dimension for Q/K/V (default 512)
        dropout    : attention dropout probability
    """

    def __init__(self, v_dim: int = 512, l_dim: int = 512,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

        # Linear projections for Query, Key, Value
        self.W_q = nn.Linear(v_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(l_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(l_dim, hidden_dim, bias=False)

        # Output projection + layer norm
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual_feat: torch.Tensor,          # (B, v_dim)
        lang_tokens: torch.Tensor,          # (B, seq_len, l_dim)
        lang_mask: Optional[torch.Tensor] = None  # (B, seq_len) bool, True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context   : (B, hidden_dim)  -- the language-grounded visual context
            attn_weights : (B, seq_len)  -- attention distribution over tokens
                          (useful for visualisation / analysis)
        """
        B = visual_feat.size(0)

        # Q shape: (B, 1, hidden_dim)  -- unsqueeze for bmm
        Q = self.W_q(visual_feat).unsqueeze(1)
        # K, V shapes: (B, seq_len, hidden_dim)
        K = self.W_k(lang_tokens)
        V = self.W_v(lang_tokens)

        # Scaled dot-product attention scores: (B, 1, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Mask padding tokens with -inf so softmax ignores them
        if lang_mask is not None:
            # lang_mask: True where padded → set those scores to -inf
            scores = scores.masked_fill(
                lang_mask.unsqueeze(1),   # (B, 1, seq_len)
                float('-inf')
            )

        attn_weights = F.softmax(scores, dim=-1)          # (B, 1, seq_len)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values: (B, 1, hidden_dim) → (B, hidden_dim)
        context = torch.bmm(attn_weights, V).squeeze(1)
        context = self.out_proj(context)
        context = self.layer_norm(context)

        return context, attn_weights.squeeze(1)  # (B, hidden_dim), (B, seq_len)


# ---------------------------------------------------------------------------
# 2. Message Aggregator
#    Attends over the message buffer: a variable-length set of visual features
#    broadcast by teammate agents in recent timesteps.
#    If the buffer is empty, returns a zero vector (no information received).
# ---------------------------------------------------------------------------

class MessageAggregator(nn.Module):
    """
    Aggregates a variable number of incoming teammate messages using attention.

    Each message is a visual feature vector from a teammate agent. The current
    agent's GRU hidden state acts as the query — it asks "which of my
    teammates' recent observations is most useful to me right now?"

    Args:
        hidden_dim : dimension of agent hidden state = message feature dim
        dropout    : dropout on attention weights
    """

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.scale = math.sqrt(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Learned gate: starts near 0 (ignore messages), learns to open gradually
        # Initialised with negative bias so sigmoid(x) ≈ 0.1 at start
        self.msg_gate = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.msg_gate.weight)
        nn.init.constant_(self.msg_gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(
        self,
        agent_hidden: torch.Tensor,          # (B, hidden_dim)
        messages: Optional[torch.Tensor],    # (B, num_msgs, hidden_dim) or None
    ) -> torch.Tensor:
        """
        Returns:
            aggregated : (B, hidden_dim)
                Near-zero at start, gradually learns to incorporate messages.
        """
        B = agent_hidden.size(0)
        device = agent_hidden.device

        # No messages received → return zeros
        if messages is None or messages.size(1) == 0:
            return torch.zeros(B, self.hidden_dim, device=device)

        Q = self.W_q(agent_hidden).unsqueeze(1)   # (B, 1, hidden_dim)
        K = self.W_k(messages)                     # (B, num_msgs, hidden_dim)
        V = self.W_v(messages)                     # (B, num_msgs, hidden_dim)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 1, num_msgs)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        aggregated = torch.bmm(attn, V).squeeze(1)  # (B, hidden_dim)
        aggregated = self.out_proj(aggregated)
        aggregated = self.layer_norm(aggregated)

        # Learned gate: controls how much of the message to let through
        # Starts near zero (safe), opens as training progresses
        gate = torch.sigmoid(self.msg_gate(agent_hidden))  # (B, hidden_dim)
        aggregated = gate * aggregated

        return aggregated  # (B, hidden_dim)


# ---------------------------------------------------------------------------
# 3. Agent State GRU
#    Maintains the agent's recurrent hidden state across timesteps.
#    Input at each step: [cross-modal context, aggregated messages, prev action]
# ---------------------------------------------------------------------------

class AgentStateGRU(nn.Module):
    """
    GRU that updates the agent's hidden state each timestep.

    Input concatenation: [c_i(t), m_i(t), action_embed(a_{t-1})]
        c_i(t)       : (B, hidden_dim)  cross-modal context
        m_i(t)       : (B, hidden_dim)  aggregated teammate messages
        action_embed : (B, action_dim)  embedding of previous action

    Args:
        hidden_dim  : GRU hidden state size (and input component size)
        action_dim  : number of discrete navigation actions (e.g. 4 or 36)
        action_embed_dim : embedding size for action tokens
    """

    def __init__(self, hidden_dim: int = 512,
                 action_dim: int = 36, action_embed_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)

        # GRU input: concat of [context, messages, action_embed]
        gru_input_dim = hidden_dim + hidden_dim + action_embed_dim
        self.gru = nn.GRUCell(gru_input_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        context: torch.Tensor,        # (B, hidden_dim)  cross-modal context
        messages: torch.Tensor,       # (B, hidden_dim)  aggregated messages
        prev_action: torch.Tensor,    # (B,)             integer action indices
        hidden: torch.Tensor          # (B, hidden_dim)  previous hidden state
    ) -> torch.Tensor:
        """
        Returns:
            new_hidden : (B, hidden_dim)
        """
        action_embed = self.action_embedding(prev_action)   # (B, action_embed_dim)
        gru_input    = torch.cat([context, messages, action_embed], dim=-1)
        new_hidden   = self.gru(gru_input, hidden)
        new_hidden   = self.layer_norm(new_hidden)
        return new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns zero initial hidden state."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# ---------------------------------------------------------------------------
# 4. Navigation Head
#    Maps the agent's hidden state to action logits.
#    In R2R, actions = which candidate viewpoint to move to (up to 36).
# ---------------------------------------------------------------------------

class NavigationHead(nn.Module):
    """
    Two-layer MLP that scores each candidate action given the agent hidden state.

    In R2R, at each viewpoint there are up to 36 candidate next viewpoints,
    each described by a visual feature. The agent scores each candidate by
    comparing its hidden state to the candidate features.

    For simplicity here we implement the flat action-space version (fixed
    action_dim). For the full candidate-scoring version, see forward_candidates().

    Args:
        hidden_dim  : agent hidden state dimension
        action_dim  : number of discrete actions (e.g. 4 for {fwd, left, right, stop})
    """

    def __init__(self, hidden_dim: int = 512, action_dim: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden : (B, hidden_dim)
        Returns:
            logits : (B, action_dim)  — pass through CrossEntropy for training
        """
        return self.mlp(hidden)

    def forward_candidates(
        self,
        hidden: torch.Tensor,          # (B, hidden_dim)
        candidate_feats: torch.Tensor  # (B, num_candidates, hidden_dim)
    ) -> torch.Tensor:
        """
        R2R-style: score each candidate viewpoint by dot product with hidden state.
        Returns:
            scores : (B, num_candidates)
        """
        # hidden: (B, hidden_dim) → (B, hidden_dim, 1)
        h = hidden.unsqueeze(-1)
        # scores: (B, num_candidates)
        scores = torch.bmm(candidate_feats, h).squeeze(-1)
        return scores


# ---------------------------------------------------------------------------
# 5. Communication Gating Module  ← THE NOVEL CONTRIBUTION
#
#    Takes the agent's current hidden state and outputs:
#      (a) p_send : probability of sending this timestep
#      (b) gate   : sampled binary decision (differentiable via REINFORCE)
#
#    This is a small MLP — intentionally lightweight. The agent's hidden state
#    already encodes everything it knows (what it sees, where it's been, what
#    its teammates have shared). The gate just needs to read that and decide.
#
#    Training signal: REINFORCE — the gate is treated as a policy.
#    Reward: navigation outcome (episode success + shaped distance reduction).
#    Penalty: exceeding bandwidth budget B.
# ---------------------------------------------------------------------------

class CommunicationGating(nn.Module):
    """
    Learned binary communication gate.

    Architecture:
        hidden_state (hidden_dim)
            → Linear(hidden_dim, gate_hidden)
            → LayerNorm → ReLU
            → Linear(gate_hidden, gate_hidden // 2)
            → ReLU
            → Linear(gate_hidden // 2, 1)
            → Sigmoid
            → p_send ∈ (0, 1)
            → Bernoulli sample → gate ∈ {0, 1}

    The gate also receives an optional "budget signal": how much of the
    bandwidth budget remains. This lets the agent learn to save budget for
    later critical moments. Concatenated to hidden_state if provided.

    Args:
        hidden_dim   : agent hidden state dimension (e.g. 512)
        gate_hidden  : gating MLP hidden size (smaller than hidden_dim)
        use_budget_signal : whether to pass remaining budget fraction as input
    """

    def __init__(self, hidden_dim: int = 512,
                 gate_hidden: int = 128,
                 use_budget_signal: bool = True):
        super().__init__()
        self.use_budget_signal = use_budget_signal

        # Input dim: hidden_dim + 1 (budget fraction) if use_budget_signal
        in_dim = hidden_dim + 1 if use_budget_signal else hidden_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, gate_hidden // 2),
            nn.ReLU(),
            nn.Linear(gate_hidden // 2, 1),
            nn.Sigmoid()   # output is p_send ∈ (0, 1)
        )

        # Small log std for entropy regularisation tracking
        self._last_entropy = None

    def forward(
        self,
        hidden: torch.Tensor,                          # (B, hidden_dim)
        budget_remaining: Optional[torch.Tensor] = None,  # (B, 1) fraction 0→1
        deterministic: bool = False                    # True at eval/inference
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden           : agent hidden state, (B, hidden_dim)
            budget_remaining : fraction of budget left this episode, (B, 1)
                               e.g. if budget=20 and 14 sends remain → 0.7
            deterministic    : if True, threshold at 0.5 (no sampling)

        Returns:
            gate     : (B,)    binary {0, 1} — the actual send decision
            p_send   : (B,)    send probability — used in REINFORCE loss
            log_prob : (B,)    log probability of the gate decision
        """
        if self.use_budget_signal and budget_remaining is not None:
            gate_input = torch.cat([hidden, budget_remaining], dim=-1)
        else:
            gate_input = hidden

        p_send = self.gate_mlp(gate_input).squeeze(-1)   # (B,)

        dist   = Bernoulli(probs=p_send)
        self._last_entropy = dist.entropy().mean()

        if deterministic:
            gate = (p_send >= 0.5).float()
        else:
            gate = dist.sample()                          # (B,) ∈ {0., 1.}

        log_prob = dist.log_prob(gate)                    # (B,)

        return gate, p_send, log_prob

    @property
    def last_entropy(self) -> Optional[torch.Tensor]:
        """Entropy of the gate distribution — use for entropy regularisation."""
        return self._last_entropy


# ---------------------------------------------------------------------------
# 6. Full Cooperative VLN Agent
#    Wraps all modules. One instance per agent in the team.
# ---------------------------------------------------------------------------

class CooperativeVLNAgent(nn.Module):
    """
    Single agent in the bandwidth-constrained cooperative VLN system.

    At each timestep t, the agent:
      1. Encodes visual observation via CLIP (frozen, done externally)
      2. Attends over language tokens with CrossModalAttention
      3. Aggregates teammate messages with MessageAggregator
      4. Updates its GRU hidden state
      5. Predicts navigation action with NavigationHead
      6. Decides whether to broadcast with CommunicationGating

    The CLIP encoders are NOT part of this module — pass pre-computed
    features (v_feat, l_tokens) for efficiency (encode once, use everywhere).

    Args:
        v_dim        : CLIP visual feature dim (512 for ViT-B/32)
        l_dim        : CLIP language token dim (512)
        hidden_dim   : GRU and attention hidden dim
        action_dim   : number of navigation actions
        gate_hidden  : gating MLP hidden size
    """

    def __init__(
        self,
        v_dim: int = 512,
        l_dim: int = 512,
        hidden_dim: int = 512,
        action_dim: int = 36,
        gate_hidden: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cross_attn   = CrossModalAttention(v_dim, l_dim, hidden_dim, dropout=0.4)
        self.msg_agg      = MessageAggregator(hidden_dim, dropout=0.4)
        self.agent_gru    = AgentStateGRU(hidden_dim, action_dim)
        self.nav_head     = NavigationHead(hidden_dim, action_dim)
        self.comm_gate    = CommunicationGating(hidden_dim, gate_hidden,
                                                use_budget_signal=True)

    def forward(
        self,
        v_feat: torch.Tensor,                           # (B, v_dim)
        l_tokens: torch.Tensor,                         # (B, seq_len, l_dim)
        hidden: torch.Tensor,                           # (B, hidden_dim)
        prev_action: torch.Tensor,                      # (B,) int
        messages: Optional[torch.Tensor],               # (B, num_msgs, hidden_dim) or None
        budget_remaining: Optional[torch.Tensor] = None,# (B, 1) fraction
        lang_mask: Optional[torch.Tensor] = None,       # (B, seq_len) padding mask
        deterministic: bool = False,
    ) -> dict:
        """
        Single forward step for one agent at one timestep.

        Returns a dict with all outputs needed for loss computation:
            action_logits : (B, action_dim)  — for nav CE loss
            new_hidden    : (B, hidden_dim)  — pass back next step
            gate          : (B,)             — binary send decision
            p_send        : (B,)             — gate probability
            log_prob_gate : (B,)             — for REINFORCE loss
            attn_weights  : (B, seq_len)     — language attention (for analysis)
            broadcast_feat: (B, hidden_dim)  — what to send if gate=1
        """
        # Step 1: Cross-modal attention → language-grounded visual context
        context, attn_weights = self.cross_attn(v_feat, l_tokens, lang_mask)
        # context: (B, hidden_dim)

        # Step 2: Aggregate incoming messages from teammates
        aggregated_msgs = self.msg_agg(hidden, messages)
        # aggregated_msgs: (B, hidden_dim)

        # Step 3: Update GRU hidden state
        new_hidden = self.agent_gru(context, aggregated_msgs, prev_action, hidden)
        # new_hidden: (B, hidden_dim)

        # Step 4: Navigation action logits
        action_logits = self.nav_head(new_hidden)
        # action_logits: (B, action_dim)

        # Step 5: Communication gating decision
        gate, p_send, log_prob_gate = self.comm_gate(
            new_hidden, budget_remaining, deterministic
        )

        # What we broadcast = current visual feature (projected to hidden_dim)
        # Using the cross-modal context as the broadcast signal — it's more
        # informative than raw v_feat because it's already instruction-aware
        broadcast_feat = context  # (B, hidden_dim)

        return {
            "action_logits" : action_logits,
            "new_hidden"    : new_hidden,
            "gate"          : gate,
            "p_send"        : p_send,
            "log_prob_gate" : log_prob_gate,
            "attn_weights"  : attn_weights,
            "broadcast_feat": broadcast_feat,
        }

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.agent_gru.init_hidden(batch_size, device)


# ---------------------------------------------------------------------------
# 7. Multi-Agent VLN System
#    Orchestrates N agents, manages the message buffer, and enforces
#    the bandwidth budget across an episode.
# ---------------------------------------------------------------------------

class MultiAgentVLNSystem(nn.Module):
    """
    Wraps N CooperativeVLNAgent instances and manages:
      - Shared language encoding (computed once per episode)
      - Message buffer per agent (what they've received from teammates)
      - Bandwidth tracking (how many sends used so far this episode)
      - Budget signal computation (remaining fraction for each agent)

    Args:
        n_agents    : number of agents (e.g. 2)
        bandwidth_budget : max total sends per episode across all agents
        **agent_kwargs  : passed to CooperativeVLNAgent
    """

    def __init__(self, n_agents: int = 2,
                 bandwidth_budget: int = 20,
                 **agent_kwargs):
        super().__init__()
        self.n_agents = n_agents
        self.bandwidth_budget = bandwidth_budget

        # One agent module per agent (shared weights or separate — here separate)
        self.agents = nn.ModuleList([
            CooperativeVLNAgent(**agent_kwargs)
            for _ in range(n_agents)
        ])

    def reset_episode(self, batch_size: int, device: torch.device):
        """Call at the start of each episode."""
        self.hiddens = [
            agent.init_hidden(batch_size, device)
            for agent in self.agents
        ]
        # Message buffer: list of lists — msg_buffer[i] = list of tensors from teammates
        self.msg_buffer = [[] for _ in range(self.n_agents)]
        self.sends_used = 0  # episode-level counter
        self.batch_size = batch_size
        self.device = device

    def step(
        self,
        v_feats: List[torch.Tensor],    # [agent_0_feat, agent_1_feat, ...] each (B, v_dim)
        l_tokens: torch.Tensor,         # (B, seq_len, l_dim)  shared instruction
        prev_actions: List[torch.Tensor],  # [(B,), ...] per agent
        lang_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> List[dict]:
        """
        One timestep for all agents.

        1. Compute budget signal for each agent
        2. Forward each agent
        3. For agents that gate=1, add their broadcast_feat to all other agents' buffers
        4. Increment sends_used

        Returns:
            outputs : list of per-agent output dicts (from CooperativeVLNAgent.forward)
        """
        budget_fraction = max(0.0, 1.0 - self.sends_used / self.bandwidth_budget)
        budget_tensor = torch.full(
            (self.batch_size, 1), budget_fraction, device=self.device
        )

        outputs = []
        for i, agent in enumerate(self.agents):
            # Build message tensor for agent i from its buffer
            if len(self.msg_buffer[i]) > 0:
                # Stack messages: (B, num_msgs, hidden_dim)
                msgs = torch.stack(self.msg_buffer[i], dim=1)
            else:
                msgs = None

            out = agent(
                v_feat           = v_feats[i],
                l_tokens         = l_tokens,
                hidden           = self.hiddens[i],
                prev_action      = prev_actions[i],
                messages         = msgs,
                budget_remaining = budget_tensor,
                lang_mask        = lang_mask,
                deterministic    = deterministic,
            )
            self.hiddens[i] = out["new_hidden"].detach()
            outputs.append(out)

        # Process gating decisions — update buffers and counter
        for i, out in enumerate(outputs):
            gate = out["gate"]  # (B,)
            # Only send if budget allows and gate fired
            # For batch training, we use a soft budget via the penalty loss.
            # Here we track the mean gate decision across the batch.
            mean_gate = gate.mean().item()
            self.sends_used += mean_gate  # approximate episode-level tracking

            if mean_gate > 0:
                # Broadcast to all OTHER agents' buffers
                feat = out["broadcast_feat"].detach()  # (B, hidden_dim)
                # Scale by gate (so un-gated items contribute zero)
                gated_feat = feat * gate.unsqueeze(-1)  # (B, hidden_dim)
                for j in range(self.n_agents):
                    if j != i:
                        self.msg_buffer[j].append(gated_feat)

        # Clear buffers after each step (keep only current step's messages)
        # For a sliding window, replace with a deque with maxlen
        for i in range(self.n_agents):
            self.msg_buffer[i] = []

        return outputs


# ---------------------------------------------------------------------------
# 8. Bandwidth-Constrained Loss
#    Combines:
#      (a) Navigation imitation learning loss (cross-entropy on GT actions)
#      (b) REINFORCE loss on gating decisions
#      (c) Bandwidth penalty when sends exceed budget
# ---------------------------------------------------------------------------

class BandwidthConstrainedLoss(nn.Module):
    """
    Combined training loss for the multi-agent VLN system.

    L_total = L_nav + alpha * L_gate + lambda_bw * L_bandwidth

    L_nav       : CrossEntropy(action_logits, gt_action) — imitation learning
    L_gate      : REINFORCE: -log_prob_gate * R_gate     — policy gradient on gate
    L_bandwidth : max(0, sends_used - budget)             — soft budget penalty

    R_gate (gate reward) can be:
      - Sparse: +1 if episode reached goal, 0 otherwise
      - Shaped: distance_to_goal_reduction at each step (faster learning)
      Recommend starting with sparse for simplicity.

    Args:
        alpha      : weight for gate REINFORCE loss (start at 0.1)
        lambda_bw  : weight for bandwidth penalty (start at 1.0)
        bandwidth_budget : budget B for the penalty term
        entropy_beta : coefficient for entropy regularisation (encourages exploration)
    """

    def __init__(self, alpha: float = 0.1, lambda_bw: float = 1.0,
                 bandwidth_budget: int = 20, entropy_beta: float = 0.01):
        super().__init__()
        self.alpha    = alpha
        self.lambda_bw = lambda_bw
        self.bandwidth_budget = bandwidth_budget
        self.entropy_beta = entropy_beta
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        action_logits_list: List[torch.Tensor],   # per-agent, per-step: (B, action_dim)
        gt_actions_list:    List[torch.Tensor],   # per-agent, per-step: (B,)
        log_prob_gate_list: List[torch.Tensor],   # per-agent, per-step: (B,)
        gate_rewards:       List[torch.Tensor],   # per-agent, per-step: (B,)
        total_sends:        torch.Tensor,         # (B,) total sends this episode
        entropy_list:       Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """
        All list arguments are flat lists over (agent_idx * timestep) combinations.
        Flatten across agents and timesteps before calling.

        Returns dict with individual loss components for logging.
        """
        # -- Navigation loss (imitation learning) --
        all_logits  = torch.cat(action_logits_list, dim=0)   # (B*T*N, action_dim)
        all_gt      = torch.cat(gt_actions_list,    dim=0)   # (B*T*N,)
        L_nav = self.ce_loss(all_logits, all_gt)

        # -- REINFORCE gate loss --
        # Subtract baseline (mean reward) to reduce variance
        all_log_prob = torch.cat(log_prob_gate_list, dim=0)   # (B*T*N,)
        all_rewards  = torch.cat(gate_rewards,        dim=0)  # (B*T*N,)
        baseline = all_rewards.mean().detach()
        advantage = (all_rewards - baseline)
        L_gate = -(all_log_prob * advantage).mean()

        # -- Bandwidth penalty --
        # total_sends: (B,) how many sends this episode per batch element
        L_bandwidth = F.relu(total_sends - self.bandwidth_budget).mean()

        # -- Optional entropy bonus (encourages gate to explore) --
        L_entropy = torch.tensor(0.0)
        if entropy_list is not None:
            L_entropy = -torch.stack(entropy_list).mean()  # negative = maximise entropy

        L_total = (L_nav
                   + self.alpha * L_gate
                   + self.lambda_bw * L_bandwidth
                   + self.entropy_beta * L_entropy)

        return {
            "loss"        : L_total,
            "L_nav"       : L_nav.item(),
            "L_gate"      : L_gate.item(),
            "L_bandwidth" : L_bandwidth.item(),
            "L_entropy"   : L_entropy.item() if isinstance(L_entropy, torch.Tensor) else 0.0,
        }


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify shapes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cpu")

    B        = 4     # batch size
    seq_len  = 20    # instruction token length
    v_dim    = 512   # CLIP visual dim
    l_dim    = 512   # CLIP language dim
    hidden_dim = 512
    action_dim = 36
    n_agents = 2

    print("=" * 60)
    print("VLN Module Sanity Check")
    print("=" * 60)

    # ---- CrossModalAttention ----
    cma = CrossModalAttention(v_dim, l_dim, hidden_dim, dropout=0.4)
    v_feat   = torch.randn(B, v_dim)
    l_tokens = torch.randn(B, seq_len, l_dim)
    context, attn_w = cma(v_feat, l_tokens)
    print(f"CrossModalAttention  → context: {context.shape}, attn_weights: {attn_w.shape}")
    assert context.shape == (B, hidden_dim)
    assert attn_w.shape  == (B, seq_len)

    # ---- MessageAggregator ----
    msg_agg  = MessageAggregator(hidden_dim, dropout=0.4)
    hidden   = torch.randn(B, hidden_dim)
    messages = torch.randn(B, 3, hidden_dim)   # 3 messages received
    agg      = msg_agg(hidden, messages)
    print(f"MessageAggregator    → aggregated: {agg.shape}")
    agg_none = msg_agg(hidden, None)
    print(f"MessageAggregator (no msgs) → {agg_none.shape} (all zeros: {agg_none.abs().sum().item() == 0})")
    assert agg.shape == (B, hidden_dim)

    # ---- AgentStateGRU ----
    gru      = AgentStateGRU(hidden_dim, action_dim)
    prev_act = torch.zeros(B, dtype=torch.long)
    h0       = gru.init_hidden(B, device)
    h1       = gru(context, agg, prev_act, h0)
    print(f"AgentStateGRU        → new_hidden: {h1.shape}")
    assert h1.shape == (B, hidden_dim)

    # ---- NavigationHead ----
    nav_head = NavigationHead(hidden_dim, action_dim)
    logits   = nav_head(h1)
    print(f"NavigationHead       → logits: {logits.shape}")
    assert logits.shape == (B, action_dim)

    # ---- CommunicationGating ----
    gate_mod = CommunicationGating(hidden_dim, gate_hidden=128)
    budget_r = torch.full((B, 1), 0.6)   # 60% budget remaining
    gate, p_send, log_prob = gate_mod(h1, budget_r)
    print(f"CommunicationGating  → gate: {gate.shape}, p_send mean: {p_send.mean():.3f}")
    print(f"  gate values (first 4): {gate.tolist()}")
    print(f"  entropy: {gate_mod.last_entropy:.4f}")
    assert gate.shape    == (B,)
    assert p_send.shape  == (B,)
    assert log_prob.shape == (B,)

    # ---- Full Agent ----
    agent = CooperativeVLNAgent(v_dim, l_dim, hidden_dim, action_dim)
    h_init = agent.init_hidden(B, device)
    out    = agent(v_feat, l_tokens, h_init, prev_act, messages, budget_r)
    print(f"\nCooperativeVLNAgent output keys: {list(out.keys())}")
    print(f"  action_logits:  {out['action_logits'].shape}")
    print(f"  new_hidden:     {out['new_hidden'].shape}")
    print(f"  gate:           {out['gate'].tolist()}")
    print(f"  broadcast_feat: {out['broadcast_feat'].shape}")

    # ---- Multi-Agent System ----
    system = MultiAgentVLNSystem(
        n_agents=n_agents, bandwidth_budget=20,
        v_dim=v_dim, l_dim=l_dim, hidden_dim=hidden_dim, action_dim=action_dim
    )
    system.reset_episode(B, device)
    v_feats_all  = [torch.randn(B, v_dim) for _ in range(n_agents)]
    prev_acts_all = [torch.zeros(B, dtype=torch.long) for _ in range(n_agents)]
    step_outputs = system.step(v_feats_all, l_tokens, prev_acts_all)
    print(f"\nMultiAgentVLNSystem step → {len(step_outputs)} agent outputs")
    print(f"  sends_used after step: {system.sends_used:.2f}")

    print("\nAll checks passed.")
    print("=" * 60)
    print("\nParameter counts:")
    for name, mod in [
        ("CrossModalAttention", cma),
        ("MessageAggregator",   msg_agg),
        ("AgentStateGRU",       gru),
        ("NavigationHead",      nav_head),
        ("CommunicationGating", gate_mod),
        ("CooperativeVLNAgent", agent),
    ]:
        n_params = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<25} {n_params:>8,} params")