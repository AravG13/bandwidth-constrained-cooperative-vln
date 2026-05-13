"""
hindsight_partner_labels.py
============================
Hindsight Communication Gating with Partner-Correct Labels.

Communication utility labels:
  self wrong, partner wrong  → label=0 (sending wouldn't help)
  self wrong, partner right  → label=1 (partner could have helped me)
  self right, partner wrong  → label=0 (I didn't need help)
  self right, partner right  → label=0 (didn't need help)


Usage:
    python3 hindsight_gate_train.py --overfit_test --no_wandb \
        --nav_epochs 0 --gate_epochs 15 --joint_epochs 5 \
        --init_from results/checkpoints_fixed/best_agent.pt
    
    python3 hindsight_gate_train.py \
        --nav_epochs 0 --gate_epochs 20 --joint_epochs 10 \
        --budget 3 \
        --init_from results/checkpoints_fixed/best_agent.pt \
        --save_dir results/checkpoints_partner_gate
"""

import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.connectivity import ConnectivityGraph
from r2r_dataset import R2RDataset, collate_fn
from same_goal_dataset import SameGoalPairedDataset, same_goal_collate
from multi_agent_utils import load_clip_text_encoder, encode_views
from models.vln_modules import CooperativeVLNAgent

BASE_DIR  = os.path.expanduser("~/vln_project")
CONN_DIR  = os.path.join(BASE_DIR, "Matterport3DSimulator/connectivity")
DATA_DIR  = os.path.join(BASE_DIR, "data/r2r")
FEAT_PATH = os.path.join(BASE_DIR, "data/features/CLIP-ViT-B-32-views.hdf5")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nav_epochs",     type=int,   default=0,
                   help="Phase 1: 0 = skip (use init_from directly)")
    p.add_argument("--gate_epochs",    type=int,   default=20)
    p.add_argument("--joint_epochs",   type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--hidden_dim",     type=int,   default=512)
    p.add_argument("--action_dim",     type=int,   default=36)
    p.add_argument("--max_candidates", type=int,   default=10)
    p.add_argument("--max_len",        type=int,   default=20)
    p.add_argument("--budget",         type=int,   default=3)
    p.add_argument("--save_dir",       type=str,
                   default=os.path.join(BASE_DIR, "results/checkpoints_partner_gate"))
    p.add_argument("--init_from",      type=str,
                   default="results/checkpoints_fixed/best_agent.pt")
    p.add_argument("--overfit_test",   action="store_true")
    p.add_argument("--no_wandb",       action="store_true")
    return p.parse_args()


def get_scheduler(optimizer, warmup, total):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        p = (epoch - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * p)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def single_agent_rollout(agent, encode_lang, vp_feats, cand_feats,
                         cand_masks, gt_actions, lang_tokens, lang_mask, device):
    """
    Run one agent for a full episode. Returns scores, preds, hidden states.
    No messages — used for label collection.
    """
    B, T, C, _ = cand_feats.shape
    W1 = agent.nav_head.mlp[0]
    h  = agent.agent_gru.init_hidden(B, device)
    prev = torch.zeros(B, dtype=torch.long, device=device)

    all_scores, all_preds, all_hidden = [], [], []

    for t in range(T):
        obs   = vp_feats[:, t, :]
        cands = cand_feats[:, t, :, :]
        mask  = cand_masks[:, t, :]
        gt    = gt_actions[:, t]

        ctx, _ = agent.cross_attn(obs, lang_tokens, lang_mask)
        msgs   = agent.msg_agg(h, None)
        h      = agent.agent_gru(ctx, msgs, prev, h)

        hp  = W1(h)
        cp  = W1(cands.reshape(B * C, 512)).reshape(B, C, -1)
        sc  = torch.bmm(cp, hp.unsqueeze(-1)).squeeze(-1)
        sc  = sc.masked_fill(~mask, float("-inf"))

        all_scores.append(sc)
        all_preds.append(sc.argmax(dim=-1))    
        all_hidden.append(h.detach())            

        prev = gt.clamp(min=0)

    return all_scores, all_preds, all_hidden

def collect_partner_labels(agent0, agent1, encode_lang, loader,
                           device, budget, max_batches=100):

    agent0.eval(); agent1.eval()

    all_h0, all_l0 = [], []
    all_h1, all_l1 = [], []
    n = 0

    with torch.no_grad():
        for batch_a, batch_b, is_real in tqdm(loader,
                                               desc="Collecting partner labels",
                                               leave=False):
            
            t0  = batch_a["tokens"].to(device)
            vp0 = encode_views(batch_a["vp_features"].to(device))
            cf0 = encode_views(batch_a["cand_feats"].to(device))
            cm0 = batch_a["cand_masks"].to(device)
            gt0 = batch_a["gt_actions"].to(device)
            lm0 = encode_lang(t0)
            lmask0 = (t0 == 0).to(device)

          
            t1  = batch_b["tokens"].to(device)
            vp1 = encode_views(batch_b["vp_features"].to(device))
            cf1 = encode_views(batch_b["cand_feats"].to(device))
            cm1 = batch_b["cand_masks"].to(device)
            gt1 = batch_b["gt_actions"].to(device)
            lm1 = encode_lang(t1)
            lmask1 = (t1 == 0).to(device)

            T = min(cf0.shape[1], cf1.shape[1])

            sc0, pred0, h0 = single_agent_rollout(
                agent0, encode_lang, vp0, cf0, cm0, gt0, lm0, lmask0, device)
            sc1, pred1, h1 = single_agent_rollout(
                agent1, encode_lang, vp1, cf1, cm1, gt1, lm1, lmask1, device)

            for t in range(T):
                valid0 = gt0[:, t] >= 0
                valid1 = gt1[:, t] >= 0

                self0_wrong     = (pred0[t] != gt0[:, t]).float()
                partner1_correct = (pred1[t] == gt1[:, t]).float()
                self1_wrong     = (pred1[t] != gt1[:, t]).float()
                partner0_correct = (pred0[t] == gt0[:, t]).float()

                label0 = self0_wrong * partner1_correct * valid0.float()
                label1 = self1_wrong * partner0_correct * valid1.float()

                all_h0.append(h0[t])   
                all_l0.append(label0) 
                all_h1.append(h1[t])
                all_l1.append(label1)

            n += 1
            if n >= max_batches:
                break

    hiddens0 = torch.cat(all_h0, dim=0)
    labels0  = torch.cat(all_l0, dim=0)
    hiddens1 = torch.cat(all_h1, dim=0)
    labels1  = torch.cat(all_l1, dim=0)

    pos0 = labels0.mean().item()
    pos1 = labels1.mean().item()
    print(f"  Agent0 labels: {len(hiddens0):,} samples, "
          f"positive rate: {pos0*100:.1f}%")
    print(f"  Agent1 labels: {len(hiddens1):,} samples, "
          f"positive rate: {pos1*100:.1f}%")
    print(f"  Interpretation: gate fires when self wrong AND partner right")

    return hiddens0, labels0, hiddens1, labels1

def train_gate_bce(agent, hiddens, labels, device, epochs=20, lr=1e-4):
    """
    Trainint comm_gate with BCE + pos_weight to handle class imbalance.
    """
    for p in agent.parameters():
        p.requires_grad = False
    for p in agent.comm_gate.parameters():
        p.requires_grad = True

    pos_rate   = labels.float().mean().item()
    neg_rate   = 1.0 - pos_rate
    pos_weight = torch.tensor([neg_rate / max(pos_rate, 1e-6)], device=device)
    print(f"  pos_weight={pos_weight.item():.1f} "
          f"(pos={pos_rate*100:.1f}%, neg={neg_rate*100:.1f}%)")

    optimizer = torch.optim.Adam(agent.comm_gate.parameters(), lr=lr)
    budget_sig = torch.full((1, 1), 0.5, device=device)
    dataset    = TensorDataset(hiddens.to(device), labels.float().to(device))
    loader     = DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"Training gate BCE for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        tl, ta, n = 0., 0., 0
        for h_batch, l_batch in loader:
            B        = h_batch.shape[0]
            budget_b = budget_sig.expand(B, 1)

            _, p_send, _ = agent.comm_gate(h_batch, budget_b, deterministic=False)
            loss = F.binary_cross_entropy(p_send, l_batch)
            acc  = ((p_send >= 0.5).float() == l_batch).float().mean().item()

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item(); ta += acc; n += 1

        print(f"  Gate ep{epoch:2d}: BCE={tl/max(n,1):.4f} "
              f"acc={ta/max(n,1):.3f} p_send={p_send.mean():.3f}")

    for p in agent.parameters():
        p.requires_grad = True



def nav_forward(agent, encode_lang, batch, device):
    tokens     = batch["tokens"].to(device)
    vp_feats   = encode_views(batch["vp_features"].to(device))
    cand_feats = encode_views(batch["cand_feats"].to(device))
    cand_masks = batch["cand_masks"].to(device)
    gt_actions = batch["gt_actions"].to(device)
    B, T, C, _ = cand_feats.shape

    lang = encode_lang(tokens)
    lmask = (tokens == 0).to(device)
    W1 = agent.nav_head.mlp[0]
    h  = agent.agent_gru.init_hidden(B, device)
    prev = torch.zeros(B, dtype=torch.long, device=device)

    all_sc, all_gt = [], []
    for t in range(T):
        ctx, _ = agent.cross_attn(vp_feats[:, t], lang, lmask)
        msgs   = agent.msg_agg(h, None)
        h      = agent.agent_gru(ctx, msgs, prev, h)
        hp     = W1(h)
        cp     = W1(cand_feats[:, t].reshape(B*C, 512)).reshape(B, C, -1)
        sc     = torch.bmm(cp, hp.unsqueeze(-1)).squeeze(-1)
        sc     = sc.masked_fill(~cand_masks[:, t], float("-inf"))
        all_sc.append(sc); all_gt.append(gt_actions[:, t])
        prev = gt_actions[:, t].clamp(min=0)

    sf = torch.stack(all_sc).view(T*B, C)
    gf = torch.stack(all_gt).view(T*B)
    valid = gf >= 0
    if valid.sum() == 0:
        return torch.tensor(0., device=device, requires_grad=True), 0.
    loss = F.cross_entropy(sf[valid], gf[valid])
    acc  = (sf[valid].argmax(-1) == gf[valid]).float().mean().item()
    return loss, acc



def joint_forward(agent0, agent1, encode_lang, batch_a, batch_b,
                  is_real, device, budget):
    def prep(b):
        t   = b["tokens"].to(device)
        vp  = encode_views(b["vp_features"].to(device))
        cf  = encode_views(b["cand_feats"].to(device))
        cm  = b["cand_masks"].to(device)
        gt  = b["gt_actions"].to(device)
        lm  = encode_lang(t)
        lmk = (t == 0).to(device)
        return vp, cf, cm, gt, lm, lmk

    vp0,cf0,cm0,gt0,lm0,lmask0 = prep(batch_a)
    vp1,cf1,cm1,gt1,lm1,lmask1 = prep(batch_b)
    is_real = is_real.to(device)
    B, T, C, _ = cf0.shape
    W0 = agent0.nav_head.mlp[0]
    W1 = agent1.nav_head.mlp[0]
    h0 = agent0.agent_gru.init_hidden(B, device)
    h1 = agent1.agent_gru.init_hidden(B, device)
    p0 = torch.zeros(B, dtype=torch.long, device=device)
    p1 = torch.zeros(B, dtype=torch.long, device=device)
    s0 = torch.zeros(B, device=device)
    s1 = torch.zeros(B, device=device)
    m0 = m1 = None
    nav_l, nav_g = [], []

    for t in range(T):
        br0 = (1 - s0 / max(budget, 1)).clamp(0, 1).unsqueeze(-1)
        br1 = (1 - s1 / max(budget, 1)).clamp(0, 1).unsqueeze(-1)

        ctx0, _ = agent0.cross_attn(vp0[:, t], lm0, lmask0)
        agg0    = agent0.msg_agg(h0, m0.unsqueeze(1) if m0 is not None else None)
        h0      = agent0.agent_gru(ctx0, agg0, p0, h0)
        sc0     = torch.bmm(W0(cf0[:, t].reshape(B*C,512)).reshape(B,C,-1),
                            W0(h0).unsqueeze(-1)).squeeze(-1)
        sc0     = sc0.masked_fill(~cm0[:, t], float('-inf'))

        ctx1, _ = agent1.cross_attn(vp1[:, t], lm1, lmask1)
        agg1    = agent1.msg_agg(h1, m1.unsqueeze(1) if m1 is not None else None)
        h1      = agent1.agent_gru(ctx1, agg1, p1, h1)
        sc1     = torch.bmm(W1(cf1[:, t].reshape(B*C,512)).reshape(B,C,-1),
                            W1(h1).unsqueeze(-1)).squeeze(-1)
        sc1     = sc1.masked_fill(~cm1[:, t], float('-inf'))


        _, pg0, _ = agent0.comm_gate(h0, br0, deterministic=False)
        _, pg1, _ = agent1.comm_gate(h1, br1, deterministic=False)
        g0 = torch.bernoulli(pg0) * (s0 < budget).float() * is_real
        g1 = torch.bernoulli(pg1) * (s1 < budget).float() * is_real

        m1 = ctx0 * g0.unsqueeze(-1)
        m0 = ctx1 * g1.unsqueeze(-1)
        s0 += g0; s1 += g1

        nav_l.extend([sc0, sc1])
        nav_g.extend([gt0[:, t], gt1[:, t]])
        p0 = gt0[:, t].clamp(min=0)
        p1 = gt1[:, t].clamp(min=0)

    lc = torch.cat(nav_l); gc = torch.cat(nav_g)
    valid = gc >= 0
    if valid.sum() == 0:
        return None, 0., 0.
    loss = F.cross_entropy(lc[valid], gc[valid])
    acc  = (lc[valid].argmax(-1) == gc[valid]).float().mean().item()
    return loss, acc, (s0.mean() + s1.mean()).item() / 2

def main():
    import torch.nn as nn
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="vln-partner-gate", config=vars(args),
                       name=f"partner_b{args.budget}")
        except Exception as e:
            print(f"wandb: {e}"); use_wandb = False

    graph = ConnectivityGraph(CONN_DIR)
    kw    = dict(features_path=FEAT_PATH, conn_graph=graph,
                 max_len=args.max_len, max_candidates=args.max_candidates)

    print("Loading datasets...")
    train_single = R2RDataset(
        os.path.join(DATA_DIR, "R2R_train.json"),
        split="train", aug_instructions=True, **kw)
    val_single   = R2RDataset(
        os.path.join(DATA_DIR, "R2R_val_seen.json"),
        split="val_seen", aug_instructions=False, **kw)
    train_paired = SameGoalPairedDataset(
        os.path.join(DATA_DIR, "R2R_train.json"),
        split="train", aug_instructions=True, **kw)
    val_paired   = SameGoalPairedDataset(
        os.path.join(DATA_DIR, "R2R_val_seen.json"),
        split="val_seen", aug_instructions=False, **kw)

    if args.overfit_test:
        from torch.utils.data import Subset
        train_single = Subset(train_single, list(range(20)))
        val_single   = Subset(val_single,   list(range(20)))
        train_paired = Subset(train_paired, list(range(20)))
        val_paired   = Subset(val_paired,   list(range(20)))
        print("OVERFIT TEST — 20 episodes")

    sl_train = DataLoader(train_single, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate_fn,
                          num_workers=4, pin_memory=True)
    sl_val   = DataLoader(val_single,   batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=2)
    pl_train = DataLoader(train_paired, batch_size=args.batch_size // 2,
                          shuffle=True, collate_fn=same_goal_collate,
                          num_workers=4, pin_memory=True)
    pl_val   = DataLoader(val_paired,   batch_size=args.batch_size // 2,
                          shuffle=False, collate_fn=same_goal_collate,
                          num_workers=2)

    mk = dict(v_dim=512, l_dim=512, hidden_dim=args.hidden_dim,
              action_dim=args.action_dim, gate_hidden=128)
    agent0 = CooperativeVLNAgent(**mk).to(DEVICE)
    agent1 = CooperativeVLNAgent(**mk).to(DEVICE)

    if args.init_from and os.path.exists(args.init_from):
        ckpt  = torch.load(args.init_from, map_location=DEVICE)
        state = ckpt.get("model", ckpt.get("agent0"))
        if state:
            agent0.load_state_dict(state, strict=False)
            agent1.load_state_dict(state, strict=False)
            print(f"Warm-started from {args.init_from} "
                  f"(epoch {ckpt.get('epoch','?')})")

    encode_lang = load_clip_text_encoder(DEVICE)

  #Phase 1:

  
    if args.nav_epochs > 0:
        print(f"\n{'='*55}")
        print(f"PHASE 1: Navigation ({args.nav_epochs} epochs)")
        print(f"{'='*55}")
        p1_opt   = torch.optim.AdamW(
            list(agent0.parameters()) + list(agent1.parameters()),
            lr=args.lr, weight_decay=1e-4)
        p1_sched = get_scheduler(p1_opt, warmup=3, total=args.nav_epochs)
        best_p1  = float("inf")

        for epoch in range(1, args.nav_epochs + 1):
            agent0.train()
            tl, ta, n = 0., 0., 0
            for batch in tqdm(sl_train, desc=f"P1 {epoch} train", leave=False):
                loss, acc = nav_forward(agent0, encode_lang, batch, DEVICE)
                p1_opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent0.parameters(), 5.0)
                p1_opt.step()
                tl += loss.item(); ta += acc; n += 1
            agent0.eval()
            vl, va, nv = 0., 0., 0
            with torch.no_grad():
                for batch in tqdm(sl_val, desc=f"P1 {epoch} val", leave=False):
                    loss, acc = nav_forward(agent0, encode_lang, batch, DEVICE)
                    vl += loss.item(); va += acc; nv += 1
            p1_sched.step()
            print(f"P1 Ep{epoch:2d} | train={tl/max(n,1):.4f} "
                  f"acc={ta/max(n,1):.3f} | val={vl/max(nv,1):.4f} "
                  f"acc={va/max(nv,1):.3f}")
            if vl/max(nv,1) < best_p1:
                best_p1 = vl/max(nv,1)
                torch.save({"epoch":epoch,"agent0":agent0.state_dict(),
                            "val_loss":best_p1,"args":vars(args)},
                           os.path.join(args.save_dir,"phase1_best.pt"))
        agent1.load_state_dict(agent0.state_dict())
    else:
        print("Phase 1 skipped — using init_from weights directly")
        agent1.load_state_dict(agent0.state_dict())


    # Phase 2: Partner-correct gate labels + BCE training

    print(f"\n{'='*55}")
    print(f"PHASE 2: Partner-correct gate training ({args.gate_epochs} epochs)")
    print(f"{'='*55}")
    print("Key: label=1 ONLY when self wrong AND partner correct")

    max_b = 5 if args.overfit_test else 100
    h0, l0, h1, l1 = collect_partner_labels(
        agent0, agent1, encode_lang, pl_train, DEVICE, args.budget, max_b)

    print("\nTraining agent0 gate...")
    train_gate_bce(agent0, h0, l0, DEVICE, epochs=args.gate_epochs, lr=1e-4)
    print("\nTraining agent1 gate...")
    train_gate_bce(agent1, h1, l1, DEVICE, epochs=args.gate_epochs, lr=1e-4)

    # Validate gate selectivity
    agent0.eval(); agent1.eval()
    with torch.no_grad():
        btest = torch.full((min(200, len(h0)), 1), 0.5, device=DEVICE)
        _, ps0, _ = agent0.comm_gate(h0[:200].to(DEVICE), btest, deterministic=False)
        _, ps1, _ = agent1.comm_gate(h1[:200].to(DEVICE), btest, deterministic=False)
    print(f"\nGate validation:")
    print(f"  Agent0: avg p_send={ps0.mean():.3f} "
          f"fire_rate={(ps0>=0.5).float().mean():.3f}")
    print(f"  Agent1: avg p_send={ps1.mean():.3f} "
          f"fire_rate={(ps1>=0.5).float().mean():.3f}")
    print(f"  Target p_send ≈ pos_rate "
          f"(~{l0.mean().item()*100:.1f}% / {l1.mean().item()*100:.1f}%)")

    torch.save({"agent0": agent0.state_dict(), "agent1": agent1.state_dict(),
                "phase": 2, "args": vars(args)},
               os.path.join(args.save_dir, "phase2_best.pt"))


    # Phase 3: Joint fine-tuning
  
    if args.joint_epochs > 0:
        print(f"\n{'='*55}")
        print(f"PHASE 3: Joint fine-tuning ({args.joint_epochs} epochs)")
        print(f"{'='*55}")

        p3_opt   = torch.optim.AdamW(
            list(agent0.parameters()) + list(agent1.parameters()),
            lr=args.lr * 0.2, weight_decay=1e-4)
        p3_sched = get_scheduler(p3_opt, warmup=2, total=args.joint_epochs)
        best_p3  = float("inf")

        for epoch in range(1, args.joint_epochs + 1):
            agent0.train(); agent1.train()
            tl, ta, ts, n = 0., 0., 0., 0
            for ba, bb, is_real in tqdm(pl_train,
                                        desc=f"P3 {epoch} train", leave=False):
                res = joint_forward(agent0, agent1, encode_lang,
                                    ba, bb, is_real, DEVICE, args.budget)
                if res[0] is None: continue
                loss, acc, sends = res
                p3_opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent0.parameters())+list(agent1.parameters()), 5.0)
                p3_opt.step()
                tl += loss.item(); ta += acc; ts += sends; n += 1

            agent0.eval(); agent1.eval()
            vl, va, vs, nv = 0., 0., 0., 0
            with torch.no_grad():
                for ba, bb, is_real in tqdm(pl_val,
                                            desc=f"P3 {epoch} val", leave=False):
                    res = joint_forward(agent0, agent1, encode_lang,
                                        ba, bb, is_real, DEVICE, args.budget)
                    if res[0] is None: continue
                    loss, acc, sends = res
                    vl += loss.item(); va += acc; vs += sends; nv += 1

            p3_sched.step()
            d  = max(n, 1); dv = max(nv, 1)
            print(f"P3 Ep{epoch:2d} | train={tl/d:.4f} acc={ta/d:.3f} "
                  f"sends={ts/d:.2f} | val={vl/dv:.4f} acc={va/dv:.3f} "
                  f"sends={vs/dv:.2f}")

            if vl/dv < best_p3:
                best_p3 = vl/dv
                torch.save({"epoch": epoch, "phase": 3,
                            "agent0": agent0.state_dict(),
                            "agent1": agent1.state_dict(),
                            "val_loss": vl/dv, "val_acc": va/dv,
                            "args": vars(args)},
                           os.path.join(args.save_dir,
                                        f"best_budget{args.budget}.pt"))
                print(f"  ✓ P3 saved (val={vl/dv:.4f})")

    print("\nAll phases complete.")
    if use_wandb:
        import wandb; wandb.finish()

    os.system(f"python3 evaluate_multiagent.py --split val_unseen "
              f"--budget {args.budget} --ckpt_dir {args.save_dir}")


if __name__ == "__main__":
    main()
