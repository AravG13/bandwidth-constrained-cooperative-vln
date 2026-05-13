"""
evaluate.py
===========
Evaluates a trained VLN agent on val_seen / val_unseen.
Computes SR (Success Rate) and SPL (Success weighted by Path Length).

SR  = fraction of episodes where agent ends within 3m of goal viewpoint
SPL = mean(SR_i * shortest_path_len / max(pred_path_len, shortest_path_len))

Usage:
    python3 evaluate.py --split val_seen
    python3 evaluate.py --split val_unseen
    python3 evaluate.py --split val_unseen --ckpt results/checkpoints/best_agent.pt
"""

import os, json, argparse, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.connectivity import ConnectivityGraph
from r2r_dataset import R2RDataset, collate_fn
from models.vln_modules import CooperativeVLNAgent

BASE_DIR  = os.path.expanduser("~/vln_project")
CONN_DIR  = os.path.join(BASE_DIR, "Matterport3DSimulator/connectivity")
DATA_DIR  = os.path.join(BASE_DIR, "data/r2r")
FEAT_PATH = os.path.join(BASE_DIR, "data/features/CLIP-ViT-B-32-views.hdf5")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SUCCESS_DIST = 3.0   # metres — standard R2R threshold

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      default="val_unseen",
                   choices=["val_seen", "val_unseen", "train"])
    p.add_argument("--ckpt",       default="results/checkpoints/best_agent.pt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len",    type=int, default=20)
    p.add_argument("--max_candidates", type=int, default=10)
    return p.parse_args()

# ── CLIP text encoder (same as train.py) ──────────────────────────────────────
def load_clip_text_encoder(device):
    import clip as clip_lib
    model, _ = clip_lib.load("ViT-B/32", device=device)
    model.float(); model.eval()
    for p in model.parameters():
        p.requires_grad = False
    def encode(tokens):
        tokens = tokens.to(device).long()
        with torch.no_grad():
            x = model.token_embedding(tokens)
            x = x + model.positional_embedding
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)
            x = model.ln_final(x)
        return x
    return encode

# ── viewpoint distance via connectivity graph ─────────────────────────────────
def load_distances(conn_dir):
    """
    Precompute shortest-path distances between all viewpoint pairs per scan.
    Uses BFS on the connectivity graph.
    Returns dict: distances[scan][vp_a][vp_b] = metres (float)
    """
    import collections
    distances = {}
    for fname in os.listdir(conn_dir):
        if not fname.endswith("_connectivity.json"):
            continue
        scan = fname.replace("_connectivity.json", "")
        conn = json.load(open(os.path.join(conn_dir, fname)))

        # Build adjacency: vp -> list of (neighbour_vp, distance_metres)
        positions = {}   # vp -> (x, y, z)
        adj = collections.defaultdict(list)
        for node in conn:
            vp = node["image_id"]
            positions[vp] = node["pose"][3], node["pose"][7], node["pose"][11]
        for node in conn:
            vp = node["image_id"]
            for i, nb in enumerate(node.get("unobstructed", [])):
                if nb:
                    nb_vp = conn[i]["image_id"]
                    p1 = np.array(positions[vp])
                    p2 = np.array(positions[nb_vp])
                    dist = float(np.linalg.norm(p1 - p2))
                    adj[vp].append((nb_vp, dist))

        # BFS from each viewpoint
        scan_dist = {}
        for start in positions:
            dist_map = {start: 0.0}
            queue = collections.deque([start])
            while queue:
                cur = queue.popleft()
                for nb, d in adj[cur]:
                    if nb not in dist_map:
                        dist_map[nb] = dist_map[cur] + d
                        queue.append(nb)
            scan_dist[start] = dist_map
        distances[scan] = scan_dist
    return distances

# ── greedy agent rollout ───────────────────────────────────────────────────────

def encode_views(views):
    """(*, 36, 512) -> (*, 512) via attention-weighted pooling."""
    orig = views.shape[:-2]
    V = views.reshape(-1, 36, 512)
    q = V.mean(dim=1)
    w = torch.nn.functional.softmax(torch.bmm(V, q.unsqueeze(-1)).squeeze(-1), dim=-1)
    return (w.unsqueeze(-1) * V).sum(dim=1).reshape(*orig, 512)

@torch.no_grad()
def run_greedy(agent, encode_lang, batch, device, max_steps):
    """
    Run agent greedily (argmax actions) for up to max_steps.
    Returns predicted path indices (which candidate was chosen at each step).
    Shape: (B, max_steps) — padded with -1 after episode ends.
    """
    tokens     = batch["tokens"].to(device)
    vp_feats   = batch["vp_features"].to(device)
    cand_feats = batch["cand_feats"].to(device)
    cand_masks = batch["cand_masks"].to(device)
    if cand_feats.dim() == 5:
        cand_feats = encode_views(cand_feats)
    if vp_feats.dim() == 4:
        vp_feats = encode_views(vp_feats)
    B, T, C, _ = cand_feats.shape

    lang_tokens = encode_lang(tokens)
    lang_mask   = (tokens == 0).to(device)
    h           = agent.agent_gru.init_hidden(B, device)
    prev_action = torch.zeros(B, dtype=torch.long, device=device)
    W1          = agent.nav_head.mlp[0]

    pred_actions = torch.full((B, T), -1, dtype=torch.long)

    for t in range(T):
        obs   = vp_feats[:, t, :]
        cands = cand_feats[:, t, :, :]
        mask  = cand_masks[:, t, :]

        context, _      = agent.cross_attn(obs, lang_tokens, lang_mask)
        aggregated_msgs = agent.msg_agg(h, None)
        h               = agent.agent_gru(context, aggregated_msgs, prev_action, h)

        h_proj     = W1(h)
        cands_proj = W1(cands.reshape(B * C, 512)).reshape(B, C, -1)
        scores     = torch.bmm(cands_proj, h_proj.unsqueeze(-1)).squeeze(-1)
        scores     = scores.masked_fill(~mask, float("-inf"))

        pred = scores.argmax(dim=-1)   # (B,) greedy
        pred_actions[:, t] = pred.cpu()
        prev_action = pred

    return pred_actions   # (B, T)

# ── SR / SPL computation ───────────────────────────────────────────────────────
def compute_sr_spl(episodes, pred_paths, distances, success_dist=3.0):
    """
    episodes  : list of R2R episode dicts (with 'scan', 'path')
    pred_paths: list of viewpoint-id lists (predicted path taken)
    distances : precomputed BFS distances
    """
    sr_list, spl_list = [], []

    for ep, pred_path in zip(episodes, pred_paths):
        scan        = ep["scan"]
        gt_path     = ep["path"]
        goal_vp     = gt_path[-1]
        start_vp    = gt_path[0]
        pred_end_vp = pred_path[-1] if pred_path else start_vp

        # Distance from predicted endpoint to goal
        try:
            end_dist = distances[scan][pred_end_vp][goal_vp]
        except KeyError:
            end_dist = float("inf")

        success = float(end_dist <= success_dist)

        # Shortest-path distance (gt path length approximation)
        try:
            shortest = distances[scan][start_vp][goal_vp]
        except KeyError:
            shortest = 1.0

        # Predicted path length (sum of step distances)
        pred_len = 0.0
        for i in range(len(pred_path) - 1):
            try:
                pred_len += distances[scan][pred_path[i]][pred_path[i+1]]
            except KeyError:
                pred_len += 3.0   # fallback

        spl = success * shortest / max(shortest, pred_len, 1e-6)

        sr_list.append(success)
        spl_list.append(spl)

    return float(np.mean(sr_list)), float(np.mean(spl_list))

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("Loading connectivity graph...")
    graph = ConnectivityGraph(CONN_DIR)

    print("Precomputing distances (BFS)...")
    distances = load_distances(CONN_DIR)
    print(f"  Distances loaded for {len(distances)} scans.")

    print(f"Loading {args.split} episodes...")
    json_path = os.path.join(DATA_DIR, f"R2R_{args.split}.json")
    with open(json_path) as f:
        all_episodes = json.load(f)

    ds = R2RDataset(json_path, FEAT_PATH, graph,
                    split=args.split,
                    max_len=args.max_len,
                    max_candidates=args.max_candidates)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location=DEVICE)
    agent = CooperativeVLNAgent(512, 512,
                                ckpt["args"]["hidden_dim"],
                                ckpt["args"]["action_dim"],
                                128).to(DEVICE)
    agent.load_state_dict(ckpt["model"], strict=False)
    agent.eval()
    print(f"  Loaded epoch {ckpt['epoch']} checkpoint.")

    encode_lang = load_clip_text_encoder(DEVICE)

    # ── rollout all episodes ──────────────────────────────────────────────────
    all_pred_paths = []
    ep_idx = 0

    for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
        B        = batch["tokens"].shape[0]
        paths    = batch["paths"] if "paths" in batch else [[] for _ in range(B)]
        pred_act = run_greedy(agent, encode_lang, batch, DEVICE, args.max_len)

        for b in range(B):
            ep   = all_episodes[ep_idx]
            path = ep["path"]
            T    = min(len(path) - 1, args.max_len)

            # Reconstruct predicted viewpoint path from action indices
            pred_vp_path = [path[0]]
            scan = ep["scan"]
            for t in range(T):
                act = pred_act[b, t].item()
                if act < 0:
                    break
                cur_vp     = pred_vp_path[-1]
                neighbours = graph.get_neighbours(scan, cur_vp)
                candidates = neighbours + [cur_vp]   # last = STOP
                if act < len(candidates):
                    next_vp = candidates[act]
                    if next_vp == cur_vp:   # STOP — stay in place
                        break
                    pred_vp_path.append(next_vp)
                else:
                    break  # invalid action = implicit stop

            all_pred_paths.append(pred_vp_path)
            ep_idx += 1

    # ── compute metrics ───────────────────────────────────────────────────────
    sr, spl = compute_sr_spl(all_episodes, all_pred_paths, distances, SUCCESS_DIST)

    print(f"\n{'='*40}")
    print(f"Split      : {args.split}")
    print(f"Episodes   : {len(all_episodes)}")
    print(f"SR         : {sr*100:.1f}%")
    print(f"SPL        : {spl*100:.1f}%")
    print(f"{'='*40}")
    print(f"\nBaseline to beat (R2R seq2seq):")
    print(f"  val_seen   SR ~39%  SPL ~33%")
    print(f"  val_unseen SR ~22%  SPL ~18%")

if __name__ == "__main__":
    main()
