"""
multi_agent_utils.py
====================
Shared dataset, collate functions, and helpers for multi-agent VLN training.
Imported by both multi_agent_train.py and multi_agent_train_v2.py.
"""

import os, random, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
import clip
import h5py


# ── CLIP text encoder (frozen) ─────────────────────────────────────────────────
def load_clip_text_encoder(device):
    model, _ = clip.load("ViT-B/32", device=device)
    model.float(); model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("CLIP ViT-B/32 loaded and frozen (fp32).")
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


# ── view aggregation ───────────────────────────────────────────────────────────
def encode_views(views):
    """(*, 36, 512) -> (*, 512) via attention-weighted pooling."""
    orig = views.shape[:-2]
    V = views.reshape(-1, 36, 512)
    q = V.mean(dim=1)
    w = F.softmax(torch.bmm(V, q.unsqueeze(-1)).squeeze(-1), dim=-1)
    return (w.unsqueeze(-1) * V).sum(dim=1).reshape(*orig, 512)


# ── LR schedule ───────────────────────────────────────────────────────────────
def get_scheduler(optimizer, warmup, total):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        p = (epoch - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * p)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── paired episode dataset ─────────────────────────────────────────────────────
class PairedR2RDataset(Dataset):
    """
    Each item = a PAIR of episodes from the same scan.
    Agent 0 navigates ep_a, Agent 1 navigates ep_b.
    """
    def __init__(self, json_path, features_path, conn_graph,
                 split='train', max_len=20, max_candidates=10,
                 aug_instructions=True):
        self.features_path  = features_path
        self.graph          = conn_graph
        self.max_len        = max_len
        self.max_candidates = max_candidates
        self.aug            = aug_instructions and (split == 'train')

        with open(json_path) as f:
            episodes = json.load(f)

        by_scan = defaultdict(list)
        for ep in episodes:
            by_scan[ep["scan"]].append(ep)

        self.pairs = []
        for scan, eps in by_scan.items():
            for i in range(0, len(eps) - 1, 2):
                self.pairs.append((eps[i], eps[i + 1]))
            if len(eps) % 2 == 1:
                self.pairs.append((eps[-1], eps[0]))

        print(f"  {split}: {len(episodes)} episodes → {len(self.pairs)} pairs")

        # Pre-tokenise all instructions
        self._tok_cache = {}
        all_eps = [ep for eps in by_scan.values() for ep in eps]
        for ep in all_eps:
            key = ep["scan"] + ep["path"][0]
            if key not in self._tok_cache:
                toks = []
                for instr in ep["instructions"]:
                    t = clip.tokenize([instr], truncate=True).squeeze(0)
                    toks.append(t)
                self._tok_cache[key] = toks

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features not found: {features_path}")

    def __len__(self):
        return len(self.pairs)

    def _load_episode(self, ep):
        scan = ep["scan"]
        path = ep["path"]
        T    = min(len(path) - 1, self.max_len)

        key  = scan + path[0]
        toks = self._tok_cache[key]
        tokens = toks[random.randint(0, len(toks)-1)] if self.aug else toks[0]

        vp_feats_list, cand_feats_list, cand_masks_list = [], [], []
        gt_actions_list, cand_vpids_list = [], []

        with h5py.File(self.features_path, 'r') as f:
            for t in range(T + 1):
                vp   = path[t]
                feat = f[scan][vp][:] if (scan in f and vp in f[scan]) \
                       else np.zeros((36, 512), dtype=np.float32)
                vp_feats_list.append(feat)

            for t in range(T):
                cur_vp  = path[t]
                next_vp = path[t + 1]
                neighbours = self.graph.get_neighbours(scan, cur_vp)
                candidates = neighbours + [cur_vp]
                cand_vps = candidates[:self.max_candidates]
                gt_act = min(
                    candidates.index(next_vp) if next_vp in candidates else 0,
                    self.max_candidates - 1
                )
                cf, mask = [], []

                for vp in cand_vps:
                    feat = f[scan][vp][:] if (scan in f and vp in f[scan]) \
                        else np.zeros((36, 512), dtype=np.float32)

                    cf.append(feat)
                    mask.append(True)

                while len(cf) < self.max_candidates:
                    cf.append(np.zeros((36, 512), dtype=np.float32))
                    mask.append(False)
                    cand_vps.append(None)
                cand_feats_list.append(np.stack(cf[:self.max_candidates]))
                cand_masks_list.append(mask[:self.max_candidates])
                cand_vpids_list.append(cand_vps[:self.max_candidates])
                gt_actions_list.append(gt_act)

        return {
            "tokens"     : tokens,
            "vp_features": torch.FloatTensor(np.stack(vp_feats_list)),
            "cand_feats" : torch.FloatTensor(np.stack(cand_feats_list)),
            "cand_masks" : torch.BoolTensor(np.array(cand_masks_list)),
            "gt_actions" : torch.LongTensor(gt_actions_list),
            "cand_vpids": cand_vpids_list,
            "scan"       : scan,
            "path"       : path[:T+1],
        }

    def __getitem__(self, idx):
        ep_a, ep_b = self.pairs[idx]
        return self._load_episode(ep_a), self._load_episode(ep_b)


def paired_collate_fn(batch):
    eps_a = [item[0] for item in batch]
    eps_b = [item[1] for item in batch]
    return _collate_single(eps_a), _collate_single(eps_b)


def _collate_single(batch):
    max_T = max(b["gt_actions"].shape[0] for b in batch)
    tokens, vp_feats = [], []
    cand_feats, cand_masks = [], []
    gt_acts, cand_vpids = [], []
    for b in batch:
        T = b["gt_actions"].shape[0]; pad = max_T - T
        C = b["cand_feats"].shape[1]
        tokens.append(b["tokens"])
        vf = b["vp_features"]
        if pad: vf = torch.cat([vf, torch.zeros(pad, 36, 512)])
        vp_feats.append(vf)
        cf = b["cand_feats"]
        if pad: cf = torch.cat([cf, torch.zeros(pad, C, 36, 512)])
        cand_feats.append(cf)
        cm = b["cand_masks"]
        if pad: cm = torch.cat([cm, torch.zeros(pad, C, dtype=torch.bool)])
        cand_masks.append(cm)
        ga = b["gt_actions"]
        if pad: ga = torch.cat([ga, torch.full((pad,), -1, dtype=torch.long)])
        gt_acts.append(ga)
        cv = b["cand_vpids"]

        if pad:
            cv += [[None] * C for _ in range(pad)]

        cand_vpids.append(cv)
    return {
        "tokens"     : torch.stack(tokens),
        "vp_features": torch.stack(vp_feats),
        "cand_feats" : torch.stack(cand_feats),
        "cand_masks" : torch.stack(cand_masks),
        "gt_actions" : torch.stack(gt_acts),
        "cand_vpids": cand_vpids,
        
    }