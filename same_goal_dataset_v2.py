"""
same_goal_dataset_v2.py
=======================
Two agents navigate the SAME path to the SAME goal.
Agent 0: starts at path[0], navigates full path
Agent 1: starts at path[len//2], navigates second half

This creates real information asymmetry:
- Agent 1 has already seen the second half of the route
- Agent 1's observations are genuinely useful to Agent 0
- Communication has a real signal to learn from
"""
import json, os, random
import numpy as np
import torch
import h5py
import clip
from torch.utils.data import Dataset
from collections import defaultdict


class AsymmetricPathDataset(Dataset):
    def __init__(self, json_path, features_path, conn_graph,
                 split='train', max_len=20, max_candidates=10,
                 aug_instructions=True):
        self.feat_path      = features_path
        self.graph          = conn_graph
        self.max_len        = max_len
        self.max_candidates = max_candidates
        self.aug            = aug_instructions and split == 'train'

        with open(json_path) as f:
            self.episodes = json.load(f)

        # Only use episodes with path length >= 4 (need meaningful split)
        self.episodes = [ep for ep in self.episodes if len(ep['path']) >= 4]
        print(f"  {split}: {len(self.episodes)} episodes (len>=4)")

        # Pre-tokenise
        self.tokens = []
        for ep in self.episodes:
            ep_toks = []
            for instr in ep['instructions']:
                t = clip.tokenize([instr], truncate=True).squeeze(0)
                ep_toks.append(t)
            self.tokens.append(ep_toks)

        if not os.path.exists(features_path):
            raise FileNotFoundError(features_path)

    def __len__(self):
        return len(self.episodes)

    def _load_path(self, ep, path, tokens):
        """Load features for a specific path (may be sub-path)."""
        scan = ep['scan']
        T    = min(len(path) - 1, self.max_len)

        vp_list, cf_list, cm_list, gt_list = [], [], [], []
        with h5py.File(self.feat_path, 'r') as f:
            for t in range(T + 1):
                vp   = path[t]
                feat = f[scan][vp][:] if scan in f and vp in f[scan] \
                       else np.zeros((36, 512), np.float32)
                vp_list.append(feat)

            for t in range(T):
                cur, nxt = path[t], path[t+1]
                nbrs  = self.graph.get_neighbours(scan, cur)
                cands = nbrs + [cur]
                gt    = min(cands.index(nxt) if nxt in cands else 0,
                            self.max_candidates - 1)
                cf, mask = [], []
                for vp in cands[:self.max_candidates]:
                    feat = f[scan][vp][:] if scan in f and vp in f[scan] \
                           else np.zeros((36, 512), np.float32)
                    cf.append(feat); mask.append(True)
                while len(cf) < self.max_candidates:
                    cf.append(np.zeros((36, 512), np.float32))
                    mask.append(False)
                cf_list.append(np.stack(cf[:self.max_candidates]))
                cm_list.append(mask[:self.max_candidates])
                gt_list.append(gt)

        return {
            'tokens'     : tokens,
            'vp_features': torch.FloatTensor(np.stack(vp_list)),
            'cand_feats' : torch.FloatTensor(np.stack(cf_list)),
            'cand_masks' : torch.BoolTensor(np.array(cm_list)),
            'gt_actions' : torch.LongTensor(gt_list),
            'scan'       : scan,
            'path'       : path[:T+1],
        }

    def __getitem__(self, idx):
        ep   = self.episodes[idx]
        path = ep['path']
        toks = self.tokens[idx]
        tok  = toks[random.randint(0, len(toks)-1)] if self.aug else toks[0]

        # Agent 0: full path
        ep_a = self._load_path(ep, path, tok)

        # Agent 1: second half of path (starts at midpoint)
        mid  = len(path) // 2
        ep_b = self._load_path(ep, path[mid:], tok)

        # is_real flag: 1.0 for all real pairs
        is_real = torch.tensor(1.0)

        return ep_a, ep_b, is_real


def asymmetric_collate(batch):
    from multi_agent_utils import _collate_single
    eps_a   = [b[0] for b in batch]
    eps_b   = [b[1] for b in batch]
    is_real = torch.stack([b[2] for b in batch])
    return _collate_single(eps_a), _collate_single(eps_b), is_real
