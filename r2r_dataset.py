"""
r2r_dataset.py  —  R2R with full 36-view candidate features (no mean-pool)
"""
import json, os, random
import numpy as np
import torch
from torch.utils.data import Dataset
import clip
import h5py
from utils.connectivity import ConnectivityGraph


class R2RDataset(Dataset):
    def __init__(self, json_path, features_path, conn_graph,
                 split='train', max_len=20, max_candidates=10,
                 aug_instructions=True):
        "
        self.features_path  = features_path
        self.graph          = conn_graph
        self.split          = split
        self.max_len        = max_len
        self.max_candidates = max_candidates
        self.aug_instructions = aug_instructions and (split == 'train')

        print(f"Loading {split} episodes from {json_path}...")
        with open(json_path) as f:
            self.episodes = json.load(f)
        print(f"  {len(self.episodes)} episodes loaded")

        print("  Pre-tokenising instructions...")
        self.tokens = []  
        for ep in self.episodes:
            ep_toks = []
            for instr in ep["instructions"]:
                tok = clip.tokenize([instr], truncate=True)
                ep_toks.append(tok.squeeze(0))   
            self.tokens.append(ep_toks)
        print(f"  Done.")

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features not found: {features_path}")
        with h5py.File(features_path, 'r') as f:
            print(f"  Features file: {len(list(f.keys()))} scans")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep   = self.episodes[idx]
        scan = ep["scan"]
        path = ep["path"]
        T    = min(len(path) - 1, self.max_len)

        
        if self.aug_instructions:
            instr_idx = random.randint(0, len(self.tokens[idx]) - 1)
        else:
            instr_idx = 0
        tokens = self.tokens[idx][instr_idx]   

        candidate_features = []   
        gt_actions         = []
        candidate_masks    = []
        vp_features        = []   

        with h5py.File(self.features_path, 'r') as f:
           
            for t in range(T + 1):
                vp = path[t]
                if scan in f and vp in f[scan]:
                    feat = f[scan][vp][:]             
                else:
                    feat = np.zeros((36, 512), dtype=np.float32)
                vp_features.append(feat)

            for t in range(T):
                current_vp = path[t]
                next_vp    = path[t + 1]

                neighbours = self.graph.get_neighbours(scan, current_vp)
                candidates = neighbours + [current_vp]   

                gt_act = candidates.index(next_vp) if next_vp in candidates else 0
                gt_act = min(gt_act, self.max_candidates - 1)

                cand_feats, mask = [], []
                for vp in candidates[:self.max_candidates]:
                    if scan in f and vp in f[scan]:
                        feat = f[scan][vp][:]         
                    else:
                        feat = np.zeros((36, 512), dtype=np.float32)
                    cand_feats.append(feat)
                    mask.append(True)

                while len(cand_feats) < self.max_candidates:
                    cand_feats.append(np.zeros((36, 512), dtype=np.float32))
                    mask.append(False)

                candidate_features.append(np.stack(cand_feats[:self.max_candidates]))
                candidate_masks.append(mask[:self.max_candidates])
                gt_actions.append(gt_act)

        return {
            "tokens"     : tokens,
          
            "vp_features": torch.FloatTensor(np.stack(vp_features)),
         
            "cand_feats" : torch.FloatTensor(np.stack(candidate_features)),
            "cand_masks" : torch.BoolTensor(np.array(candidate_masks)),
            "gt_actions" : torch.LongTensor(gt_actions),
            "scan"       : scan,
            "path"       : path[:T+1],
        }


def collate_fn(batch):
    max_T = max(b["gt_actions"].shape[0] for b in batch)
    tokens, vp_feats, cand_feats, cand_masks, gt_acts = [], [], [], [], []

    for b in batch:
        T   = b["gt_actions"].shape[0]
        pad = max_T - T
        C   = b["cand_feats"].shape[1]   

        tokens.append(b["tokens"])


        vf = b["vp_features"]
        if pad > 0:
            vf = torch.cat([vf, torch.zeros(pad, 36, 512)])
        vp_feats.append(vf)

      
        cf = b["cand_feats"]
        if pad > 0:
            cf = torch.cat([cf, torch.zeros(pad, C, 36, 512)])
        cand_feats.append(cf)

        cm = b["cand_masks"]
        if pad > 0:
            cm = torch.cat([cm, torch.zeros(pad, C, dtype=torch.bool)])
        cand_masks.append(cm)

        ga = b["gt_actions"]
        if pad > 0:
            ga = torch.cat([ga, torch.full((pad,), -1, dtype=torch.long)])
        gt_acts.append(ga)

    return {
        "tokens"     : torch.stack(tokens),       
        "vp_features": torch.stack(vp_feats),    
        "cand_feats" : torch.stack(cand_feats),  
        "cand_masks" : torch.stack(cand_masks),  
        "gt_actions" : torch.stack(gt_acts),     
    }
