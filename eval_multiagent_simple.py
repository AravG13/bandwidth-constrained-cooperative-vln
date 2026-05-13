"""
Simple multi-agent evaluator using W1 dot product scoring.
Works with any checkpoint that has agent0/agent1 keys.
"""
import os, json, argparse, collections
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.connectivity import ConnectivityGraph
from multi_agent_utils import (PairedR2RDataset, paired_collate_fn,
                                load_clip_text_encoder, encode_views)
from models.vln_modules import CooperativeVLNAgent

BASE = os.path.expanduser("~/vln_project")
CONN = os.path.join(BASE, "Matterport3DSimulator/connectivity")
DATA = os.path.join(BASE, "data/r2r")
FEAT = os.path.join(BASE, "data/features/CLIP-ViT-B-32-views.hdf5")
DEV  = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val_unseen")
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gate_thresh", type=float, default=0.1)
    p.add_argument("--no_comm", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_candidates", type=int, default=10)
    return p.parse_args()

def load_distances(conn_dir):
    distances = {}
    for fname in os.listdir(conn_dir):
        if not fname.endswith("_connectivity.json"): continue
        scan = fname.replace("_connectivity.json","")
        conn = json.load(open(os.path.join(conn_dir, fname)))
        pos  = {v["image_id"]: (v["pose"][3],v["pose"][7],v["pose"][11]) for v in conn}
        adj  = collections.defaultdict(list)
        for node in conn:
            vp = node["image_id"]
            for i, nb in enumerate(node.get("unobstructed",[])):
                if nb:
                    nb_vp = conn[i]["image_id"]
                    d = float(np.linalg.norm(np.array(pos[vp])-np.array(pos[nb_vp])))
                    adj[vp].append((nb_vp, d))
        sd = {}
        for start in pos:
            dm = {start:0.}; q = collections.deque([start])
            while q:
                cur = q.popleft()
                for nb,d in adj[cur]:
                    if nb not in dm: dm[nb]=dm[cur]+d; q.append(nb)
            sd[start] = dm
        distances[scan] = sd
    return distances

def compute_sr_spl(episodes, pred_paths, distances):
    sr, spl = [], []
    for ep, pred in zip(episodes, pred_paths):
        scan=ep["scan"]; gt=ep["path"]
        goal=gt[-1]; start=gt[0]
        end=pred[-1] if pred else start
        try: d=distances[scan][end][goal]
        except: d=float("inf")
        succ=float(d<=3.0)
        try: short=distances[scan][start][goal]
        except: short=1.0
        plen=0.
        for i in range(len(pred)-1):
            try: plen+=distances[scan][pred[i]][pred[i+1]]
            except: plen+=3.0
        sr.append(succ); spl.append(succ*short/max(short,plen,1e-6))
    return float(np.mean(sr)), float(np.mean(spl))

def main():
    args = parse_args()
    graph = ConnectivityGraph(CONN)
    distances = load_distances(CONN)

    json_path = os.path.join(DATA, f"R2R_{args.split}.json")
    with open(json_path) as f:
        all_eps = json.load(f)

    ds = PairedR2RDataset(json_path, FEAT, graph,
                          split=args.split, max_len=20,
                          max_candidates=args.max_candidates,
                          aug_instructions=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=paired_collate_fn, num_workers=4)

    ckpt  = torch.load(args.ckpt, map_location=DEV)
    sargs = ckpt.get("args", {"hidden_dim":512,"action_dim":36})

    agent0 = CooperativeVLNAgent(512,512,sargs["hidden_dim"],
                                  sargs["action_dim"],128).to(DEV)
    agent0.load_state_dict(ckpt.get("agent0", ckpt.get("model")))

    if ckpt.get("agent1") is not None:
        agent1 = CooperativeVLNAgent(512,512,sargs["hidden_dim"],
                                      sargs["action_dim"],128).to(DEV)
        agent1.load_state_dict(ckpt["agent1"])
    else:
        agent1 = agent0

    agent0.eval(); agent1.eval()
    enc = load_clip_text_encoder(DEV)

    # Pair episodes same way as dataset
    by_scan = collections.defaultdict(list)
    for ep in all_eps: by_scan[ep["scan"]].append(ep)
    paired_eps = []
    for scan, eps in by_scan.items():
        for i in range(0, len(eps)-1, 2):
            paired_eps.append((eps[i], eps[i+1]))
        if len(eps)%2==1: paired_eps.append((eps[-1], eps[0]))

    all_p0, all_p1, ep0_list, ep1_list = [], [], [], []
    total_sends = 0.; n_batches = 0; pair_idx = 0

    with torch.no_grad():
        for ba, bb in tqdm(loader, desc=f"{args.split} B={args.budget}"):
            B = ba["tokens"].shape[0]
            t0=ba["tokens"].to(DEV); t1=bb["tokens"].to(DEV)
            vp0=encode_views(ba["vp_features"].to(DEV))
            vp1=encode_views(bb["vp_features"].to(DEV))
            cf0=encode_views(ba["cand_feats"].to(DEV))
            cf1=encode_views(bb["cand_feats"].to(DEV))
            cm0=ba["cand_masks"].to(DEV); cm1=bb["cand_masks"].to(DEV)
            lm0=enc(t0); lm1=enc(t1)
            lmask0=(t0==0).to(DEV); lmask1=(t1==0).to(DEV)
            _,T,C,_=cf0.shape

            h0=agent0.agent_gru.init_hidden(B,DEV)
            h1=agent1.agent_gru.init_hidden(B,DEV)
            p0=torch.zeros(B,dtype=torch.long,device=DEV)
            p1=torch.zeros(B,dtype=torch.long,device=DEV)
            s0=torch.zeros(B,device=DEV); s1=torch.zeros(B,device=DEV)
            m0=m1=None
            pred_a0=torch.full((B,T),-1,dtype=torch.long)
            pred_a1=torch.full((B,T),-1,dtype=torch.long)
            ep_sends=torch.zeros(B,device=DEV)

            for t in range(T):
                br0=(1-s0/max(args.budget,1)).clamp(0,1).unsqueeze(-1)
                br1=(1-s1/max(args.budget,1)).clamp(0,1).unsqueeze(-1)

                ctx0,_=agent0.cross_attn(vp0[:,t],lm0,lmask0)
                agg0=agent0.msg_agg(h0,m0.unsqueeze(1) if m0 is not None else None)
                h0=agent0.agent_gru(ctx0,agg0,p0,h0)
                W0=agent0.nav_head.mlp[0]
                sc0=torch.bmm(W0(cf0[:,t].reshape(B*C,512)).reshape(B,C,-1),
                              W0(h0).unsqueeze(-1)).squeeze(-1)
                sc0=sc0.masked_fill(~cm0[:,t],float("-inf"))
                pred0=sc0.argmax(-1); pred_a0[:,t]=pred0.cpu()

                ctx1,_=agent1.cross_attn(vp1[:,t],lm1,lmask1)
                agg1=agent1.msg_agg(h1,m1.unsqueeze(1) if m1 is not None else None)
                h1=agent1.agent_gru(ctx1,agg1,p1,h1)
                W1=agent1.nav_head.mlp[0]
                sc1=torch.bmm(W1(cf1[:,t].reshape(B*C,512)).reshape(B,C,-1),
                              W1(h1).unsqueeze(-1)).squeeze(-1)
                sc1=sc1.masked_fill(~cm1[:,t],float("-inf"))
                pred1=sc1.argmax(-1); pred_a1[:,t]=pred1.cpu()

                if args.no_comm or args.budget == 0:
                    g0=g1=torch.zeros(B,device=DEV)
                else:
                    _,ps0,_=agent0.comm_gate(h0,br0,deterministic=False)
                    _,ps1,_=agent1.comm_gate(h1,br1,deterministic=False)
                    if args.budget >= 999:  # full comm
                        g0=(s0<999).float(); g1=(s1<999).float()
                    else:
                        g0=(ps0>args.gate_thresh).float()*(s0<args.budget).float()
                        g1=(ps1>args.gate_thresh).float()*(s1<args.budget).float()

                m1=ctx0*g0.unsqueeze(-1); m0=ctx1*g1.unsqueeze(-1)
                s0+=g0; s1+=g1; ep_sends+=g0+g1
                p0=pred0; p1=pred1

            total_sends+=ep_sends.mean().item(); n_batches+=1

            for b in range(B):
                if pair_idx>=len(paired_eps): break
                ea,eb=paired_eps[pair_idx]; pair_idx+=1
                def to_path(ep, acts):
                    scan=ep["scan"]; path=ep["path"]
                    T2=min(len(path)-1,20); vp_path=[path[0]]
                    for step in range(T2):
                        a=acts[b,step].item()
                        if a<0: break
                        cur=vp_path[-1]
                        nbrs=graph.get_neighbours(scan,cur)
                        cands=nbrs+[cur]
                        if a<len(cands):
                            nxt=cands[a]; vp_path.append(nxt)
                            if nxt==cur: break
                        else: break
                    return vp_path
                all_p0.append(to_path(ea,pred_a0))
                all_p1.append(to_path(eb,pred_a1))
                ep0_list.append(ea); ep1_list.append(eb)

    sr0,spl0=compute_sr_spl(ep0_list,all_p0,distances)
    sr1,spl1=compute_sr_spl(ep1_list,all_p1,distances)
    avg=(sr0+sr1)/2; savg=(spl0+spl1)/2
    sends=total_sends/max(n_batches,1)

    print(f"\n{'='*50}")
    print(f"Split: {args.split} | Budget: {args.budget} | Sends: {sends:.2f}")
    print(f"Agent0  SR:{sr0*100:.1f}%  SPL:{spl0*100:.1f}%")
    print(f"Agent1  SR:{sr1*100:.1f}%  SPL:{spl1*100:.1f}%")
    print(f"Average SR:{avg*100:.1f}%  SPL:{savg*100:.1f}%")
    print(f"{'='*50}")

if __name__=="__main__":
    main()
