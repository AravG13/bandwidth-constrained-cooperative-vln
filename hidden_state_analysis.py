"""
Measures hidden state alignment between agents with vs without communication.
Tests the hypothesis: communication aligns recurrent representations.


"""

import os
import torch
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_rel

from utils.connectivity import ConnectivityGraph
from multi_agent_utils import (
    PairedR2RDataset,
    paired_collate_fn,
    load_clip_text_encoder,
    encode_views,
)
from models.vln_modules import CooperativeVLNAgent
from torch.utils.data import DataLoader
import argparse


p = argparse.ArgumentParser()

p.add_argument("--ckpt", required=True)
p.add_argument("--budget", type=int, default=3)

p.add_argument(
    "--policy",
    type=str,
    default="learned",
    choices=["learned", "random", "always", "late", "none"]
)

p.add_argument("--split", type=str, default="val_unseen")
p.add_argument("--max_len", type=int, default=20)

p.add_argument(
    "--long_only",
    action="store_true",
    help="Only analyze long trajectories"
)

p.add_argument("--min_steps", type=int, default=8)

args = p.parse_args()



BASE   = os.path.expanduser("~/vln_project")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLDS = {
    0: 1.0,
    1: 0.07,
    3: 0.05,
    5: 0.05,
    8: 0.07,
    10: 0.08
}

THRESH = THRESHOLDS.get(args.budget, 0.10)

print(f"\nPolicy={args.policy}")
print(f"Budget={args.budget}")
print(f"Threshold={THRESH:.3f}")


graph = ConnectivityGraph(
    f"{BASE}/Matterport3DSimulator/connectivity"
)

ds = PairedR2RDataset(
    f"{BASE}/data/r2r/R2R_{args.split}.json",
    f"{BASE}/data/features/CLIP-ViT-B-32-views.hdf5",
    graph,
    split=args.split,
    max_len=args.max_len,
    max_candidates=10,
    aug_instructions=False,
)

loader = DataLoader(
    ds,
    batch_size=16,
    shuffle=False,
    collate_fn=paired_collate_fn,
    num_workers=0,
)


ckpt = torch.load(args.ckpt, map_location=DEVICE)

agent = CooperativeVLNAgent(
    512, 512, 512, 36, 128
).to(DEVICE)

state = ckpt.get("agent0", ckpt.get("model"))

agent.load_state_dict(state, strict=False)

agent.eval()

enc = load_clip_text_encoder(DEVICE)

W = agent.nav_head.mlp[0]

sim_with_comm    = defaultdict(list)
sim_without_comm = defaultdict(list)
gate_rates       = defaultdict(list)

num_batches = 0
num_long = 0

with torch.no_grad():

    for batch_a, batch_b in loader:

        t0 = batch_a["tokens"].to(DEVICE)
        t1 = batch_b["tokens"].to(DEVICE)

        vp0 = encode_views(batch_a["vp_features"].to(DEVICE))
        vp1 = encode_views(batch_b["vp_features"].to(DEVICE))

        cf0 = encode_views(batch_a["cand_feats"].to(DEVICE))
        cf1 = encode_views(batch_b["cand_feats"].to(DEVICE))

        cm0 = batch_a["cand_masks"].to(DEVICE)
        cm1 = batch_b["cand_masks"].to(DEVICE)

        lm0 = enc(t0)
        lm1 = enc(t1)

        lmask0 = (t0 == 0).to(DEVICE)
        lmask1 = (t1 == 0).to(DEVICE)

        B, T, C, _ = cf0.shape

        if args.long_only and T < args.min_steps:
            continue

        if T >= args.min_steps:
            num_long += 1

        num_batches += 1


        h0c = agent.agent_gru.init_hidden(B, DEVICE)
        h1c = agent.agent_gru.init_hidden(B, DEVICE)

        p0c = torch.zeros(B, dtype=torch.long, device=DEVICE)
        p1c = torch.zeros(B, dtype=torch.long, device=DEVICE)

        s0 = torch.zeros(B, device=DEVICE)

        m0c = None
        m1c = None

    

        h0n = agent.agent_gru.init_hidden(B, DEVICE)
        h1n = agent.agent_gru.init_hidden(B, DEVICE)

        p0n = torch.zeros(B, dtype=torch.long, device=DEVICE)
        p1n = torch.zeros(B, dtype=torch.long, device=DEVICE)

        

        for step in range(T):

 
            brem = (
                1 - s0 / max(args.budget, 1)
            ).clamp(0, 1).unsqueeze(-1)

            ctx0, _ = agent.cross_attn(
                vp0[:, step],
                lm0,
                lmask0,
            )

            agg0 = agent.msg_agg(
                h0c,
                m1c.unsqueeze(1) if m1c is not None else None,
            )

            h0c = agent.agent_gru(
                ctx0,
                agg0,
                p0c,
                h0c,
            )

            ctx1, _ = agent.cross_attn(
                vp1[:, step],
                lm1,
                lmask1,
            )

            agg1 = agent.msg_agg(
                h1c,
                m0c.unsqueeze(1) if m0c is not None else None,
            )

            h1c = agent.agent_gru(
                ctx1,
                agg1,
                p1c,
                h1c,
            )

         

            if args.budget == 0:

                gate = torch.zeros(B, device=DEVICE)
                ps = torch.zeros(B, device=DEVICE)

            else:

                _, ps, _ = agent.comm_gate(
                    h0c,
                    brem,
                    deterministic=False,
                )

                if args.policy == "learned":

                    gate = (
                        (ps > THRESH).float()
                        * (s0 < args.budget).float()
                    )

                elif args.policy == "random":

                    gate = (
                        torch.bernoulli(
                            torch.full_like(ps, THRESH)
                        )
                        * (s0 < args.budget).float()
                    )

                elif args.policy == "always":

                    gate = (
                        torch.ones_like(ps)
                        * (s0 < args.budget).float()
                    )

                elif args.policy == "late":

                    gate = (
                        torch.full_like(
                            ps,
                            float(step >= 3)
                        )
                        * (s0 < args.budget).float()
                    )

                elif args.policy == "none":

                    gate = torch.zeros_like(ps)

            gate_rates[step].extend(
                gate.cpu().tolist()
            )

            m1c = ctx0 * gate.unsqueeze(-1)
            m0c = ctx1 * gate.unsqueeze(-1)

            s0 += gate

           

            sc0c = torch.bmm(
                W(cf0[:, step].reshape(B * C, 512)).reshape(B, C, -1),
                W(h0c).unsqueeze(-1),
            ).squeeze(-1)

            sc0c = sc0c.masked_fill(
                ~cm0[:, step],
                float("-inf"),
            )

            p0c = sc0c.argmax(-1)

            sc1c = torch.bmm(
                W(cf1[:, step].reshape(B * C, 512)).reshape(B, C, -1),
                W(h1c).unsqueeze(-1),
            ).squeeze(-1)

            sc1c = sc1c.masked_fill(
                ~cm1[:, step],
                float("-inf"),
            )

            p1c = sc1c.argmax(-1)

         

            ctx0n, _ = agent.cross_attn(
                vp0[:, step],
                lm0,
                lmask0,
            )

            h0n = agent.agent_gru(
                ctx0n,
                agent.msg_agg(h0n, None),
                p0n,
                h0n,
            )

            ctx1n, _ = agent.cross_attn(
                vp1[:, step],
                lm1,
                lmask1,
            )

            h1n = agent.agent_gru(
                ctx1n,
                agent.msg_agg(h1n, None),
                p1n,
                h1n,
            )


            sc0n = torch.bmm(
                W(cf0[:, step].reshape(B * C, 512)).reshape(B, C, -1),
                W(h0n).unsqueeze(-1),
            ).squeeze(-1)

            sc0n = sc0n.masked_fill(
                ~cm0[:, step],
                float("-inf"),
            )

            p0n = sc0n.argmax(-1)

            sc1n = torch.bmm(
                W(cf1[:, step].reshape(B * C, 512)).reshape(B, C, -1),
                W(h1n).unsqueeze(-1),
            ).squeeze(-1)

            sc1n = sc1n.masked_fill(
                ~cm1[:, step],
                float("-inf"),
            )

            p1n = sc1n.argmax(-1)

          

            cos_with = torch.nn.functional.cosine_similarity(
                h0c,
                h1c,
                dim=-1,
            )

            cos_without = torch.nn.functional.cosine_similarity(
                h0n,
                h1n,
                dim=-1,
            )

            sim_with_comm[step].extend(
                cos_with.cpu().tolist()
            )

            sim_without_comm[step].extend(
                cos_without.cpu().tolist()
            )


print("\n" + "=" * 70)
print("Hidden State Alignment: With vs Without Communication")
print("=" * 70)

print(f"\nBatches analyzed: {num_batches}")
print(f"Long trajectories >= {args.min_steps}: {num_long}")

print(
    f"\n{'Step':>4}  "
    f"{'With':>10}  "
    f"{'No comm':>10}  "
    f"{'Δ':>10}  "
    f"{'p-value':>10}  "
    f"{'Gate %':>10}"
)

print("-" * 70)

steps = sorted(sim_with_comm.keys())

all_delta = []

for step in steps:

    with_vals = np.array(sim_with_comm[step])
    no_vals   = np.array(sim_without_comm[step])

    w = np.mean(with_vals)
    n = np.mean(no_vals)

    delta = w - n

    all_delta.append(delta)

    gate_rate = np.mean(gate_rates[step]) * 100

    try:
        _, pval = ttest_rel(with_vals, no_vals)
    except:
        pval = 1.0

    sig = "*" if pval < 0.05 else ""

    print(
        f"{step:4d}  "
        f"{w:10.4f}  "
        f"{n:10.4f}  "
        f"{delta:+10.4f}  "
        f"{pval:10.4f}{sig}  "
        f"{gate_rate:9.1f}%"
    )

cum_gain = np.sum(all_delta)
mean_gain = np.mean(all_delta)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Cumulative alignment gain: {cum_gain:+.4f}")
print(f"Mean per-step alignment gain: {mean_gain:+.4f}")
print(f"Positive Δ steps: {sum(d > 0 for d in all_delta)}/{len(all_delta)}")

print("\nInterpretation:")
print("Positive Δ => communication increases alignment")
print("Early gate spikes => communication concentrated early")
print("Persistent Δ => recurrent synchronization effect")



import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

w_vals = [
    np.mean(sim_with_comm[s])
    for s in steps
]

n_vals = [
    np.mean(sim_without_comm[s])
    for s in steps
]

delta_vals = [
    w - n
    for w, n in zip(w_vals, n_vals)
]

w_err = [
    np.std(sim_with_comm[s]) /
    np.sqrt(len(sim_with_comm[s]))
    for s in steps
]

n_err = [
    np.std(sim_without_comm[s]) /
    np.sqrt(len(sim_without_comm[s]))
    for s in steps
]

gate_plot = [
    np.mean(gate_rates[s]) * 100
    for s in steps
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))



axes[0].errorbar(
    steps,
    w_vals,
    yerr=w_err,
    fmt='o-',
    linewidth=2,
    capsize=3,
    label='With communication',
)

axes[0].errorbar(
    steps,
    n_vals,
    yerr=n_err,
    fmt='s--',
    linewidth=2,
    capsize=3,
    label='No communication',
)

axes[0].set_xlabel('Step')
axes[0].set_ylabel('Cosine similarity')
axes[0].set_title('Hidden-state alignment')
axes[0].legend()
axes[0].grid(alpha=0.3)


colors = [
    'green' if d > 0 else 'red'
    for d in delta_vals
]

axes[1].bar(
    steps,
    delta_vals,
    color=colors,
)

axes[1].axhline(
    0,
    color='black',
    linewidth=1,
)

axes[1].set_xlabel('Step')
axes[1].set_ylabel('Δ similarity')
axes[1].set_title('Communication alignment gain')
axes[1].grid(alpha=0.3, axis='y')

#Gate firing:

axes[2].plot(
    steps,
    gate_plot,
    'o-',
    linewidth=2,
)

axes[2].set_xlabel('Step')
axes[2].set_ylabel('Gate firing rate (%)')
axes[2].set_title('Communication over time')
axes[2].grid(alpha=0.3)

plt.tight_layout()

pdf_path = (
    f"{BASE}/results/"
    f"hidden_alignment_{args.policy}_b{args.budget}.pdf"
)

png_path = (
    f"{BASE}/results/"
    f"hidden_alignment_{args.policy}_b{args.budget}.png"
)

plt.savefig(
    pdf_path,
    bbox_inches='tight',
)

plt.savefig(
    png_path,
    bbox_inches='tight',
    dpi=150,
)

print(f"\nSaved:")
print(f"  {pdf_path}")
print(f"  {png_path}")
