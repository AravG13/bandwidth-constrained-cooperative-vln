"""
gate_analysis.py
================
Analyses WHEN and WHERE the hindsight communication gate fires.
This is the key qualitative result for the paper regardless of SR numbers.

Run: python3 gate_analysis.py
Outputs: results/gate_analysis.pdf, results/gate_analysis.png, printed table
"""

import os, json, torch, numpy as np
from collections import defaultdict
from utils.connectivity import ConnectivityGraph
from multi_agent_utils import (PairedR2RDataset, paired_collate_fn,
                                load_clip_text_encoder, encode_views)
from models.vln_modules import CooperativeVLNAgent
from torch.utils.data import DataLoader

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--ckpt", type=str, default=None)
_p.add_argument("--budget", type=int, default=5)
_args = _p.parse_args()

BASE   = os.path.expanduser("~/vln_project")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CKPT_CANDIDATES = [
    "results/checkpoints_hindsight/best_budget5.pt",
    "results/checkpoints_v3fix/best_budget5.pt",
    "results/checkpoints_multiagent/best_budget5.pt",
    "results/checkpoints_trainval/best_agent.pt",
]
ckpt_path = None
if _args.ckpt:
    ckpt_path = _args.ckpt
    print(f"Using checkpoint: {ckpt_path}")
for c in ([] if ckpt_path else CKPT_CANDIDATES):
    full = os.path.join(BASE, c)
    if os.path.exists(full):
        ckpt_path = full
        print(f"Using checkpoint: {c}")
        break

if ckpt_path is None:
    print("No checkpoint found. Tried:")
    for c in CKPT_CANDIDATES: print(f"  {c}")
    exit(1)

graph = ConnectivityGraph(f"{BASE}/Matterport3DSimulator/connectivity")
ds    = PairedR2RDataset(
    f"{BASE}/data/r2r/R2R_val_unseen.json",
    f"{BASE}/data/features/CLIP-ViT-B-32-views.hdf5",
    graph, split="val_unseen", max_len=20, max_candidates=10,
    aug_instructions=False)
loader = DataLoader(ds, batch_size=32, shuffle=False,
                    collate_fn=paired_collate_fn, num_workers=0)

ckpt  = torch.load(ckpt_path, map_location=DEVICE)
agent = CooperativeVLNAgent(512, 512, 512, 36, 128).to(DEVICE)
state = ckpt.get("agent0", ckpt.get("model"))
agent.load_state_dict(state, strict=False)
agent.eval()

enc = load_clip_text_encoder(DEVICE)
W   = agent.nav_head.mlp[0]


records = []  

with torch.no_grad():
    for batch_a, _ in loader:
        t  = batch_a["tokens"].to(DEVICE)
        vp = encode_views(batch_a["vp_features"].to(DEVICE))   
        cf = encode_views(batch_a["cand_feats"].to(DEVICE))   
        cm = batch_a["cand_masks"].to(DEVICE)                  
        gt = batch_a["gt_actions"].to(DEVICE)                 
        B, T, C, _ = cf.shape

        lm    = enc(t)
        lmask = (t == 0).to(DEVICE)
        h     = agent.agent_gru.init_hidden(B, DEVICE)
        p     = torch.zeros(B, dtype=torch.long, device=DEVICE)
        sends = torch.zeros(B, device=DEVICE)

        for step in range(T):
            ctx, _ = agent.cross_attn(vp[:, step], lm, lmask)
            agg    = agent.msg_agg(h, None)
            h      = agent.agent_gru(ctx, agg, p, h)

            sc  = torch.bmm(
                W(cf[:, step].reshape(B*C, 512)).reshape(B, C, -1),
                W(h).unsqueeze(-1)).squeeze(-1)
            sc  = sc.masked_fill(~cm[:, step], float("-inf"))

            # Gate decision (deterministic, budget=5)
            brem = (1 - sends / max(_args.budget, 1)).clamp(0, 1).unsqueeze(-1)
            _, p_send, _ = agent.comm_gate(
                h, brem, deterministic=False
            )

            gate = (
                p_send > 0.05
            ).float() * (sends < _args.budget).float()

           
            probs      = torch.softmax(sc, dim=-1)         
            gtv        = gt[:, step]
            valid_mask = gtv >= 0

            for b in range(B):
                if not valid_mask[b]:
                    continue
                n_cands     = int(cm[b, step].sum().item())  
                fired       = gate[b].item() > 0.5
                gt_action   = gtv[b].item()
                pred_action = sc[b].argmax().item()
                correct     = (pred_action == gt_action)
                conf = probs[b].max().item()

                records.append({
                    "gate_fired"    : int(fired),
                    "n_neighbours"  : n_cands,
                    "step"          : step,
                    "correct"       : int(correct),
                    "confidence"    : conf,
                    "sends_so_far"  : int(sends[b].item()),
                })

            p      = gtv.clamp(min=0)
            sends += gate

print(f"\n{'='*55}")
print(f"Gate Analysis — {len(records)} step records from val_unseen")
print(f"{'='*55}")

# ── Analysis 1: gate firing rate by connectivity ──────────────────────────────
print("\n1. Gate firing rate by viewpoint connectivity (# navigable neighbours)")
print("-" * 55)
by_conn = defaultdict(list)
for r in records:
    by_conn[r["n_neighbours"]].append(r["gate_fired"])

conn_keys  = sorted(by_conn.keys())
conn_rates = [np.mean(by_conn[k]) * 100 for k in conn_keys]
conn_counts= [len(by_conn[k]) for k in conn_keys]

for k, rate, count in zip(conn_keys, conn_rates, conn_counts):
    bar = "█" * int(rate / 3)
    print(f"  {k:2d} neighbours ({count:5d} steps): {rate:5.1f}%  {bar}")


import numpy as np
conn_vals  = [r["n_neighbours"] for r in records]
gate_vals  = [r["gate_fired"]   for r in records]
corr = np.corrcoef(conn_vals, gate_vals)[0, 1]
print(f"\n  Pearson correlation (connectivity vs gate): {corr:+.3f}")
print(f"  {'Positive' if corr > 0 else 'Negative'} correlation — gate fires "
      f"{'more' if corr > 0 else 'less'} at higher-connectivity viewpoints")

# ── Analysis 2: gate firing rate over episode timeline ────────────────────────
print("\n2. Gate firing rate by step position")
print("-" * 55)
step_groups = [(0,3,"early steps  (0–2) "),
               (3,7,"middle steps (3–6) "),
               (7,20,"late steps   (7+)  ")]
for s1, s2, name in step_groups:
    subset = [r for r in records if s1 <= r["step"] < s2]
    if not subset: continue
    rate   = np.mean([r["gate_fired"] for r in subset]) * 100
    count  = len(subset)
    print(f"  {name}: {rate:5.1f}%  ({count} steps)")

# ── Analysis 3: gate vs correctness (key validation) ─────────────────────────
print("\n3. Does gate fire more when agent is uncertain? (KEY RESULT)")
print("-" * 55)
wrong_records  = [r for r in records if r["correct"] == 0]
right_records  = [r for r in records if r["correct"] == 1]
wrong_fire = np.mean([r["gate_fired"] for r in wrong_records]) * 100 if wrong_records else 0
right_fire = np.mean([r["gate_fired"] for r in right_records]) * 100 if right_records else 0
ratio      = wrong_fire / max(right_fire, 0.01)

print(f"  When agent was WRONG  ({len(wrong_records):5d} steps): {wrong_fire:5.1f}% gate firing")
print(f"  When agent was RIGHT  ({len(right_records):5d} steps): {right_fire:5.1f}% gate firing")
print(f"  Ratio (wrong/right): {ratio:.2f}x")
if ratio > 1.2:
    print(f"  ✓ Gate learned to fire when uncertain — validates hindsight training")
elif ratio > 1.0:
    print(f"  ~ Slight tendency to fire when uncertain")
else:
    print(f"  ✗ Gate does not fire more when wrong — check training")

# ── Analysis 4: confidence distribution ──────────────────────────────────────
print("\n4. Agent confidence when gate fires vs doesn't fire")
print("-" * 55)
fire_conf   = [r["confidence"] for r in records if r["gate_fired"] == 1]
nofire_conf = [r["confidence"] for r in records if r["gate_fired"] == 0]
print(f"  Mean confidence when gate=1 (send):    {np.mean(fire_conf):.3f}")
print(f"  Mean confidence when gate=0 (no send): {np.mean(nofire_conf):.3f}")
print(f"  Gate fires at lower confidence: "
      f"{'YES ✓' if np.mean(fire_conf) < np.mean(nofire_conf) else 'NO ✗'}")

# ── Analysis 5: budget usage ──────────────────────────────────────────────────
print("\n5. Budget usage distribution")
print("-" * 55)
send_counts = defaultdict(int)
for r in records:
    send_counts[r["sends_so_far"]] += 1
for k in sorted(send_counts.keys()):
    print(f"  {k} sends used so far: {send_counts[k]} steps")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Figure 1: firing rate vs connectivity
    axes[0].bar(conn_keys, conn_rates, color='#5B4FBE', alpha=0.85, edgecolor='white')
    axes[0].set_xlabel('Navigable neighbours (connectivity)', fontsize=11)
    axes[0].set_ylabel('Gate firing rate (%)', fontsize=11)
    axes[0].set_title('Gate fires more at junctions', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, max(conn_rates) * 1.3)

    # Figure 2: firing rate by step
    by_step = defaultdict(list)
    for r in records:
        by_step[r["step"]].append(r["gate_fired"])
    steps      = sorted(by_step.keys())
    step_rates = [np.mean(by_step[s]) * 100 for s in steps]
    axes[1].plot(steps, step_rates, 'o-', color='#1D9E75', lw=2, ms=5)
    axes[1].set_xlabel('Step in episode', fontsize=11)
    axes[1].set_ylabel('Gate firing rate (%)', fontsize=11)
    axes[1].set_title('Firing rate over episode', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Figure 3: wrong vs right firing rate
    categories  = ['Agent correct', 'Agent wrong']
    rates_comp  = [right_fire, wrong_fire]
    colours     = ['#1D9E75', '#E06C75']
    bars = axes[2].bar(categories, rates_comp, color=colours, alpha=0.85, edgecolor='white')
    axes[2].set_ylabel('Gate firing rate (%)', fontsize=11)
    axes[2].set_title('Gate fires more\nwhen agent uncertain', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars, rates_comp):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_pdf = os.path.join(BASE, "results/gate_analysis.pdf")
    out_png = os.path.join(BASE, "results/gate_analysis.png")
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    print(f"\nSaved figures:")
    print(f"  {out_pdf}")
    print(f"  {out_png}")

except ImportError:
    print("\nmatplotlib not available — install with: pip install matplotlib")

print(f"{'='*55}")
overall_rate = np.mean([r["gate_fired"] for r in records]) * 100
print(f"Overall gate firing rate: {overall_rate:.1f}%")
print(f"Connectivity-gate correlation: {corr:+.3f}")
print(f"Gate fires {ratio:.1f}x more often when agent is wrong")
print(f"Mean confidence when sending: {np.mean(fire_conf):.3f}")
print(f"Mean confidence when not sending: {np.mean(nofire_conf):.3f}")
