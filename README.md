# Early to Share, Late to Save: Synchronisation-Driven Communication Gating in Bandwidth-Constrained Cooperative VLN

**Arav Gupta, BITS Pilani**

> Submitted to RLxF @ ICML 2026 Workshop

---

## Overview

This repository contains code for **bandwidth-constrained cooperative Vision-Language Navigation (VLN)** — a new problem formulation where two agents navigate indoor environments following natural language instructions, with a hard limit on how many observations they can broadcast to each other.

**Key finding:** Contrary to the intuition that agents should communicate when uncertain, trained gates fire predominantly in *early episode steps* and *when agents are confident*. We explain this through recurrent hidden-state alignment — early communication injects grounded trajectory representations that persist through subsequent GRU updates.

---

## Results Summary

| Method | Val-Seen SR | Val-Unseen SR |
|--------|-------------|---------------|
| Seq2Seq baseline | 39% | 22% |
| Single-agent (ours) | **43.2%** | 9.2% |
| Multi-agent full-comm | — | ~9.3% |
| Hindsight gate B=3 | — | ~8.4% |

**Alignment finding:** Learned gating achieves **+63% more hidden-state alignment gain** per transmission than random gating at matched budget (cumulative Δ +0.031 vs +0.019).

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/AravG13/vln-bandwidth.git
cd vln-bandwidth
```

### 2. Create conda environment

```bash
conda create -n vln python=3.10
conda activate vln
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/CLIP.git
pip install h5py tqdm wandb scipy matplotlib pandas
```

### 3. Clone MatterSim connectivity files

```bash
mkdir -p Matterport3DSimulator
git clone https://github.com/peteanderson80/Matterport3DSimulator.git Matterport3DSimulator
```

### 4. Download R2R data

```bash
mkdir -p data/r2r
cd data/r2r
bash ../../Matterport3DSimulator/tasks/R2R/data/download.sh
cd ../..
```

### 5. Request Matterport3D access

Fill in the form at [niessner.github.io/Matterport](https://niessner.github.io/Matterport) to get access to the skybox images. Once you receive the download script:

```bash
mkdir -p data/matterport
# Download skybox images for the 72 R2R scans only
while read scan; do
    python3 download_mp.py -o data/matterport --id $scan --type matterport_skybox_images
done < data/r2r/r2r_scans.txt
```

### 6. Extract CLIP features

```bash
python3 extract_features.py --img_dir data/matterport
# Produces: data/features/CLIP-ViT-B-32-views.hdf5
# Takes ~2-3 hours on a GPU
```

---

## Training

### Phase 1: Single-agent navigation

```bash
python3 train_fixed.py \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --max_candidates 15 \
    --save_dir results/checkpoints_fixed
```

### Phase 2 & 3: Hindsight gate training

```bash
python3 hindsight_gate_train.py \
    --nav_epochs 0 \
    --gate_epochs 20 \
    --joint_epochs 15 \
    --budget 3 \
    --init_from results/checkpoints_fixed/best_agent.pt \
    --save_dir results/checkpoints_hindsight
```

To train at different budgets:

```bash
for budget in 1 3 5; do
    python3 hindsight_gate_train.py \
        --nav_epochs 0 \
        --gate_epochs 20 \
        --joint_epochs 15 \
        --budget $budget \
        --init_from results/checkpoints_fixed/best_agent.pt \
        --save_dir results/checkpoints_budget${budget}
done
```

---

## Evaluation

### Single-agent

```bash
python3 evaluate_new.py \
    --split val_unseen \
    --ckpt results/checkpoints_fixed/best_agent.pt
```

### Multi-agent with hindsight gate

```bash
python3 eval_multiagent_simple.py \
    --split val_unseen \
    --budget 3 \
    --ckpt results/checkpoints_hindsight/best_budget3.pt
```

### No-communication baseline

```bash
python3 eval_multiagent_simple.py \
    --split val_unseen \
    --budget 0 \
    --no_comm \
    --ckpt results/checkpoints_hindsight/best_budget3.pt
```

### Full communication baseline (B=∞)

```bash
python3 eval_multiagent_simple.py \
    --split val_unseen \
    --budget 999 \
    --ckpt results/checkpoints_hindsight/best_budget3.pt
```

---

## Analysis

### Gate firing analysis

```bash
python3 gate_analysis.py \
    --ckpt results/checkpoints_hindsight/best_budget3.pt \
    --budget 3
# Outputs: results/gate_analysis.pdf
```

### Hidden-state alignment analysis

```bash
# Compare all communication policies
for policy in learned random always none; do
    python3 hidden_state_analysis.py \
        --ckpt results/checkpoints_hindsight/best_budget3.pt \
        --budget 3 \
        --policy $policy
done
# Outputs: results/hidden_alignment_{policy}_b3.pdf
```

### Budget scaling

```bash
for budget in 1 3 5; do
    python3 hidden_state_analysis.py \
        --ckpt results/checkpoints_hindsight/best_budget3.pt \
        --budget $budget \
        --policy learned
done
```

---

## Repository Structure

```
vln-bandwidth/
├── models/
│   └── vln_modules.py          # Core architecture: CrossModalAttn, GRU, CommGate
├── utils/
│   └── connectivity.py         # Matterport3D navigation graph
├── train_fixed.py              # Single-agent navigation training
├── hindsight_gate_train.py     # 3-phase hindsight gating training
├── eval_multiagent_simple.py   # Multi-agent evaluation (reliable)
├── evaluate_new.py             # Single-agent evaluation
├── gate_analysis.py            # Gate firing pattern analysis
├── hidden_state_analysis.py    # Hidden-state alignment measurement
├── multi_agent_utils.py        # Paired dataset and utilities
├── same_goal_dataset_v2.py     # Asymmetric path dataset
├── extract_features.py         # CLIP feature extraction
├── r2r_dataset.py              # R2R dataset loader
└── data/
    ├── r2r/                    # R2R JSON files
    └── features/               # Precomputed CLIP features (HDF5)
```

---

## Key Design Decisions

**Why asymmetric path pairing?**
Agent 0 navigates the full path; Agent 1 starts at the midpoint. This gives Agent 1 genuine knowledge of the goal region that Agent 0 lacks, creating real information asymmetry. Without this, agents navigate different rooms in the same building — messages contain irrelevant observations.

**Why hindsight BCE instead of REINFORCE?**
REINFORCE suffers from high gradient variance when the causal link between a gate decision and episode outcome is weak. BCE on post-hoc failure labels provides a direct, low-variance signal at every step with no policy gradients needed.

**Why measure hidden-state alignment instead of SR?**
Our base agent generalises poorly to unseen buildings (9.2% val-unseen SR), meaning partner messages carry incorrect context. Hidden-state alignment measures whether communication produces consistent internal representations — a necessary condition for coordination that holds even when SR gains require a stronger base policy.

---

## Citation

```bibtex
@article{gupta2026hindsight,
  title     = {Early to Share, Late to Save: Synchronisation-Driven
               Communication Gating in Bandwidth-Constrained
               Cooperative Vision-Language Navigation},
  author    = {Gupta, Arav},
  journal   = {RLxF @ ICML 2026 Workshop},
  year      = {2026}
}
```

---

## Acknowledgements

Built on [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) and [CLIP](https://github.com/openai/CLIP).
R2R dataset from [Anderson et al., 2018](https://arxiv.org/abs/1711.07280).
