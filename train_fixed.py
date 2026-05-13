"""
train_fixed.py
==============
Single-agent VLN training.

"""
import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.connectivity import ConnectivityGraph
from r2r_dataset import R2RDataset, collate_fn
from models.vln_modules import CooperativeVLNAgent

BASE_DIR  = os.path.expanduser("~/vln_project")
CONN_DIR  = os.path.join(BASE_DIR, "Matterport3DSimulator/connectivity")
DATA_DIR  = os.path.join(BASE_DIR, "data/r2r")
FEAT_PATH = os.path.join(BASE_DIR, "data/features/CLIP-ViT-B-32-views.hdf5")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--hidden_dim",     type=int,   default=512)
    p.add_argument("--action_dim",     type=int,   default=36)
    p.add_argument("--max_candidates", type=int,   default=15)
    p.add_argument("--max_len",        type=int,   default=20)
    p.add_argument("--warmup_epochs",  type=int,   default=5)
    p.add_argument("--save_dir",       type=str,
                   default=os.path.join(BASE_DIR, "results/checkpoints_fixed"))
    p.add_argument("--overfit_test",   action="store_true")
    p.add_argument("--no_wandb",       action="store_true")
    return p.parse_args()


def load_clip(device):
    import clip
    model, _ = clip.load("ViT-B/32", device=device)
    model.float(); model.eval()
    for p in model.parameters(): p.requires_grad = False
    def encode(tokens):
        tokens = tokens.to(device).long()
        with torch.no_grad():
            x = model.token_embedding(tokens)
            x = x + model.positional_embedding
            x = x.permute(1,0,2); x = model.transformer(x)
            x = x.permute(1,0,2); x = model.ln_final(x)
        return x
    return encode


def encode_views(views):
    """
    (*, 36, 512) -> (*, 512) via attention-weighted pooling.
    No learned parameters — pure geometric mean as query.
    """
    orig = views.shape[:-2]
    V = views.reshape(-1, 36, 512)
    q = V.mean(dim=1)
    w = F.softmax(torch.bmm(V, q.unsqueeze(-1)).squeeze(-1), dim=-1)
    return (w.unsqueeze(-1) * V).sum(dim=1).reshape(*orig, 512)


def forward_batch(agent, encode_lang, batch, device, tf_ratio=1.0):
  
    tokens     = batch["tokens"].to(device)
    vp_feats   = batch["vp_features"].to(device)  
    cand_feats = batch["cand_feats"].to(device)    
    cand_masks = batch["cand_masks"].to(device)   
    gt_actions = batch["gt_actions"].to(device)    

    B, T, C, _, _ = cand_feats.shape

    vp_enc   = encode_views(vp_feats)    
    cand_enc = encode_views(cand_feats)  

    lang = encode_lang(tokens)           
    lmask = (tokens == 0).to(device)

    h    = agent.agent_gru.init_hidden(B, device)
    prev = torch.zeros(B, dtype=torch.long, device=device)

    all_scores, all_gt = [], []

    for t in range(T):
        obs   = vp_enc[:, t, :]
        cands = cand_enc[:, t, :, :]  
        mask  = cand_masks[:, t, :]
        gt    = gt_actions[:, t]

        ctx, _ = agent.cross_attn(obs, lang, lmask)
        agg    = agent.msg_agg(h, None)
        h      = agent.agent_gru(ctx, agg, prev, h)

        scores = torch.bmm(cands, h.unsqueeze(-1)).squeeze(-1)  
        scores = scores.masked_fill(~mask, float("-inf"))

        all_scores.append(scores)
        all_gt.append(gt)

        
        if torch.rand(1).item() < tf_ratio:
            prev = gt.clamp(min=0)          
        else:
            prev = scores.argmax(-1)         

    scores_flat = torch.stack(all_scores).view(T * B, C)
    gt_flat     = torch.stack(all_gt).view(T * B)
    valid       = gt_flat >= 0
    if valid.sum() == 0:
        return torch.tensor(0., device=device, requires_grad=True), 0.

    loss = F.cross_entropy(scores_flat[valid], gt_flat[valid])
    acc  = (scores_flat[valid].argmax(-1) == gt_flat[valid]).float().mean().item()
    return loss, acc


def run_epoch(agent, encode_lang, loader, optimizer, device,
              train=True, desc="", tf_ratio=1.0):
    agent.train(train)
    tot_loss, tot_acc, n = 0., 0., 0
    from tqdm import tqdm
    for batch in tqdm(loader, desc=desc, leave=False):
        loss, acc = forward_batch(agent, encode_lang, batch, device,
                                  tf_ratio=tf_ratio if train else 1.0)
        if train and loss.requires_grad:
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
            optimizer.step()
        tot_loss += loss.item(); tot_acc += acc; n += 1
    return tot_loss/max(n,1), tot_acc/max(n,1)


def get_scheduler(optimizer, warmup, total):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch+1)/warmup
        p = (epoch-warmup)/max(1, total-warmup)
        return 0.1 + 0.9*0.5*(1+torch.cos(torch.tensor(3.14159*p)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="vln-fixed", config=vars(args),
                       name="fixed_direct_dot")
        except Exception as e:
            print(f"wandb: {e}"); use_wandb = False

    graph = ConnectivityGraph(CONN_DIR)
    kw    = dict(features_path=FEAT_PATH, conn_graph=graph,
                 max_len=args.max_len, max_candidates=args.max_candidates)

    print("Loading datasets...")
  
    train_ds = R2RDataset(os.path.join(DATA_DIR, "R2R_trainval.json"),
                          split="train", aug_instructions=True, **kw)
    val_ds   = R2RDataset(os.path.join(DATA_DIR, "R2R_val_seen.json"),
                          split="val_seen", aug_instructions=False, **kw)

    if args.overfit_test:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(20)))
        val_ds   = Subset(val_ds,   list(range(20)))
        print("OVERFIT TEST")

    train_ldr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)

    agent     = CooperativeVLNAgent(512, 512, args.hidden_dim,
                                    args.action_dim, 128).to(DEVICE)
    enc       = load_clip(DEVICE)
    optimizer = torch.optim.AdamW(agent.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs)

    print(f"Parameters: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")
    print(f"Training on: {len(train_ds)} episodes")

    best_val = float("inf")
    print(f"\nStarting training...\n")

    for epoch in range(1, args.epochs+1):
       
        tf = max(0.75, 1.0 - 0.0125 * epoch)

        tl, ta = run_epoch(agent, enc, train_ldr, optimizer, DEVICE,
                           train=True, desc=f"Ep{epoch}/{args.epochs}[train]",
                           tf_ratio=tf)
        with torch.no_grad():
            vl, va = run_epoch(agent, enc, val_ldr, optimizer, DEVICE,
                               train=False, desc=f"Ep{epoch}/{args.epochs}[val]  ")
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:2d} | train: loss={tl:.4f} acc={ta:.3f} tf={tf:.2f} | "
              f"val: loss={vl:.4f} acc={va:.3f} | lr={lr:.1e}")

        if use_wandb:
            import wandb
            wandb.log({"train/loss":tl,"train/acc":ta,"val/loss":vl,
                       "val/acc":va,"lr":lr,"tf_ratio":tf,"epoch":epoch})

        if vl < best_val:
            best_val = vl
            torch.save({"epoch":epoch,"model":agent.state_dict(),
                        "val_loss":vl,"val_acc":va,"args":vars(args)},
                       os.path.join(args.save_dir,"best_agent.pt"))
            print(f"  ✓ Saved (val={vl:.4f})")

    print("\nDone.")
    if use_wandb:
        import wandb; wandb.finish()


if __name__ == "__main__":
    main()
