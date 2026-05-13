

import os, json, argparse, math
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
import torch
import clip

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default='data/matterport')
parser.add_argument('--out', default='data/features/CLIP-ViT-B-32-views.hdf5')
parser.add_argument('--batch_size', default=36, type=int)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
for p in model.parameters():
    p.requires_grad = False


base = os.getcwd()
scans_needed = set()
for split in ['train', 'val_seen', 'val_unseen']:
    jpath = os.path.join(base, f'data/r2r/R2R_{split}.json')
    if os.path.exists(jpath):
        for ep in json.load(open(jpath)):
            scans_needed.add(ep['scan'])
print(f"Scans needed: {len(scans_needed)}")


conn_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'Matterport3DSimulator/connectivity'
)

os.makedirs(os.path.dirname(args.out), exist_ok=True)


HEADINGS   = [i * (2 * math.pi / 12) for i in range(12)]
ELEVATIONS = [-math.pi / 6, 0, math.pi / 6]

def get_skybox_path(img_dir, scan, viewpoint, heading_idx):
    """Matterport skybox images: face index = heading_idx % 6"""
    face = heading_idx % 6
    return os.path.join(
        img_dir, scan, scan,
        'matterport_skybox_images',
        f'{viewpoint}_skybox{face}_sami.jpg'
    )

with h5py.File(args.out, 'w') as out_f:
    for scan in tqdm(sorted(scans_needed), desc='Scans'):
       
        conn_path = os.path.join(conn_dir, f'{scan}_connectivity.json')
        if not os.path.exists(conn_path):
            print(f"  SKIP {scan} — no connectivity file")
            continue

        conn = json.load(open(conn_path))
        viewpoints = [v['image_id'] for v in conn if v.get('included', True)]

        scan_grp = out_f.create_group(scan)

        for vp in viewpoints:
            imgs = []
            for h_idx in range(12):
                for e_idx in range(3):
                    img_path = get_skybox_path(args.img_dir, scan, vp, h_idx)
                    if os.path.exists(img_path):
                        try:
                            img = preprocess(Image.open(img_path).convert('RGB'))
                        except Exception:
                            img = torch.zeros(3, 224, 224)
                    else:
                        img = torch.zeros(3, 224, 224)
                    imgs.append(img)

            imgs_t = torch.stack(imgs).to(device)  

            with torch.no_grad():
                feats = model.encode_image(imgs_t)         
                feats = feats / feats.norm(dim=-1, keepdim=True)  

            scan_grp.create_dataset(vp, data=feats.cpu().float().numpy())

print(f"\nDone. Saved to {args.out}")
