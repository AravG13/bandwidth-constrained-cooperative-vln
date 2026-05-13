
import json, os

base = os.path.expanduser("~/vln_project/data/r2r")

for split in ["train"]:  
    data = json.load(open(f"{base}/R2R_{split}.json"))
    augmented = []
    
    for ep in data:
        path = ep["path"]
      
        augmented.append(ep)
        
        for start_idx in range(1, len(path) - 1):
            if len(path) - start_idx < 2:
                continue
            new_ep = dict(ep)
            new_ep["path"]     = path[start_idx:]
            new_ep["path_id"]  = ep["path_id"] * 1000 + start_idx
           
            new_ep["distance"] = ep.get("distance", 5.0) * (
                (len(path) - start_idx) / len(path))
            augmented.append(new_ep)
    
    out = f"{base}/R2R_{split}_augmented.json"
    json.dump(augmented, open(out, "w"))
    print(f"{split}: {len(data)} → {len(augmented)} episodes ({len(augmented)/len(data):.1f}x)")

print("Done.")
