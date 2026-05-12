import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_splits, IUXrayDataset
from model   import MedCLIP

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

def get_all_embeddings(model, loader):
    all_img, all_txt = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding"):
            pv  = batch["pixel_values"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            am  = batch["attention_mask"].to(DEVICE)
            img_emb, txt_emb = model.get_embeddings(pv, ids, am)
            all_img.append(img_emb.cpu())
            all_txt.append(txt_emb.cpu())
    return torch.cat(all_img), torch.cat(all_txt)


def recall_at_k(img_embs, txt_embs, ks=(1, 5, 10)):
    """
    Image-to-text and text-to-image retrieval.
    Ground truth: index i matches index i.
    """
    n = img_embs.size(0)
    #cosine similarity matrix (already normalized)
    sim = img_embs @ txt_embs.T    # (n, n)

    results = {}
    for k in ks:
        #img to text
        topk_i2t = sim.topk(k, dim=1).indices # (n, k)
        labels = torch.arange(n).unsqueeze(1)# (n, 1)
        i2t = (topk_i2t == labels).any(dim=1).float().mean().item()

        #text to img
        topk_t2i = sim.T.topk(k, dim=1).indices
        t2i = (topk_t2i == labels).any(dim=1).float().mean().item()

        results[f"i2t_R@{k}"] = round(i2t * 100, 2)
        results[f"t2i_R@{k}"] = round(t2i * 100, 2)

    return results

def semantic_recall_at_k(img_embs, txt_embs, test_df, ks=(1, 5, 10)):
    """
    R@K but a retrieved text counts as correct if it shares
    at least one MeSH label with the query image's report.
    """
    n = img_embs.size(0)
    sim = img_embs @ txt_embs.T   # (n, n)

    # Build label sets for the test set
    label_sets = []
    for mesh in test_df["mesh_labels"].fillna(""):
        labels = set(x.strip() for x in mesh.split("|") if x.strip())
        label_sets.append(labels)

    results = {}
    for k in ks:
        topk_indices = sim.topk(k, dim=1).indices   # (n, k)
        hits = 0
        for i in range(n):
            query_labels = label_sets[i]
            if not query_labels:
                # no label info — fall back to exact match
                if i in topk_indices[i].tolist():
                    hits += 1
                continue
            # Check if ANY of the top-k retrieved texts share a label
            for j in topk_indices[i].tolist():
                retrieved_labels = label_sets[j]
                if query_labels & retrieved_labels:  # intersection non-empty
                    hits += 1
                    break
        results[f"sem_i2t_R@{k}"] = round(hits / n * 100, 2)

    return results


def evaluate(checkpoint_path, split="test"):
    _, val_df, test_df = load_splits()
    df = test_df if split == "test" else val_df

    ds = IUXrayDataset(df)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = MedCLIP().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    img_embs, txt_embs = get_all_embeddings(model, loader)
    metrics = recall_at_k(img_embs, txt_embs)

    

    print(f"\nResults for: {checkpoint_path}")
    print(f"{'Metric':<15} {'Value':>6}")
    print("-" * 22)
    for k, v in metrics.items():
        print(f"{k:<15} {v:>6.2f}%")


    sem_metrics = semantic_recall_at_k(img_embs, txt_embs, df)
    print("\nSemantic Recall (MeSH-based):")
    for k, v in sem_metrics.items():
        print(f"{k:<18} {v:>6.2f}%")
    metrics.update(sem_metrics)

    return metrics


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None, help="Save metrics to this JSON path")
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint, args.split)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved to {args.out}")