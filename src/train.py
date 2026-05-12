import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_splits, build_mesh_similarity, IUXrayDataset
from model   import MedCLIP
from losses  import clip_loss, masked_infonce, soft_target_infonce

#args 
parser = argparse.ArgumentParser()
parser.add_argument("--loss", default="baseline", choices=["baseline", "masked", "soft"], help="Which loss to use")
parser.add_argument("--epochs", type=int,   default=5)
parser.add_argument("--batch_size", type=int,   default=32)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--threshold", type=float, default=0.5, help="Jaccard threshold for masked loss")
parser.add_argument("--tau", type=float, default=0.5, help="Temperature for soft-target loss")
args = parser.parse_args()

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else   #apple silicon
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
print(f"Device: {DEVICE}  |  Loss: {args.loss}")

#data
train_df, val_df, _ = load_splits()

#precompute full train similarity matrix (used to slice per-batch)
print("Computing similarity matrix...")
train_sim = build_mesh_similarity(train_df)   #(N, N) numpy float32
train_sim_t = torch.tensor(train_sim)         #keep on CPU, slice per batch

train_ds = IUXrayDataset(train_df)
val_ds = IUXrayDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

#model
model = MedCLIP().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

#training loop
def compute_loss(logits_img, logits_txt, idxs, loss_type):
    if loss_type == "baseline":
        return clip_loss(logits_img)

    #get the sim sub-matrix for this batch
    sim = train_sim_t[idxs][:, idxs].to(DEVICE)

    if loss_type == "masked":
        return masked_infonce(logits_img, sim, threshold=args.threshold)
    if loss_type == "soft":
        return soft_target_infonce(logits_img, sim, tau=args.tau)


best_val_loss = float("inf")
os.makedirs("results", exist_ok=True)
save_path = f"results/{args.loss}_best.pt"

for epoch in range(args.epochs):
    #train
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
        pv = batch["pixel_values"].to(DEVICE)
        ids = batch["input_ids"].to(DEVICE)
        am = batch["attention_mask"].to(DEVICE)
        idxs = batch["idx"]

        logits_img, logits_txt = model(pv, ids, am)
        loss = compute_loss(logits_img, logits_txt, idxs, args.loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_train = total_loss / len(train_loader)

    #val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]  "):
            pv = batch["pixel_values"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            am = batch["attention_mask"].to(DEVICE)
            logits_img, logits_txt = model(pv, ids, am)
            #use standard loss for val (fair comparison across methods)
            val_loss += clip_loss(logits_img).item()

    avg_val = val_loss / len(val_loader)
    print(f"  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), save_path)
        print(f" Saved best model to: {save_path}")

print(f"\nDone. Best val loss: {best_val_loss:.4f}")