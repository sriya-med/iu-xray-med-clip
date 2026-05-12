from dataset import load_splits, build_mesh_similarity, IUXrayDataset
from torch.utils.data import DataLoader

train_df, val_df, test_df = load_splits()

#check similarity matrix on a small slice
import numpy as np
small_sim = build_mesh_similarity(train_df.head(10))
print("Similarity matrix (10x10):")
print(np.round(small_sim, 2))

#check one batch loads correctly
train_ds = IUXrayDataset(train_df)
loader   = DataLoader(train_ds, batch_size=4, shuffle=False)
batch    = next(iter(loader))

print("\nBatch keys:", list(batch.keys()))
print("pixel_values shape:", batch["pixel_values"].shape)   # (4, 3, 224, 224)
print("input_ids shape:   ", batch["input_ids"].shape)       # (4, 77)
print("idx:               ", batch["idx"])