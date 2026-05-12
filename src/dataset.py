import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import CLIPProcessor

PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_splits(csv_path="data/dataset.csv", seed=42):
    df = pd.read_csv(csv_path).reset_index(drop=True)

    #use impreession as the text
    #fall back to findings if impression is empty
    df["text"] = df["impression"].fillna("").str.strip()
    empty_mask = df["text"] == ""
    df.loc[empty_mask, "text"] = df.loc[empty_mask, "findings"].fillna("").str.strip()

    #drop any remaining rows with no text
    df = df[df["text"] != ""].reset_index(drop=True)

    #parse mesh_labels back into a list
    df["mesh_list"] = df["mesh_labels"].fillna("").apply(
        lambda s: [x.strip() for x in s.split("|") if x.strip()]
    )

    # 80/10/10 split — stratify isn't easy with multi-label, so just random
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
    val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df


def build_mesh_similarity(df):
    """
    compute a Jaccard similarity matrix over MeSH labels for the full df.
    Returns an (N x N) numpy float32 matrix.
    Used later to identify false negatives during training.
    """
    label_sets = [set(row) for row in df["mesh_list"]]
    n = len(label_sets)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            a, b = label_sets[i], label_sets[j]
            if not a and not b:
                #both empty — treat as similar (both "No Indexing" / unlabeled)
                val = 1.0
            elif not a or not b:
                val = 0.0
            else:
                val = len(a & b) / len(a | b)   # jaccard
            sim[i, j] = val
            sim[j, i] = val
    return sim


class IUXrayDataset(Dataset):
    def __init__(self, df, processor=None):
        self.df = df
        self.processor = processor or PROCESSOR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")

        text = row["text"]

        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),   # (3, 224, 224)
            "input_ids": encoding["input_ids"].squeeze(0),       # (77,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (77,)
            "idx": idx,   # in order to look up similarity matrix during training
        }