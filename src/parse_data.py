import os
import xml.etree.ElementTree as ET
import pandas as pd

REPORTS_DIR = "data/reports"
IMAGES_DIR  = "data/images"

rows = []

for fname in sorted(os.listdir(REPORTS_DIR)):
    if not fname.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(REPORTS_DIR, fname))
    root = tree.getroot()

    #report id
    uid = root.findtext("uId[@id]")  # fallback
    uid_el = root.find("uId")
    report_id = uid_el.get("id") if uid_el is not None else fname.replace(".xml", "")

    #text fields: findings and impression
    findings   = ""
    impression = ""
    for ab in root.findall(".//AbstractText"):
        label = ab.get("Label", "")
        text  = (ab.text or "").strip()
        if label == "FINDINGS":
            findings = text
        elif label == "IMPRESSION":
            impression = text

    # Mesh labels: take all major ones
    mesh_labels = [el.text.strip() for el in root.findall(".//MeSH/major") if el.text]

    #images, take first parent image
    images = root.findall("parentImage")
    if not images:
        continue
    image_id  = images[0].get("id")          # e.g. CXR1_1_IM-0001-3001
    image_path = os.path.join(IMAGES_DIR, image_id + ".png")

    #skip if image file doesn't exist
    if not os.path.exists(image_path):
        continue

    #skip if both text fields are empty
    if not findings and not impression:
        continue

    rows.append({
        "report_id":   report_id,
        "image_path":  image_path,
        "findings":    findings,
        "impression":  impression,
        "mesh_labels": "|".join(mesh_labels),  # pipe-separated for CSV storage
    })

df = pd.DataFrame(rows)
print(f"Total samples: {len(df)}")
print(f"MeSH label sample:\n{df['mesh_labels'].value_counts().head(10)}")
print(df.head(3).to_string())

df.to_csv("data/dataset.csv", index=False)
print("\nSaved to data/dataset.csv")