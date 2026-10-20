"""
prepare_data.py
Builds:
1. data/curated_training_dataset.csv - Balanced, deduplicated dataset for fast ML training.
2. data/electronics_catalog.json - Pre-indexed electronics catalog with ratings, inflation deltas, and review distributions.
3. data/electronics_reviews_sample.csv - Sample of electronics reviews for download/inspection.
"""

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_DATA_PATH = BASE / "public_reviews_dataset_cleaned.csv"

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(f"Missing {RAW_DATA_PATH}")

print("Phase 1: Loading raw dataset...")
cols = [
    "asin", "review_id", "reviewer_id", "review_title", "review_text",
    "review_rating", "product_title", "reviewer_classified_fake",
    "reviewer_classified_honest", "review_is_removed_by_amazon"
]

chunks = []
for chunk in pd.read_csv(RAW_DATA_PATH, usecols=lambda c: c in cols, chunksize=100000, low_memory=False):
    chunk["review_text"] = chunk["review_text"].fillna("").astype(str).str.strip()
    chunk["product_title"] = chunk["product_title"].fillna("Unknown Product").astype(str).str.strip()
    chunk["review_rating"] = pd.to_numeric(chunk["review_rating"], errors="coerce")
    chunk = chunk[chunk["review_text"].str.len() > 12]
    chunk = chunk[chunk["review_rating"].between(1, 5)]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df):,} valid review rows.")

# -------------------------------------------------------------
# Category Classification
# -------------------------------------------------------------
CATEGORIES = {
    "🎧 Headphones & Earbuds": r"earbud|headphone|headset|earphone|airpod|in-ear|over-ear",
    "💻 Laptops, Docks & PC": r"laptop|computer|macbook|keyboard|mouse|monitor|docking station|usb hub|ssd|hard drive|hdmi switch",
    "📱 Phones, Cables & Mounts": r"phone charger|lightning cable|phone mount|phone case|screen protector|iphone|smartphone",
    "⌚ Smartwatches & Fitness": r"smart watch|smartwatch|fitness tracker|fitness watch|wristband|step counter",
    "🔊 Bluetooth Speakers": r"soundbar|bluetooth speaker|portable speaker|car speaker|audio speaker",
    "📷 Cameras & Surveillance": r"camera|webcam|dashcam|security camera|baby monitor|microphone",
    "🔋 Power Banks & Wireless": r"power bank|wireless charger|charging station|solar charger|usb power adapter"
}

def classify_category(title):
    t = str(title).lower()
    for cat_name, pattern in CATEGORIES.items():
        if re.search(pattern, t):
            return cat_name
    return "🔌 General Tech & Accessories"

df["category"] = df["product_title"].apply(classify_category)

# -------------------------------------------------------------
# Phase 2: Build Curated Training Dataset
# -------------------------------------------------------------
print("Phase 2: Extracting curated balanced training set...")
fake_mask = df["reviewer_classified_fake"].astype(str).str.lower().eq("true")
honest_mask = df["reviewer_classified_honest"].astype(str).str.lower().eq("true")

# Exclusive research proxy labels
labeled_df = df.loc[fake_mask ^ honest_mask].copy()
labeled_df["label"] = np.where(fake_mask.loc[labeled_df.index], "suspicious", "genuine")

# Deduplicate by reviewer_id to prevent reviewer-level leakage
labeled_df = labeled_df.drop_duplicates(subset=["reviewer_id"], keep="first")
counts = labeled_df["label"].value_counts()
print(f"Unique reviewers per class: {counts.to_dict()}")

per_class = min(15000, int(counts.min()))
balanced_train = pd.concat([
    labeled_df[labeled_df["label"] == "genuine"].sample(n=per_class, random_state=42),
    labeled_df[labeled_df["label"] == "suspicious"].sample(n=per_class, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

train_save_cols = ["review_id", "reviewer_id", "product_title", "category", "review_rating", "review_text", "label"]
CURATED_TRAIN_PATH = DATA_DIR / "curated_training_dataset.csv"
balanced_train[train_save_cols].to_csv(CURATED_TRAIN_PATH, index=False)
print(f"Saved curated training dataset ({len(balanced_train):,} rows) to {CURATED_TRAIN_PATH.name}")

# -------------------------------------------------------------
# Phase 3: Build Electronics Product Catalog
# -------------------------------------------------------------
print("Phase 3: Building electronics product catalog...")

# Electronics reviews
elec_df = df[df["category"] != "🔌 General Tech & Accessories"].copy()

# Group by ASIN to calculate ratings and inflation
catalog = []
for asin, group in elec_df.groupby("asin"):
    total_reviews = len(group)
    if total_reviews < 15:
        continue
    
    title = group["product_title"].iloc[0]
    category = group["category"].iloc[0]
    
    # Flags for fake vs genuine
    is_fake = (group["reviewer_classified_fake"].astype(str).str.lower() == "true") | (group["review_is_removed_by_amazon"] == 1.0)
    fake_count = int(is_fake.sum())
    genuine_count = int((~is_fake).sum())
    fake_pct = round(100.0 * fake_count / total_reviews, 1)
    
    all_rating = float(group["review_rating"].mean())
    
    # If there are genuine reviews, calculate genuine rating; else fallback
    if genuine_count >= 5:
        genuine_rating = float(group.loc[~is_fake, "review_rating"].mean())
    else:
        # Heuristic if too few genuine reviews
        genuine_rating = max(1.0, all_rating - (fake_pct / 100.0) * (all_rating - 1.0))
        
    inflation = round(all_rating - genuine_rating, 2)
    inflation_pct = round((inflation / all_rating) * 100.0, 1) if all_rating > 0 else 0.0
    
    # Trust Grade & Verdict
    if inflation <= 0.15 and fake_pct <= 15:
        grade = "A"
        verdict = "High Trust — Verified Genuine"
        status_color = "green"
    elif inflation <= 0.45 and fake_pct <= 35:
        grade = "B"
        verdict = "Good Trust — Minor Distortion"
        status_color = "blue"
    elif inflation <= 0.80 and fake_pct <= 60:
        grade = "C"
        verdict = "Moderate Inflation — Proceed with Caution"
        status_color = "orange"
    elif inflation <= 1.20:
        grade = "D"
        verdict = "Significant Inflation — Deceptive Reviews Detected"
        status_color = "red"
    else:
        grade = "F"
        verdict = "Severe Manipulation — Extreme Review Bombing/Boosting"
        status_color = "darkred"
        
    # Star distribution all vs genuine
    star_dist_all = group["review_rating"].value_counts().to_dict()
    star_dist_all = {int(k): int(v) for k, v in star_dist_all.items()}
    for s in range(1, 6):
        star_dist_all.setdefault(s, 0)
        
    genuine_group = group.loc[~is_fake]
    star_dist_gen = genuine_group["review_rating"].value_counts().to_dict()
    star_dist_gen = {int(k): int(v) for k, v in star_dist_gen.items()}
    for s in range(1, 6):
        star_dist_gen.setdefault(s, 0)
        
    # Sample top reviews (up to 8 genuine and 8 suspicious)
    sample_reviews = []
    suspicious_samples = group.loc[is_fake].head(8)
    genuine_samples = group.loc[~is_fake].head(8)
    
    for _, row in suspicious_samples.iterrows():
        sample_reviews.append({
            "review_id": str(row["review_id"]),
            "stars": int(row["review_rating"]),
            "text": str(row["review_text"])[:400],
            "label": "suspicious",
            "score": 0.88
        })
    for _, row in genuine_samples.iterrows():
        sample_reviews.append({
            "review_id": str(row["review_id"]),
            "stars": int(row["review_rating"]),
            "text": str(row["review_text"])[:400],
            "label": "genuine",
            "score": 0.15
        })
        
    catalog.append({
        "asin": str(asin),
        "title": title,
        "category": category,
        "total_reviews": total_reviews,
        "displayed_rating": round(all_rating, 2),
        "authentic_rating": round(genuine_rating, 2),
        "inflation_delta": inflation,
        "inflation_pct": inflation_pct,
        "fake_count": fake_count,
        "genuine_count": genuine_count,
        "fake_pct": fake_pct,
        "trust_grade": grade,
        "verdict": verdict,
        "status_color": status_color,
        "star_distribution_all": star_dist_all,
        "star_distribution_genuine": star_dist_gen,
        "sample_reviews": sample_reviews
    })

# Sort catalog by total reviews descending
catalog.sort(key=lambda x: x["total_reviews"], reverse=True)
CATALOG_PATH = DATA_DIR / "electronics_catalog.json"
with open(CATALOG_PATH, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2)

print(f"Saved {len(catalog)} electronics devices to {CATALOG_PATH.name}")

# Also save a sample CSV of electronics reviews
sample_df = elec_df.sample(n=min(5000, len(elec_df)), random_state=42)
SAMPLE_CSV_PATH = DATA_DIR / "electronics_reviews_sample.csv"
sample_df[["asin", "product_title", "category", "review_rating", "review_text"]].to_csv(SAMPLE_CSV_PATH, index=False)
print(f"Saved {len(sample_df)} sample reviews to {SAMPLE_CSV_PATH.name}")

print("\n--- Preparation Complete! ---")
print(f"Total catalog products: {len(catalog)}")
print("Sample product from catalog:")
print(f"Title: {catalog[0]['title'][:60]}")
print(f"Displayed: {catalog[0]['displayed_rating']} ★ | Authentic: {catalog[0]['authentic_rating']} ★ | Inflation: {catalog[0]['inflation_delta']} ★")
