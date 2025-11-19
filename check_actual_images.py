import pandas as pd
from pathlib import Path
from collections import Counter

print("Checking ACTUAL available images for top 100 landmarks...")
print("This will check ALL metadata entries to find which files exist.\n")

df = pd.read_csv('data/google_landmarks/metadata/train.csv')
train_dir = Path('data/google_landmarks/train')

print(f"Total images in metadata: {len(df)}")

# Get top 100 landmark IDs from metadata first
metadata_counts = df['landmark_id'].value_counts()
top100_ids = set(metadata_counts.head(100).index)

print(f"Top 100 landmark IDs identified: {len(top100_ids)}")
print("\nNow checking which images actually exist on disk...")

# Count only images that exist on disk for top 100 landmarks
existing_counts = Counter()
total_checked = 0
total_found = 0

for idx, row in df.iterrows():
    landmark_id = row['landmark_id']
    
    # Only check if this is one of the top 100 landmarks
    if landmark_id in top100_ids:
        img_id = row['id']
        img_path = train_dir / img_id[0] / img_id[1] / img_id[2] / f'{img_id}.jpg'
        
        if img_path.exists():
            existing_counts[landmark_id] += 1
            total_found += 1
        
        total_checked += 1
        
        if total_checked % 10000 == 0:
            print(f"Checked {total_checked} images for top 100 landmarks, found {total_found}...")

print(f"\n{'='*80}")
print("RESULTS - ACTUAL AVAILABLE IMAGES FOR TOP 100 LANDMARKS:")
print(f"{'='*80}")

# Get stats for top 100
landmark_list = [(lid, count) for lid, count in existing_counts.most_common(100)]
counts_only = [count for _, count in landmark_list]

print(f"Total landmarks with images: {len(existing_counts)}")
print(f"Images checked: {total_checked}")
print(f"Images found: {total_found}")
print(f"\nTop 100 landmarks - AVAILABLE images:")
print(f"  Minimum:  {min(counts_only)}")
print(f"  Maximum:  {max(counts_only)}")
print(f"  Median:   {sorted(counts_only)[49]}")
print(f"  Mean:     {sum(counts_only)/len(counts_only):.0f}")
print(f"\nFor balanced training, use: --samples-per-class {min(counts_only)}")
print(f"{'='*80}")
