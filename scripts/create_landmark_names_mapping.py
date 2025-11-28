"""
Create landmark_names_500classes.json with human-readable names
"""
import json
from pathlib import Path

# Load class mapping (idx -> landmark_id)
class_mapping_path = Path(__file__).parent.parent / "data" / "landmarks_500class" / "class_mapping.json"
with open(class_mapping_path) as f:
    class_mapping = json.load(f)

# Load unified landmarks (landmark_id -> name)
unified_path = Path(__file__).parent.parent / "data" / "landmarks_unified.json"
with open(unified_path) as f:
    unified = json.load(f)

# Create lookup dict for landmark_id -> name
id_to_name = {lm["landmark_id"]: lm["name"] for lm in unified["landmarks"]}

# Create idx -> name mapping
idx_to_name = {}
for landmark_id_str, idx in class_mapping["landmark_to_idx"].items():
    landmark_id = int(landmark_id_str)
    
    # Get raw name from unified database
    raw_name = id_to_name.get(landmark_id, f"landmark_{landmark_id}")
    
    # Clean up Wiki URLs to human-readable names
    if raw_name.startswith("http://commons.wikimedia.org/wiki/Category:"):
        # Extract category name and clean it
        name = raw_name.split("Category:")[-1]
        name = name.replace("_", " ")
        name = name.replace("%20", " ")
        name = name.replace("%C3%A9", "é")
        name = name.replace("%C3%A8", "è")
        name = name.replace("%C3%A0", "à")
        # Remove common prefixes
        name = name.replace("Media contributed by the ", "")
    else:
        name = raw_name
    
    idx_to_name[idx] = name

# Create output
output = {
    "num_classes": 500,
    "idx_to_name": idx_to_name
}

# Save to checkpoints directory
output_path = Path(__file__).parent.parent / "data" / "checkpoints" / "landmark_names_500classes.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Created {output_path}")
print(f"Total classes: {len(idx_to_name)}")
print("\nSample mappings:")
for i in range(10):
    print(f"  {i}: {idx_to_name[i]}")
