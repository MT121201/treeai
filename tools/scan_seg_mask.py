import os
import numpy as np
from PIL import Image
from collections import Counter

def scan_label_indices(label_dir):
    unique_values = set()
    value_counts = Counter()
    total_files = 0

    for fname in os.listdir(label_dir):
        if not fname.endswith('.png'):
            continue
        path = os.path.join(label_dir, fname)
        mask = np.array(Image.open(path))

        # Flatten and count all unique pixel values
        flat = mask.flatten()
        values, counts = np.unique(flat, return_counts=True)

        unique_values.update(values)
        value_counts.update(dict(zip(values, counts)))
        total_files += 1

    print(f"Scanned {total_files} files in: {label_dir}")
    print(f"Unique label values found: {sorted(unique_values)}")
    print(f"Total classes: {len(unique_values)}")

    if 0 in unique_values:
        print("⚠️ Class 0 is used (commonly background).")
    else:
        print("✅ Class 0 not used.")

    if 255 in unique_values:
        print("✅ Class 255 is used for ignored regions.")
    else:
        print("⚠️ Class 255 not used (ignored regions missing?).")

    print("\nTop 10 most frequent class IDs:")
    for val, cnt in value_counts.most_common(10):
        print(f"Class {val}: {cnt} pixels")

    return unique_values, value_counts

# Example usage:
label_path = '/home/a3ilab01/treeai/det_tree/segmentation/34_RGB_SemSegm_640_pL/annotations/train'  # <-- change to your folder
scan_label_indices(label_path)
