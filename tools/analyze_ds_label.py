import os
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def analyze_yolo_labels(label_dir, image_size=640):
    stats = defaultdict(lambda: {
        "images": set(),
        "features": 0,
        "sizes": []
    })

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    for label_file in tqdm(label_files, desc=f"Processing {label_dir}"):
        filename = os.path.basename(label_file)
        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            w = float(parts[3]) * image_size
            h = float(parts[4]) * image_size
            size = w * h

            stats[class_id]["images"].add(filename)
            stats[class_id]["features"] += 1
            stats[class_id]["sizes"].append(size)

    total_images = len(set(os.path.basename(f) for f in label_files))
    total_features = sum(s["features"] for s in stats.values())
    features_per_image = [len(open(f).readlines()) for f in label_files]

    print(f"\nimages = {total_images} *3*{image_size}*{image_size}")
    print(f"Class feature statistics:")
    print(f"features = {total_features}")
    print(f"features per image = [min = {min(features_per_image)}, mean = {np.mean(features_per_image):.2f}, max = {max(features_per_image)}]")
    print(f"classes = {len(stats)}")
    print(f"{'cls name':40} {'cls value':>10} {'images':>10} {'features':>10} {'min size':>10} {'mean size':>10} {'max size':>10}")

    for cls_id in sorted(stats):
        sizes = stats[cls_id]["sizes"]
        print(f"{str(cls_id):40} {cls_id:10} {len(stats[cls_id]['images']):10} {stats[cls_id]['features']:10} "
              f"{np.min(sizes):10.2f} {np.mean(sizes):10.2f} {np.max(sizes):10.2f}")

# Example usage:
analyze_yolo_labels("dataset/merged_seg_dataset/labels/train")
analyze_yolo_labels("dataset/merged_seg_dataset/labels/val")
