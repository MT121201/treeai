import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def draw_yolo_segmentation(image_path):
    if not os.path.isfile(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    parts = image_path.replace("\\", "/").split("/")
    try:
        split_idx = parts.index("images")
        root_dir = "/".join(parts[:split_idx])
        split = parts[split_idx + 1]
        fname = parts[split_idx + 2]
    except (ValueError, IndexError):
        print("❌ Invalid path structure. Must include /images/train|val/...")
        return

    label_name = fname.rsplit(".", 1)[0] + '.txt'
    label_path = os.path.join(root_dir, "labels", split, label_name)
    mask_path = os.path.join(root_dir, "masks", split, fname)

    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        return
    if not os.path.exists(mask_path):
        print(f"❌ Mask file not found: {mask_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return
    h, w = image.shape[:2]

    vis_img = image.copy()
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        class_id = int(float(parts[0]))
        polygon = list(map(float, parts[5:]))
        points = np.array([[int(x * w), int(y * h)] for x, y in zip(polygon[::2], polygon[1::2])], dtype=np.int32)
        cv2.polylines(vis_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(vis_img, str(class_id), tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    mask = np.array(Image.open(mask_path).convert("L"))
    color_mask = cv2.applyColorMap(cv2.convertScaleAbs(mask, alpha=255.0 / max(mask.max(), 1)), cv2.COLORMAP_JET)

    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)

    # Save side-by-side figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].imshow(color_mask)
    axs[0].set_title("Original Mask")
    axs[0].axis('off')

    axs[1].imshow(vis_img)
    axs[1].set_title("Image + YOLO Seg")
    axs[1].axis('off')

    plt.tight_layout()

    # Save to visualized/<split>/
    save_dir = os.path.join(root_dir, "visualized", split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"✅ Saved visualization to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_seg_label.py <path_to_image>")
    else:
        draw_yolo_segmentation(sys.argv[1])
