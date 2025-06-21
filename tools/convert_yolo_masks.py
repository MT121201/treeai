import os
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

def convert_single_mask(mask_path, output_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    height, width = mask.shape
    class_ids = np.unique(mask)
    class_ids = class_ids[class_ids != 0]

    with open(output_path, 'w') as f:
        for class_id in class_ids:
            binary_mask = (mask == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3:
                    continue
                contour = contour.squeeze(1)
                x_coords = contour[:, 0]
                y_coords = contour[:, 1]

                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                x_center = ((x_min + x_max) / 2) / width
                y_center = ((y_min + y_max) / 2) / height
                bbox_width = (x_max - x_min) / width
                bbox_height = (y_max - y_min) / height

                polygon = []
                for x, y in zip(x_coords, y_coords):
                    polygon.append(x / width)
                    polygon.append(y / height)

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f} " +
                        " ".join(f"{pt:.6f}" for pt in polygon) + "\n")

def convert_all_masks(root_dir):
    for split in ['train', 'val']:
        label_dir = os.path.join(root_dir, 'labels', split)
        mask_dir = os.path.join(root_dir, 'masks', split)
        os.makedirs(mask_dir, exist_ok=True)

        for fname in tqdm(os.listdir(label_dir), desc=f"Processing {split}"):
            if fname.endswith('.png'):
                mask_path = os.path.join(label_dir, fname)
                label_path = os.path.join(label_dir, fname.replace('.png', '.txt'))

                convert_single_mask(mask_path, label_path)
                shutil.move(mask_path, os.path.join(mask_dir, fname))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_yolo_masks.py <root_folder>")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Directory does not exist: {root}")
        sys.exit(1)

    convert_all_masks(root)
    print("✅ All masks converted and moved successfully.")
