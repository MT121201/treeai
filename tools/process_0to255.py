from PIL import Image
import numpy as np
import os

def convert_zeros_to_ignore(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if not fname.endswith('.png'):
            continue

        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        label = np.array(Image.open(src_path))

        # Replace 0 with 255
        label[label == 0] = 255

        Image.fromarray(label.astype(np.uint8)).save(dst_path)

    print(f"Converted all 0s to 255 in: {src_dir} → {dst_dir}")

# === Example Usage ===
convert_zeros_to_ignore("/home/a3ilab01/treeai/det_tree/segmentation/34_RGB_SemSegm_640_pL/unprocessd_annotations/val", 
"/home/a3ilab01/treeai/det_tree/segmentation/34_RGB_SemSegm_640_pL/annotations/val")
