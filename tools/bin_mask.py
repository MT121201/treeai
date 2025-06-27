import os
import numpy as np
from PIL import Image

def convert_to_binary_mask(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for split in ['train', 'val']:
        input_dir = os.path.join(input_root, split)
        output_dir = os.path.join(output_root, split)
        os.makedirs(output_dir, exist_ok=True)

        for fname in os.listdir(input_dir):
            if not fname.lower().endswith('.png'):
                continue

            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)

            # Load image as grayscale
            mask = np.array(Image.open(input_path).convert("L"))

            # Binary conversion: 0 stays 0, >0 becomes 1
            binary_mask = (mask > 0).astype(np.uint8)

            # Save as binary PNG (0 and 1)
            Image.fromarray(binary_mask).save(output_path)

    print(f"✅ Conversion complete. Output saved to: {output_root}")

# Example usage:
convert_to_binary_mask("/home/a3ilab01/treeai/dataset/segmentation/full/annotations_0", "/home/a3ilab01/treeai/dataset/segmentation/full/bin_mask")
