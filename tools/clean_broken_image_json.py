import os
import json
from PIL import Image

# === MODIFY THESE PATHS ===
img_dir = '/home/a3ilab01/treeai/dataset/x/0_RGB_include_3/coco/images/val'
ann_path = '/home/a3ilab01/treeai/dataset/x/0_RGB_include_3/coco/annotations/val.json'
output_ann_path = ann_path  # overwrite in-place, or change this if needed

# === Load COCO annotation ===
with open(ann_path, 'r') as f:
    coco = json.load(f)

valid_images = []
valid_image_ids = set()
removed_image_filenames = []

for img_info in coco['images']:
    img_path = os.path.join(img_dir, img_info['file_name'])
    try:
        with Image.open(img_path) as img:
            img.verify()  # Only checks header
        valid_images.append(img_info)
        valid_image_ids.add(img_info['id'])
    except Exception as e:
        print(f"[BROKEN] {img_info['file_name']} → {e}")
        removed_image_filenames.append(img_info['file_name'])
        # Optionally remove the broken image file
        if os.path.exists(img_path):
            os.remove(img_path)

# === Filter annotations for valid image ids only ===
valid_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in valid_image_ids]

# === Update COCO dict ===
coco['images'] = valid_images
coco['annotations'] = valid_annotations

# === Save cleaned annotation file ===
with open(output_ann_path, 'w') as f:
    json.dump(coco, f)

print(f"\n✅ Done. Removed {len(removed_image_filenames)} broken images and their annotations.")
