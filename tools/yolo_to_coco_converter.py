
import os
import json
import cv2
from tqdm import tqdm
from pathlib import Path

def load_categories(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def yolo_to_coco(image_dir, label_dir, categories, output_json):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    ann_id = 1
    image_id = 1
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    category_ids = set([cat["id"] for cat in categories])

    for img_path in tqdm(image_files, desc=f"Processing {image_dir.name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        height, width = img.shape[:2]

        coco_output["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })

        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    if class_id not in category_ids:
                        continue
                    x_center, y_center, w, h = map(float, parts[1:])
                    x = (x_center - w / 2) * width
                    y = (y_center - h / 2) * height
                    bbox = [x, y, w * width, h * height]
                    area = bbox[2] * bbox[3]

                    coco_output["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id += 1

        image_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=2)

    print(f"Saved: {output_json}")

if __name__ == "__main__":
    # Customize these paths
    categories_json = "det_tree/34_RGB_ObjDet_640_pL/coco_categories.json"
    yolo_base = "det_tree/34_RGB_ObjDet_640_pL"
    out_base = "det_tree/34_RGB_ObjDet_640_pL/annotations"

    os.makedirs(out_base, exist_ok=True)
    categories = load_categories(categories_json)

    yolo_to_coco(f"{yolo_base}/images/train", f"{yolo_base}/labels/train", categories, f"{out_base}/train.json")
    yolo_to_coco(f"{yolo_base}/images/val", f"{yolo_base}/labels/val", categories, f"{out_base}/val.json")
