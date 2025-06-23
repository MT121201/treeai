from pathlib import Path
from create_annotations import (
    create_image_annotation,
    create_annotation_from_yolo_format,
    create_annotation_from_yolo_results_format,
)
import cv2
import argparse
import json
import numpy as np
import imagesize

# Define COCO skeleton
coco_format = {
    "info": {
        "description": "Tree AI Dataset",
        "version": "1.0",
        "year": 2025,
        "contributor": "MT1212",
        "date_created": "2025-06-23"
    },
    "images": [],
    "annotations": [],
    "categories": []
}

# Match these exactly to the YOLO label class index (0-based)
classes = [
    "cls_3", "cls_5", "cls_6", "cls_9", "cls_11", "cls_12", "cls_13",
    "cls_15", "cls_17", "cls_20", "cls_24", "cls_25", "cls_26", "cls_30",
    "cls_35", "cls_36", "cls_40", "cls_48", "cls_49", "cls_50", "cls_51",
    "cls_52", "cls_53", "cls_54", "cls_56", "cls_57", "cls_58", "cls_59",
    "cls_60", "cls_61"
]

def get_images_info_and_annotations(opt):
    path = Path(opt.path)
    annotations = []
    images_annotations = []
    image_id = 0
    annotation_id = 1

    file_paths = sorted(path.rglob("*.png")) + sorted(path.rglob("*.jpg")) + sorted(path.rglob("*.jpeg"))

    for file_path in file_paths:
        print(f"\rProcessing {file_path.name}", end='')
        w, h = imagesize.get(str(file_path))
        images_annotations.append(create_image_annotation(file_path, w, h, image_id))

        label_path = file_path.with_suffix(".txt")
        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                items = line.strip().split()
                if len(items) < 5:
                    continue

                class_id = int(items[0])
                if class_id >= len(classes):
                    continue

                x, y, bw, bh = map(float, items[1:5])
                abs_x = w * x
                abs_y = h * y
                abs_bw = w * bw
                abs_bh = h * bh
                x_min = int(abs_x - abs_bw / 2)
                y_min = int(abs_y - abs_bh / 2)

                annotation = create_annotation_from_yolo_format(
                    x_min, y_min, int(abs_bw), int(abs_bh),
                    image_id, class_id, annotation_id,
                    segmentation=opt.box2seg,
                )
                annotations.append(annotation)
                annotation_id += 1

        image_id += 1

    return images_annotations, annotations

def main(opt):
    coco_format["images"], coco_format["annotations"] = get_images_info_and_annotations(opt)

    for idx, name in enumerate(classes):
        coco_format["categories"].append({
            "id": idx,
            "name": name,
            "supercategory": "Tree"
        })

    with open(opt.output, "w") as f:
        json.dump(coco_format, f, indent=4)
    print("\nFinished writing:", opt.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Image directory path")
    parser.add_argument("--output", default="train_coco.json", help="Output annotation file")
    parser.add_argument("--box2seg", action="store_true", help="Use bbox to create segmentation")
    args = parser.parse_args()
    main(args)
