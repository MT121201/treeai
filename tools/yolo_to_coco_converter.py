from pathlib import Path
import argparse
import json
import imagesize
from create_annotations import (
    create_image_annotation,
    create_annotation_from_yolo_format,
)

def collect_class_ids(label_paths):
    used_class_ids = set()
    for label_path in label_paths:
        with open(label_path, "r") as f:
            for line in f:
                items = line.strip().split()
                if len(items) < 5:
                    continue
                class_id = int(items[0])
                used_class_ids.add(class_id)
    return sorted(list(used_class_ids))

def get_annotations(image_dir, class_map, box2seg=False):
    image_dir = Path(image_dir)
    annotations = []
    images_annotations = []
    image_id = 0
    annotation_id = 1

    image_paths = sorted(image_dir.rglob("*.png")) + \
                  sorted(image_dir.rglob("*.jpg")) + \
                  sorted(image_dir.rglob("*.jpeg"))

    for file_path in image_paths:
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

                orig_class_id = int(items[0])
                if orig_class_id not in class_map:
                    continue

                coco_class_id = class_map[orig_class_id]

                x, y, bw, bh = map(float, items[1:5])
                abs_x = w * x
                abs_y = h * y
                abs_bw = w * bw
                abs_bh = h * bh
                x_min = int(abs_x - abs_bw / 2)
                y_min = int(abs_y - abs_bh / 2)

                annotation = create_annotation_from_yolo_format(
                    x_min, y_min, int(abs_bw), int(abs_bh),
                    image_id, coco_class_id, annotation_id,
                    segmentation=box2seg,
                )
                annotations.append(annotation)
                annotation_id += 1

        image_id += 1

    return images_annotations, annotations

def create_coco_structure():
    return {
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

def main(opt):
    train_path = Path(opt.train)
    val_path = Path(opt.val)
    output_dir = Path(opt.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get class IDs only from training data
    train_labels = sorted(train_path.rglob("*.txt"))
    used_class_ids = collect_class_ids(train_labels)
    class_map = {orig_id: new_id for new_id, orig_id in enumerate(used_class_ids)}

    # Create common categories section
    categories = [
        {"id": coco_id, "name": f"cls_{orig_id}", "supercategory": "Tree"}
        for coco_id, orig_id in enumerate(used_class_ids)
    ]

    for split_name, path in [("train", train_path), ("val", val_path)]:
        coco_format = create_coco_structure()
        coco_format["categories"] = categories
        coco_format["images"], coco_format["annotations"] = get_annotations(path, class_map, box2seg=opt.box2seg)

        with open(output_dir / f"{split_name}.json", "w") as f:
            json.dump(coco_format, f, indent=4)

        print(f"\nFinished writing: {output_dir / f'{split_name}.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training image directory path")
    parser.add_argument("--val", required=True, help="Validation image directory path")
    parser.add_argument("--out", required=True, help="Directory to save train.json and val.json")
    parser.add_argument("--box2seg", action="store_true", help="Use bbox to create segmentation")
    args = parser.parse_args()
    main(args)
