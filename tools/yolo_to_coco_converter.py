import json
import argparse
from pathlib import Path
import imagesize
from create_annotations import (
    create_image_annotation,
    create_annotation_from_yolo_format,
)

def load_global_class_list(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_annotations(image_dir, class_shift=1, box2seg=False):
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
                    raise ValueError(f"Invalid label format in {label_path}: {line.strip()}")
                    break

                coco_class_id = int(items[0]) - class_shift  # shift 1-based to 0-based
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

    global_class_list = [
  "Species_1", "Species_2", "Species_3", "Species_4", "Species_5",
  "Species_6", "Species_7", "Species_8", "Species_9", "Species_10",
  "Species_11", "Species_12", "Species_13", "Species_14", "Species_15",
  "Species_16", "Species_17", "Species_18", "Species_19", "Species_20",
  "Species_21", "Species_22", "Species_23", "Species_24", "Species_25",
  "Species_26", "Species_27", "Species_28", "Species_29", "Species_30",
  "Species_31", "Species_32", "Species_33", "Species_34", "Species_35",
  "Species_36", "Species_37", "Species_38", "Species_39", "Species_40",
  "Species_41", "Species_42", "Species_43", "Species_44", "Species_45",
  "Species_46", "Species_47", "Species_48", "Species_49", "Species_50",
  "Species_51", "Species_52", "Species_53", "Species_54", "Species_55",
  "Species_56", "Species_57", "Species_58", "Species_59", "Species_60",
  "Species_61", "Species_62", "Species_63", "Species_64"]


    categories = [
        {"id": i, "name": global_class_list[i], "supercategory": "Tree"}
        for i in range(len(global_class_list))
    ]

    for split_name, path in [("train", train_path), ("val", val_path)]:
        coco_format = create_coco_structure()
        coco_format["categories"] = categories
        coco_format["images"], coco_format["annotations"] = get_annotations(path, box2seg=opt.box2seg)

        with open(output_dir / f"{split_name}.json", "w") as f:
            json.dump(coco_format, f, indent=4)

        print(f"\nFinished writing: {output_dir / f'{split_name}.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training image directory path")
    parser.add_argument("--val", required=True, help="Validation image directory path")
    parser.add_argument("--out", required=True, help="Directory to save train.json and val.json")
    # parser.add_argument("--class", required=True, help="Path to global_class_list.json")
    parser.add_argument("--box2seg", action="store_true", help="Use bbox to create segmentation")
    args = parser.parse_args()
    main(args)
