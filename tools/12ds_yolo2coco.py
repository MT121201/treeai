import json
import argparse
from pathlib import Path
import imagesize
from create_annotations import (
    create_image_annotation,
    create_annotation_from_yolo_format,
)

def get_annotations(image_dir, class_shift=1, box2seg=False):
    image_dir = Path(image_dir)
    annotations = []
    images_annotations = []
    image_id = 0
    annotation_id = 1

    # Only allow class IDs 1–53 (1-based index)
    valid_class_indices = set(range(1, 54))  # 1 to 53 inclusive

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

                raw_class_id = int(items[0])
                if raw_class_id not in valid_class_indices:
                    continue  # Skip unwanted classes

                coco_class_id = raw_class_id - class_shift  # 1-based to 0-based
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
        "betula papyrifera", "tsuga canadensis", "picea abies", "acer saccharum",
        "betula sp.", "pinus sylvestris", "picea rubens", "betula alleghaniensis",
        "larix decidua", "fagus grandifolia", "picea sp.", "fagus sylvatica",
        "dead tree", "acer pensylvanicum", "populus balsamifera", "quercus ilex",
        "quercus robur", "pinus strobus", "larix laricina", "larix gmelinii",
        "pinus pinea", "populus grandidentata", "pinus montezumae", "abies alba",
        "betula pendula", "pseudotsuga menziesii", "fraxinus nigra",
        "dacrydium cupressinum", "cedrus libani", "acer pseudoplatanus",
        "pinus elliottii", "cryptomeria japonica", "pinus koraiensis",
        "abies holophylla", "alnus glutinosa", "fraxinus excelsior", "coniferous",
        "eucalyptus globulus", "pinus nigra", "quercus rubra", "tilia europaea",
        "abies firma", "acer sp.", "metrosideros umbellata", "acer rubrum",
        "picea mariana", "abies balsamea", "castanea sativa", "tilia cordata",
        "populus sp.", "crataegus monogyna", "quercus petraea", "acer platanoides"
    ]

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
    parser.add_argument("--box2seg", action="store_true", help="Use bbox to create segmentation")
    args = parser.parse_args()
    main(args)
