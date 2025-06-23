from pathlib import Path
import cv2
import argparse
import json
import numpy as np
import imagesize
from create_annotations import (
    create_image_annotation,
    create_annotation_from_yolo_format,
    create_annotation_from_yolo_results_format,
    coco_format,
)

YOLO_DARKNET_SUB_DIR = "YOLO_darknet"

CATEGORY_IDS = [
    3, 5, 6, 9, 11, 12, 13, 15, 17, 20,
    24, 25, 26, 30, 35, 36, 40, 48, 49, 50,
    51, 52, 53, 54, 56, 57, 58, 59, 60, 61
]
classes = [f"cls_{i}" for i in CATEGORY_IDS]


def get_images_info_and_annotations(opt):
    path = Path(opt.path)
    annotations = []
    images_annotations = []
    if path.is_dir():
        file_paths = sorted(path.rglob("*.jpg")) + sorted(path.rglob("*.jpeg")) + sorted(path.rglob("*.png"))
    else:
        with open(path, "r") as fp:
            file_paths = [Path(line.strip()) for line in fp.readlines()]

    image_id = 0
    annotation_id = 1

    for file_path in file_paths:
        print("\rProcessing " + str(image_id) + " ...", end='')

        w, h = imagesize.get(str(file_path))
        image_annotation = create_image_annotation(
            file_path=file_path, width=w, height=h, image_id=image_id
        )
        images_annotations.append(image_annotation)

        label_file_name = f"{file_path.stem}.txt"
        annotations_path = file_path.parent / (YOLO_DARKNET_SUB_DIR if opt.yolo_subdir else '') / label_file_name

        if annotations_path.exists():
            with open(str(annotations_path), "r") as label_file:
                label_read_line = label_file.readlines()

            for line1 in label_read_line:
                parts = line1.strip().split()
                if not parts or len(parts) < 5:
                    continue
                original_class_id = int(parts[0])
                if original_class_id not in CATEGORY_IDS:
                    continue
                category_id = CATEGORY_IDS.index(original_class_id) + 1

                x_center, y_center, width, height = map(float, parts[1:5])
                float_x_center = w * x_center
                float_y_center = h * y_center
                float_width = w * width
                float_height = h * height

                min_x = int(float_x_center - float_width / 2)
                min_y = int(float_y_center - float_height / 2)
                width = int(float_width)
                height = int(float_height)

                if opt.results and len(parts) >= 6:
                    conf = float(parts[5])
                    annotation = create_annotation_from_yolo_results_format(
                        min_x, min_y, width, height, image_id, category_id, conf
                    )
                else:
                    annotation = create_annotation_from_yolo_format(
                        min_x, min_y, width, height, image_id, category_id, annotation_id,
                        segmentation=opt.box2seg,
                    )
                annotations.append(annotation)
                annotation_id += 1

        image_id += 1

    return images_annotations, annotations


def get_args():
    parser = argparse.ArgumentParser("Yolo format annotations to COCO dataset format")
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default="train_coco.json", type=str)
    parser.add_argument("--yolo-subdir", action="store_true")
    parser.add_argument("--box2seg", action="store_true")
    parser.add_argument("--results", action="store_true")
    return parser.parse_args()


def main(opt):
    output_path = f"{opt.output}"
    print("Start!")
    coco_format["images"], coco_format["annotations"] = get_images_info_and_annotations(opt)

    for idx, label in enumerate(classes):
        coco_format["categories"].append({
            "supercategory": "Tree",
            "id": idx + 1,
            "name": label,
        })

    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"Finished! Output saved to {output_path}")


if __name__ == "__main__":
    options = get_args()
    main(options)
