import json
import argparse
from pathlib import Path

def clean_annotations(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    original_count = len(data["annotations"])
    data["annotations"] = [
        ann for ann in data["annotations"]
        if ann["bbox"][2] > 0 and ann["bbox"][3] > 0 and ann["bbox"][2] * ann["bbox"][3] > 1
    ]
    removed = original_count - len(data["annotations"])

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[{file_path.name}] Removed {removed} invalid annotations → Overwritten.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_dir", help="Directory containing train.json and val.json")
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    for name in ["train.json", "val.json"]:
        json_path = annotation_dir / name
        if json_path.exists():
            clean_annotations(json_path)
        else:
            print(f"[!] {name} not found in {annotation_dir}")

if __name__ == "__main__":
    main()
