import os
import glob
import yaml

def collect_classes(label_dir):
    class_ids = set()
    for file in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    try:
                        class_id = int(float(parts[0]))
                        class_ids.add(class_id)
                    except ValueError:
                        continue
    return class_ids

def generate_data_yaml(train_label_dir, val_label_dir, image_root_dir, save_path="data.yaml"):
    train_classes = collect_classes(train_label_dir)
    val_classes = collect_classes(val_label_dir)
    all_class_ids = train_classes.union(val_classes)

    if not all_class_ids:
        raise ValueError("❌ No class IDs found in the dataset!")

    max_class_id = max(all_class_ids)
    nc = max_class_id + 1  # YOLO requires names list to include all indices up to max ID

    names = []
    for i in range(nc):
        if i in all_class_ids:
            names.append(f'class_{i}')
        else:
            names.append('unused')

    data_yaml = {
        'train': os.path.join(image_root_dir, 'train'),
        'val': os.path.join(image_root_dir, 'val'),
        'nc': nc,
        'names': names
    }

    with open(save_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"✅ Generated {save_path} with nc={nc}")
    print(f"📋 Classes present: {sorted(all_class_ids)}")
    print(f"🗃️  Missing class IDs (filled with 'unused'): {[i for i in range(nc) if i not in all_class_ids]}")

# Example usage
if __name__ == "__main__":
    generate_data_yaml(
        train_label_dir="dataset/merged_seg_dataset/labels/train",
        val_label_dir="dataset/merged_seg_dataset/labels/val",
        image_root_dir="dataset/merged_seg_dataset/images",
        save_path="dataset/merged_seg_dataset/data.yaml"
    )
