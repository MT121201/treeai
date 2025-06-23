import shutil
from pathlib import Path

# Define root paths
root = Path('/home/a3ilab01/treeai/det_tree/12_RGB_ObjDet_640_fL')
label_dirs = [root / 'labels/train', root / 'labels/val']
image_dirs = [root / 'images/train', root / 'images/val']

# Copy each .txt label to corresponding image directory
for label_dir, img_dir in zip(label_dirs, image_dirs):
    label_files = list(label_dir.glob('*.txt'))
    for label_file in label_files:
        target = img_dir / label_file.name
        shutil.copy(label_file, target)

print("✅ All labels copied to corresponding image folders.")
