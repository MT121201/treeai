import os
from PIL import Image

# Set paths
img_dir = '/home/a3ilab01/treeai/dataset/segmentation/full/images/val'
mask_dir = '/home/a3ilab01/treeai/dataset/segmentation/full/annotations/val'

# Track corrupted files
bad_files = []

# Check and remove bad images and corresponding masks
for filename in os.listdir(img_dir):
    img_path = os.path.join(img_dir, filename)
    mask_path = os.path.join(mask_dir, filename)  # assumes same name

    try:
        with Image.open(img_path) as img:
            img.verify()
    except Exception as e:
        bad_files.append(filename)
        os.remove(img_path)
        if os.path.exists(mask_path):
            os.remove(mask_path)

# Print out the removed filenames
print("Removed Files:")
for f in bad_files:
    print(f)
