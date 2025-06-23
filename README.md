# Solution for TreeAI Global Initiative

## Installation
### MiniConda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
```bash
source ~/miniconda3/bin/activate
```
```bash
conda init --all
```mk

### YOLO
Refer to this [yolo_installation](/doc/yolo_installation.md)

### mm
Refer to this [mm_installation](/doc/mm_installation.md)
## Dataset
Make sure git-lfs is installed (https://git-lfs.com)
```bash
sudo apt install git-lfs
```

```bash
git lfs install
git clone https://huggingface.co/datasets/MinTR-KIEU/det_tree
```

## Detection Task
### Convert YOLO Dataset to COCO Annotation

Prepare your dataset directory with the following structure for both `train` and `val` splits:
```
root/
    ├── 00001.txt
    └── 00001.png
```

To convert the YOLO-format dataset to COCO annotation format, run:
```bash
python tools/yolo_to_coco_converter.py --train <path_to_train_folder> --val <path_to_val_folder> --out <path_to_output_json>
```




## Segmentation Task
### Convert masks to YOLO txt
Require imagesize
```bash
pip install imagesize
```
```bash
python tools/convert_yolo_masks <your_dataset_root_dir>
```

### Visualize txt label
```bash
python python tools/visuallize_seg_label.py <path to choosen image>
```

### Merge dataset for training (Only if all correct YOLO dataset structure)
Only support correct YOLO dataset structure, if not please check `scripts/format.sh`
Please check class ID before merge
Please edit dataset path in code file before run
```bash
python tools/merge_dataset.py 
```

## Training
### Segmetation Model
```bash
conda activate yolo11
```
