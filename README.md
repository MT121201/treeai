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
```

### YOLO
Refer to this [yolo_installation](/doc/yolo_installation.md)

### mm
Refer to this [mm_installation](/doc/mm_installation.md)
## Dataset
Make sure git-lfs is installed (https://git-lfs.com)
```bash
git lfs install
git clone https://huggingface.co/datasets/MinTR-KIEU/det_tree
```

### Convert masks to YOLO txt
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
