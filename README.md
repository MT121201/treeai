Here is a **cleaned-up and well-formatted version** of your TreeAI solution instructions for both Detection and Segmentation:

---

# 🌳 TreeAI Global Initiative: Solution Guide

---

## 🧭 Detection Task

### 🛠️ Installation

Please refer to the full installation guide:
📄 [`detection_installation.md`](/home/a3ilab01/treeai/doc/detection_installation.md)

---

### 🚀 Inference

Make sure you're inside the `mmdetection` directory:

```bash
cd mmdetection/
```

Optional: Download pretrained weights for faster inference:

```bash
mkdir -p weights && \
wget -O weights/multiscale_12.pth https://huggingface.co/datasets/MinTR-KIEU/det_tree/resolve/main/weights/multiscale_12.pth
```

Run prediction:

```bash
python tools/prediction.py --test_dir <path_to_test_images>
```

---

### 🏋️ Training

Use the distributed training script:

```bash
bash tools/dist_train.sh <path_to_config> <num_gpus>
```

📌 Example:

```bash
bash tools/dist_train.sh configs/_custom_/finetune_12.py 2
```

---

## 🎨 Segmentation Task

### 🛠️ Installation

Please refer to the full installation guide:
📄 [`segmentation_installation.md`](/home/a3ilab01/treeai/doc/segmentation_installation.md)

---

### 🚀 Inference

Make sure you're inside the `mmsegmentation` directory:

```bash
cd mmsegmentation/
```

Optional: Download pretrained weights:

```bash
mkdir -p weights && \
wget -O weights/seg_37.pth https://huggingface.co/datasets/MinTR-KIEU/det_tree/resolve/main/weights/seg_37.pth
```

Run prediction:

```bash
python tools/test_npy.py --test_dir <path_to_test_images>
```

---

### 🏋️ Training

Use the distributed training script:

```bash
bash tools/dist_train.sh <path_to_config> <num_gpus>
```

📌 Example:

```bash
bash tools/dist_train.sh configs/_custom_/segformer2.py 2
```
