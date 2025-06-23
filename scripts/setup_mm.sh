#!/bin/bash

# OPTIONAL: Create new clean conda env
conda create -n mmdet211 python=3.10 -y
conda activate mmdet211

# 1. Install PyTorch 2.1.0 + CUDA 11.8
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install compatible NumPy (avoid NumPy 2.x errors)
pip install numpy==1.24.4

# 3. Install OpenMMLab tools
pip install -U openmim
mim install mmengine

# 4. Install MMCV
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# 5. Clone MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v3.3.0

# 6. Install MMDetection
pip install -v -e .

# 7. Download example config + weights
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .

# 8. Run demo
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py \
    --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
    --device cpu
