# mm installation
## env
```bash
conda create -n mmdet211 python=3.10 -y
conda activate mmdet211
```

## Torch
Recommended: CUDA 11.8, PyTorch 2.1.0
```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

## MMCV & MMEngine
```bash
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
  # use compatible version for MM<4.x.
```

## MMDetection
```bash
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
### libgl1
```bash
sudo apt install -y libgl1
```
### Immediate Fix: Downgrade NumPy to a safe version
```bash
pip install numpy==1.24.4
```

This is the latest stable NumPy 1.x version that's compatible with:
- PyTorch 2.1.0
- MMCV full 2.x
- MMDetection 3.3.0

## Verify the installation
We need to download config and checkpoint files.
```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

Verify the inference demo.
```bash
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```