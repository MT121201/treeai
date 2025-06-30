## 🧪 Detection Installation Guide

### ✅ MiniConda Installation

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Activate MiniConda:

```bash
source ~/miniconda3/bin/activate
```

Initialize conda for all supported shells:

```bash
conda init --all
```

---

### 🧱 Create Environment

```bash
conda create -n mmdet211 python=3.10 -y
conda activate mmdet211
```

---

### ⚙️ Install PyTorch (CUDA 11.8, PyTorch 2.1.0)

```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 🔧 Install MMCV & MMEngine

```bash
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

> ⚠️ Make sure MMCV and MMEngine are compatible with MMDetection 3.x+

---

### 📦 Install MMDetection

```bash
git clone https://github.com/MT121201/mmdetection.git
cd mmdetection
pip install -v -e .
```

> `-v`: verbose mode
> `-e`: editable mode (code changes take effect without reinstall)

---

### 🧩 Install libGL (for OpenCV)

```bash
sudo apt install -y libgl1
```

---

### ⚠️ Fix: Downgrade NumPy (if needed)

```bash
pip install numpy==1.24.4
```

---

### ✅ Verify the Installation

Download a test config and model checkpoint:

```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

Run a quick inference test:

```bash
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py \
    --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```
