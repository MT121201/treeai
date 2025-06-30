Here is a **refactored and cleaner version** of your installation guide for MMSegmentation:

---

# 🧠 MMSegmentation Setup Guide (CUDA 11.8 + PyTorch 2.1)

## 🐍 Step 1: Install MiniConda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

---

## 🧪 Step 2: Create & Activate Environment

```bash
conda create -n mmseg211 python=3.10 -y
conda activate mmseg211
```

---

## ⚙️ Step 3: Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 Step 4: Install MMCV & MMEngine

```bash
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

---

## 🔧 Step 5: System Dependencies

Install OpenCV support:

```bash
sudo apt install -y libgl1
```

Fix NumPy compatibility (optional):

```bash
pip install numpy==1.24.4
```

---

## 📁 Step 6: Install MMSegmentation

```bash
git clone https://github.com/MT121201/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
pip install ftfy regex
```

* `-e`: install in editable mode (modifications take effect immediately)
* `-v`: verbose output

---

## ✅ Step 7: Verify the Installation

### Download config & checkpoint:

```bash
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
```

### Run a demo:

```bash
python demo/image_demo.py demo/demo.png \
    configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
    --device cuda:0 --out-file result.jpg
```

> ⚠️ You may see a visualization warning (about `vis_backend`); this is safe to ignore.

