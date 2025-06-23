### YOLO
```bash
conda create -n yolo11 python=3.10 -y
conda activate yolo11
```

```bash
# Install all packages together using conda
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```


### libgl1
```bash
sudo apt install -y libgl1
```