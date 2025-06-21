#!/bin/bash

# Start YOLOv11-segmentation training (X model)
yolo segment train \
  model=yolov11x-seg.pt \
  data=dataset/merged_seg_dataset/data.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  device=0 \
  project=runs/segment \
  name=yolov11x-seg
