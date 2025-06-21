#!/bin/bash

TARGET_DIR="/home/mt/Dataset/TreeAI_det/34_RGB_SemSegm_640_pL"

# Create YOLO structure
mkdir -p $TARGET_DIR/images/train
mkdir -p $TARGET_DIR/images/val
mkdir -p $TARGET_DIR/labels/train
mkdir -p $TARGET_DIR/labels/val

# Move data into YOLO structure
mv $TARGET_DIR/train/images/* $TARGET_DIR/images/train/
mv $TARGET_DIR/train/labels/* $TARGET_DIR/labels/train/
mv $TARGET_DIR/val/images/* $TARGET_DIR/images/val/
mv $TARGET_DIR/val/labels/* $TARGET_DIR/labels/val/

# Remove original folders
rm -r $TARGET_DIR/train
rm -r $TARGET_DIR/val