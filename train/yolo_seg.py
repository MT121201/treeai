from ultralytics import YOLO

# Load model (replace with your custom model path if needed)
model = YOLO("yolo11n-seg.pt")

# Train with custom settings
results = model.train(
    data="/home/a3ilab01/treeai/dataset/merged_seg_dataset/data.yaml",  
    epochs=100,
    imgsz=640,
    batch=16,                
    name="yolo11n_seg_exp",  
    project="runs/seg",      
    device=0                
)
