# Export segmentation model
from ultralytics import YOLO
import os

# Use Forward Slashes
seg_model = YOLO("models/yolov8n-seg.pt")

seg_model_path = "models/yolov8n-seg_openvino_model/yolov8n-seg.xml"
if not os.path.exists(seg_model_path):
    seg_model.export(format="openvino", dynamic=True, half=False)
