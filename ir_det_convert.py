# object detection model
from ultralytics import YOLO
import os

# Use Forward Slashes
det_model = YOLO("models/best.pt")

det_model_path = "models/best_openvino_model/best.xml"
if not os.path.exists(det_model_path):
    det_model.export(format="openvino", dynamic=True, half=False)
