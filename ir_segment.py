from ultralytics import YOLO
from pathlib import Path
from IPython.display import Image
from PIL import Image

SEG_MODEL_NAME = "yolov8n-seg"

IMAGE_PATH = Path("data\coco_bike.jpg")
seg_model = YOLO("models\yolov8n-seg.pt")
res = seg_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:, :, ::-1])