from ultralytics import YOLO
from pathlib import Path
from IPython.display import Image
from PIL import Image

IMAGE_PATH = Path("data\coco_bike.jpg")

# There are three lines of code below, all are correct.
# You can use one of them

# Use Raw String Literal:
# det_model = YOLO(r"models\best.pt")

# Use Forward Slashes
# det_model = YOLO("models/best.pt")

# Use Double Backslashes:
det_model = YOLO("models\\best.pt")
label_map = det_model.model.names

res = det_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:, :, ::-1])