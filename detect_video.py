""" Run the demo on given video """
from ultralytics import YOLO

model = YOLO('yolov8n_car_plate.pt')
results = model.predict(source="demo.mp4", show=True)
