""" Run the demo on given image """
from ultralytics import YOLO

model = YOLO('yolov8n_car_plate.pt')
results = model.predict(source="demo.jpeg", save=True)
