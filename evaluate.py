
""" Evaluate trained YOLOv8 Car Plate etection model """
from ultralytics import YOLO

model = YOLO('yolov8n_car_plate.pt')
model.val()
