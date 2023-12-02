""" Train YOLOv8 model on Car Plate Dataset """
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
data_path = "Car_License_Plates-1/data.yaml"
results = model.train(epochs=10, data=data_path, imgsz=640)
