""" Run the demo on given image """
from ultralytics import YOLO
import os

model = YOLO('yolov8n_car_plate.pt')
# results = model.predict(source="demo2.jpg", save=True)

image_dir = 'photo_showcase_data'
results_list = []
# results = model.predict(source="IMG_0861.MOV", show=True)

# Iterate over all images in the directory
for image_file in os.listdir(image_dir):
    # Construct full image path
    image_path = os.path.join(image_dir, image_file)

    # Perform object detection
    results = model.predict(source=image_path, save=True)

    # Append results to the results list
    results_list.append(results)

for result in results_list:
    # Bounding boxes
    # boxes = result.boxes.xyxy  # Box coordinates in (x1, y1, x2, y2) format
    # Class probabilities
    # probs = result.boxes.conf  # Confidence scores
    # Class labels
    labels = result.names[result.boxes.cls]  # Class labels
    # Print bounding boxes, class probabilities, and class labels
    # print('Boxes:', boxes)
    # print('Probabilities:', probs)
    print('Labels:', labels)
