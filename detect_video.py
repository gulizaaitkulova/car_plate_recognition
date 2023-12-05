""" Run the demo on given video """
from ultralytics import YOLO
import os

video_dir = 'video_showcase_data'
results_list = []
model = YOLO('yolov8n_car_plate.pt')
# results = model.predict(source="IMG_0861.MOV", show=True)

# Iterate over all images in the directory
for video_file in os.listdir(video_dir):
    # Construct full image path
    video_path = os.path.join(video_dir, video_file)

    # Perform object detection
    results = model.predict(source=video_path, save=True)

    # Append results to the results list
    results_list.append(results)
