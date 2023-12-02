""" Download datset from Roboflow """
from roboflow import Roboflow


rf = Roboflow(api_key="FkTP5RH4RRwJAEKLrVEH")
project = rf.workspace("moin").project("car_license_plates")

dataset1 = project.version(1).download("yolov8")
