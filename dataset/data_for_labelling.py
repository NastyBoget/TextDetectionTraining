import json
import os

import cv2
from doctr.models import detection_predictor
from tqdm import tqdm

data_dir = "/Users/anastasiabogatenkova/Downloads/passports/main_page_rotated"  # directory with images to label
out_dir = "/Users/anastasiabogatenkova/Downloads/passports/labeled"  # directory for output json files with predicted labels
os.makedirs(out_dir, exist_ok=True)

text_detector = detection_predictor(arch='db_resnet50', pretrained=True)

# {
#    "name": "1.jpg",
#    "entities": [
#        {
#            "label": "handwritten",
#            "x": 0.1,
#            "y": 0.1,
#            "width": 0.01,
#            "height": 0.01
#        },
#        { ... },
#        ...
#    ]
# }

for img_name in tqdm(os.listdir(data_dir)):
    if not img_name[0].isdigit():
        continue
    try:
        img = cv2.imread(os.path.join(data_dir, img_name))
        result = text_detector([img])
    except AttributeError:
        continue
    labeled_item = {
        "name": img_name,
        "entities": []
    }

    for bbox in result[0]:
        entity = {
            "label": "t",
            "x": bbox[0],
            "y": bbox[1],
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1]
        }
        labeled_item["entities"].append(entity)
    with open(os.path.join(out_dir, f"{img_name}.json"), "w") as f:
        json.dump(labeled_item, f)
