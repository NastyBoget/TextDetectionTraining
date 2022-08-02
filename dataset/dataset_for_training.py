import json
import random
import os
import zipfile
from copy import deepcopy

import cv2
import numpy as np

import wget as wget

from adaptive_docreader_binarizer import Binarizer


def binarize(img: np.ndarray, img_name: str, out_dir: str, target_labels: dict, bboxes: list) -> None:
    # block_size=40 delta=40
    # block_size=45 delta=50
    # block_size=15 delta=40
    binarizer_list = [Binarizer(block_size=40, delta=40),
                      Binarizer(block_size=45, delta=50),
                      Binarizer(block_size=15, delta=40)]
    for i, binarizer in enumerate(binarizer_list):
        img_name_wo_ext, ext = os.path.splitext(img_name)

        binarized_img_name = f"{img_name_wo_ext}_{i}{ext}"
        img_copy = deepcopy(img)
        img_copy = binarizer.binarize(img_copy)
        target_labels[binarized_img_name] = {"polygons": bboxes}
        cv2.imwrite(os.path.join(out_dir, binarized_img_name), img_copy)


def main(args) -> None:

    data_url = args.data_url
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    target_dir = os.path.join(data_dir, "text_detection")

    if not os.path.isdir(target_dir):
        os.makedirs(data_dir, exist_ok=True)
        archive = os.path.join(data_dir, "archive.zip")
        print(f"Downloading text detection dataset...")
        wget.download(data_url, archive)
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Dataset downloaded")
        os.remove(archive)

    # input data:
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

    # img.jpg
    # test_img.json
    # img.jpg.json


    # output data:
    # {"img_name": {
    #     "polygons":
    #         [[[x1, y1], [x2, y2]], [[x1, y1], [x2, y2]], ... ],
    #     },
    #  "img_name": {
    #     "polygons":
    #         [[[x1, y1], [x2, y2]], [[x1, y1], [x2, y2]], ... ],
    #     }, ...
    # }

    # train
    # |
    # |---images
    # |---labels.json
    #
    # val
    # |
    # |---images
    # |---labels.json

    images_dir_name = "images"
    val_dir_name = "val"
    train_dir_name = "train"

    os.makedirs(os.path.join(target_dir, val_dir_name, images_dir_name), exist_ok=True)
    os.makedirs(os.path.join(target_dir, train_dir_name, images_dir_name), exist_ok=True)

    images_file_names = [file_name for file_name in os.listdir(target_dir)
                         if file_name[0].isdigit() and not file_name.endswith("json")]
    random.shuffle(images_file_names)
    images_dict = {train_dir_name: images_file_names[:int(0.9 * len(images_file_names))],
                   val_dir_name: images_file_names[int(0.9 * len(images_file_names)):]}
    # Dataset size: train 537, val 60
    print(f"Dataset size: train {len(images_dict[train_dir_name])}, val {len(images_dict[val_dir_name])}")

    for stage in images_dict:
        target_labels = {}
        for file_name in images_dict[stage]:
            json_file_name = f"{file_name}.json"
            if not os.path.isfile(os.path.join(target_dir, json_file_name)):
                continue
            img = cv2.imread(os.path.join(target_dir, file_name))
            h, w, _ = img.shape
            with open(os.path.join(target_dir, json_file_name), "r") as f:
                labels = json.load(f)
            bboxes = [[[int(w * item["x"]), int(h * item["y"])],
                       [int(w * (item["x"] + item["width"])), int(h * (item["y"] + item["height"]))]]
                      for item in labels["entities"]]
            target_labels[file_name] = {"polygons": bboxes}

            os.rename(os.path.join(target_dir, file_name),
                      os.path.join(target_dir, stage, images_dir_name, file_name))
            if args.augment:
                binarize(img, file_name, os.path.join(target_dir, stage, images_dir_name), target_labels, bboxes)

        with open(os.path.join(target_dir, stage, "labels.json"), "w") as f:
            json.dump(target_labels, f)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Dataset for training creation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_url', type=str, help='URL for the dataset downloading')
    parser.add_argument('--augment', dest='augment', action='store_true', help='Augment data using binarization')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
