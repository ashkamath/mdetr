import json
import os
from pycocotools.coco import COCO
import time
import numpy as np
import argparse
from PIL import Image

OUTPUT_DIR = "../"
IMAGES_DIR = ""


def filter_annotations(dir_path):
    updated_file_contents = {"info": [], "licenses": [], "images": [], "annotations": [],
                             "categories": [{'supercategory': 'object', 'id': 1, 'name': 'object'}]}
    images_dict = {}
    annotations_dict = {}
    files = os.listdir(dir_path)
    start = time.time()
    for file in files:
        file_path = f"{dir_path}/{file}"
        coco = COCO(file_path)
        with open(file_path) as f:
            file_contents = json.load(f)
        images = (file_contents['images']).copy()
        print(f"File {file_path.split('/')[-1]} has total images: {len(file_contents['images'])}")
        for image in images:
            image_name = image["file_name"]
            if image_name not in annotations_dict.keys():
                annotations_dict[image_name] = []
            images_dict[image_name] = image
            image_id = image["id"]
            ann_ids = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(ann_ids)
            annotations_dict[image_name] += annotations
        print(f"It took {time.time() - start} seconds to process the file {file}.")
        start = time.time()
    print(f"Assigning ids to images and annotations. Also correcting the width and height values in the annotations.")
    start = time.time()
    # Assign new ids to the images and adjust the annotations accordingly
    for i, key in enumerate(images_dict.keys()):
        image = images_dict[key]
        image["id"] = i
        annotations = annotations_dict[key]
        annotations = class_agnostic_nms(annotations)
        for a in annotations:
            a["image_id"] = i
        # Correct the width and height data types
        image_path = f"{IMAGES_DIR}/{image['file_name']}"
        im = Image.open(image_path)
        width, height = im.size
        image["width"] = int(width)
        image["height"] = int(height)
        updated_file_contents["images"].append(image)
        updated_file_contents["annotations"] += annotations
    # Adjust the 'id' of annotations
    for i, ann in enumerate(updated_file_contents["annotations"]):
        ann["id"] = i
    print(f"It took {time.time() - start} seconds for assigning Ids and correcting width and height values.")
    print(f"Correcting category ids.")
    start = time.time()
    annotations = file_contents["annotations"]
    counter = 0
    for ann in annotations:
        if ann["category_id"] != 1:
            ann["category_id"] = 1
            counter += 1
    print(f"Correcting {counter} category ids took {time.time() - start} seconds.")
    print(f"Saving class agnostic object detection json (COCO format) file.")
    start = time.time()
    with open(f"{OUTPUT_DIR}/mdetr_ca_od_train.json", "w") as f:
        json.dump(updated_file_contents, f)
    print(f"It took {time.time() - start} seconds to save the annotation file.")


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir_path", required=True,
                    help="Path to the input directory containing json files.")
    ap.add_argument("-o", "--output_dir_path", required=True,
                    help="Path to the output directory for storing the filtered annotations.")
    args = vars(ap.parse_args())

    return args


if __name__ == "__main__":
    args = parse_arguments()
    input_dir_path = args["input_dir_path"]
    output_dir_path = args["output_dir_path"]
    OUTPUT_DIR = output_dir_path
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    get_ca_od_annotations(input_dir_path)
