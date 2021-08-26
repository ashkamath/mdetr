import json
import os
from pycocotools.coco import COCO
import time
import argparse
import spacy

OUTPUT_DIR = "../"

TASK1_KNOWN_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                       "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                       "pottedplant", "sheep", "sofa", "train", "tvmonitor", "airplane", "dining table", "motorcycle",
                       "potted plant", "couch", "tv"]

TASK2_KNOWN_CLASSES = TASK1_KNOWN_CLASSES + ["truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
                                             "bench", "elephant", "bear", "zebra", "giraffe",
                                             "backpack", "umbrella", "handbag", "tie", "suitcase",
                                             "microwave", "oven", "toaster", "sink", "refrigerator"]
TASK3_KNOWN_CLASSES = TASK2_KNOWN_CLASSES + ["frisbee", "skis", "snowboard", "sports ball", "kite",
                                             "baseball bat", "baseball glove", "skateboard", "surfboard",
                                             "tennis racket",
                                             "banana", "apple", "sandwich", "orange", "broccoli",
                                             "carrot", "hot dog", "pizza", "donut", "cake"]
TASK4_KNOWN_CLASSES = TASK3_KNOWN_CLASSES + ["bed", "toilet", "laptop", "mouse",
                                             "remote", "keyboard", "cell phone", "book", "clock",
                                             "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
                                             "wine glass", "cup", "fork", "knife", "spoon", "bowl"]
TASK1_UNKNOWN_CLASSES = [x for x in TASK4_KNOWN_CLASSES if x not in TASK1_KNOWN_CLASSES]

nlp = spacy.load("en_core_web_md")  # make sure to use larger package!


def filter_mdetr_annotation(file_path):
    filtered_file_contents = {"info": [], "licenses": [], "images": [], "annotations": [],
                              "categories": [{'supercategory': 'object', 'id': 1, 'name': 'object'}]}
    coco = COCO(file_path)
    with open(file_path) as f:
        file_contents = json.load(f)
    images = (file_contents['images']).copy()
    print(f"File {file_path.split('/')[-1]} has total images: {len(file_contents['images'])}")
    start = time.time()
    for i, image in enumerate(images):
        if i % 10000 == 0:
            print(f"On image {i}, time: {time.time() - start} seconds")
            start = time.time()
        caption = image['caption']
        tokens = nlp(caption)
        flag = False
        for cls in TASK1_UNKNOWN_CLASSES:
            for t in tokens:
                if nlp(cls).similarity(t) > 0.8:
                    flag = True
                    break
        if not flag:
            filtered_file_contents["images"].append(image)
            ann_ids = coco.getAnnIds(imgIds=image['id'])
            target = coco.loadAnns(ann_ids)
            filtered_file_contents["annotations"] += target

    with open(f"{OUTPUT_DIR}/{os.path.basename(file_path)}", "w") as f:
        json.dump(filtered_file_contents, f)


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

    for file in os.listdir(input_dir_path):
        file_path = f"{input_dir_path}/{file}"
        if file_path.endswith('.json'):
            try:
                filter_mdetr_annotation(file_path)
            except:
                pass
