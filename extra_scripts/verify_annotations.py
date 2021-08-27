import json
from pycocotools.coco import COCO
import argparse


def get_ca_od_annotations(ann_file_path, img_name):
    updated_file_contents = {"info": [], "licenses": [], "images": [], "annotations": [],
                             "categories": [{'supercategory': 'object', 'id': 1, 'name': 'object'}]}
    images_dict = {}
    annotations_dict = {}
    file_path = ann_file_path
    coco = COCO(file_path)
    for key in coco.imgs.keys():
        image = coco.imgs[key]
        image_name = image["file_name"]
        if image_name == img_name:
            if image_name not in annotations_dict.keys():
                annotations_dict[image_name] = []
            images_dict[image_name] = image
            image_id = image["id"]
            ann_ids = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(ann_ids)
            annotations_dict[image_name].append(annotations)

    for i, ann in enumerate(annotations_dict[img_name]):
        for j, key in enumerate(images_dict.keys()):
            image = images_dict[key]
            image["id"] = j
            annotations = ann
            for a in annotations:
                a["image_id"] = j
            updated_file_contents["images"].append(image)
            updated_file_contents["annotations"] += annotations
        with open(f"{img_name.split('.')[0]}_{i}.json", "w") as f:
            json.dump(updated_file_contents, f)


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--annotation_file_path", required=True,
                    help="")
    ap.add_argument("-img", "--image_name", required=True,
                    help="")
    args = vars(ap.parse_args())

    return args


if __name__ == "__main__":
    args = parse_arguments()
    ann_file_path = args["annotation_file_path"]
    image_name = args["image_name"]

    get_ca_od_annotations(ann_file_path, image_name)
