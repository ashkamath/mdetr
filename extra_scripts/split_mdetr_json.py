import json
import os
from pycocotools.coco import COCO

annotation_dir_path = "/data/final_annotations"
output_dir = "/data/output/tmp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
annotation_files = os.listdir(annotation_dir_path)
json_chunk_size = 10000

for file in annotation_files:
    file_path = f"{annotation_dir_path}/{file}"
    coco = COCO(file_path)
    with open(file_path) as f:
        file_contents = json.load(f)
    updated_file_contents = {"info": file_contents["info"], "licenses": file_contents["licenses"],
                             "images": [], "annotations": [], "categories": file_contents["categories"]}
    for i, image in enumerate(file_contents["images"]):
        if i % json_chunk_size == 0:
            print(f"Saving {output_dir}/{file.split('.')[0]}_{i}.json")
            with open(f"{output_dir}/{file.split('.')[0]}_{i}.json", "w") as f:
                json.dump(updated_file_contents, f)
            updated_file_contents = {"info": file_contents["info"], "licenses": file_contents["licenses"],
                                     "images": [], "annotations": [], "categories": file_contents["categories"]}
        updated_file_contents["images"].append(image)
        ann_ids = coco.getAnnIds(imgIds=image['id'])
        target = coco.loadAnns(ann_ids)
        for t in target:
            updated_file_contents['annotations'].append(t)
    with open(f"{output_dir}/{file.split('.')[0]}_{i}.json", "w") as f:
        json.dump(updated_file_contents, f)
