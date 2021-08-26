import json
from pycocotools.coco import COCO
import numpy as np

original_json_path = "/data/final_annotations/final_flickr_mergedGT_train.json"
combined_json_path = "/data/output/tmp/final_flickr_mergedGT_train_filtered.json"

with open(original_json_path) as f:
    original_json = json.load(f)
original_json_coco = COCO(original_json_path)

with open(combined_json_path) as f:
    combined_json = json.load(f)
combined_json_coco = COCO(combined_json_path)

# assert len(original_json["images"]) == len(combined_json["images"])
# assert len(original_json["annotations"]) == len(combined_json["annotations"])

for i in range(10):
    j = np.random.randint(0, len(original_json["images"]))
    image = original_json["images"][j]
    image_id = image["id"]

    original_ann_ids = original_json_coco.getAnnIds(imgIds=image_id)
    original_targets = original_json_coco.loadAnns(original_ann_ids)

    combined_ann_ids = combined_json_coco.getAnnIds(imgIds=image_id)
    combined_targets = original_json_coco.loadAnns(combined_ann_ids)

    assert original_targets == combined_targets

print("Success")
