import json
from PIL import Image

coco_json_path = "/home/maaz/PycharmProjects/mdetr/data/mdetr_ca_od_train.json"
images_dir_path = ""

with open(coco_json_path) as f:
    file_contents = json.load(f)

annotations = file_contents["annotations"]

counter = 0
for ann in annotations:
    category_id = ann["category_id"]
    if category_id != 1:
        ann["category_id"] = 1
        counter += 1

print(f"Total {counter} annotations with wrong category ids are corrected.")

print(f"Correcting the width and heights (str to int).")

images = file_contents["images"]
for img in images:
    image_path = f"{images_dir_path}/{img['file_name']}"
    im = Image.open(image_path)
    width, height = im.size
    img["width"] = int(width)
    img["height"] = int(height)

print(f"Saving annotations")
with open(f"{coco_json_path.split('.')[0]}_corrected.json", 'w') as f:
    json.dump(file_contents, f)
print('')
