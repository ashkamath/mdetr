import json
import os

annotation_dir_path = "/data/output/tmp"
output_dir = "/home/maaz/PycharmProjects/mdetr/data/output/tmp/combined"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
annotation_files = os.listdir(annotation_dir_path)

combined_file_contents = {"info": [], "licenses": [], "images": [], "annotations": [],
                          "categories": [{'supercategory': 'object', 'id': 1, 'name': 'object'}]}
for file in annotation_files:
    file_path = f"{annotation_dir_path}/{file}"
    if not os.path.isfile(file_path):
        continue
    with open(file_path) as f:
        file_contents = json.load(f)
    for key in ['images', 'annotations']:
        combined_file_contents[key] += file_contents[key]

with open(f"{output_dir}/combined_json.json", "w") as f:
    json.dump(combined_file_contents, f)
