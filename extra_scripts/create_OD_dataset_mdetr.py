import json
import os


annotation_dir_path = "/data/final_annotations"
output_dir = "/data/output/tmp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
annotation_files = os.listdir(annotation_dir_path)

for file in annotation_files:
    file_path = f"{annotation_dir_path}/{file}"
    with open(file_path) as f:
        file_contents = json.load(f)
    print("")
