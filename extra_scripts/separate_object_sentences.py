import json
import os
import spacy
import time

nlp = spacy.load("en_core_web_sm")  # make sure to use larger package!

annotation_dir_path = "/data/final_annotations"
output_dir = "/data/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
annotation_files = os.listdir(annotation_dir_path)

for file in annotation_files:
    file_path = f"{annotation_dir_path}/{file}"
    with open(file_path) as f:
        file_contents = json.load(f)
    images = file_contents['images']
    filtered_object_captions = {}
    start = time.time()
    for i, image in enumerate(images):
        if i > 0 and i % 10000 == 0:
            print(f"On image: {i}. Time: {time.time() - start}")
            with open(f"{output_dir}/object_captions.txt", "a") as f:
                for key in filtered_object_captions.keys():
                    f.write(f"{key}, {filtered_object_captions[key]}\n")
            filtered_object_captions = {}
            start = time.time()
        file_name = image['file_name']
        caption = image['caption']
        tokenized_caption = nlp(caption)
        for token in tokenized_caption:
            if "object" in token.text:
                filtered_object_captions[file_name] = caption
    with open(f"{output_dir}/object_captions.txt", "a") as f:
        for key in filtered_object_captions.keys():
            f.write(f"{key}, {filtered_object_captions[key]}\n")

    print(file_contents.keys())
