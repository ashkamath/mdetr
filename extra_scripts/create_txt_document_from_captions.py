import json
import os
import time

annotation_dir_path = "/data/final_annotations"
output_dir = "/data/output/tmp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
annotation_files = os.listdir(annotation_dir_path)

for file in annotation_files:
    file_path = f"{annotation_dir_path}/{file}"
    with open(file_path) as f:
        file_contents = json.load(f)
    images = file_contents['images']
    image_list = []
    caption_list = []
    start = time.time()
    for i, image in enumerate(images):
        if i > 0 and i % 10000 == 0:
            print(f"On image: {i}. Time: {time.time() - start}")
            with open(f"{output_dir}/mdetr_train_doc.txt", "a") as f:
                for img, c in zip(image_list, caption_list):
                    f.write(f"{img},{c}\n")
            image_list = []
            caption_list = []
            start = time.time()
        file_name = image['file_name']
        caption = image['caption']
        image_list.append(file_name)
        caption_list.append(image['caption'])
    with open(f"{output_dir}/mdetr_train_doc.txt", "a") as f:
        for img, c in zip(image_list, caption_list):
            f.write(f"{img},{c}\n")

    print("Finished")
