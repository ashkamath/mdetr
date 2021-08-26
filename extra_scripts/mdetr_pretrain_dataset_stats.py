import os


TASK1_KNOWN_CLASSES = []
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
TASK4_KNOWN_CLASSES = []


mdetr_text_dir_path = "/data/output/mdetr_train_doc_dict"
output_dir = "/data/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file_path = f"{output_dir}/mdetr_pretrain_stats.txt"


def main():
    total_datapoints = 0
    filtered_datapoints = 0
    files = os.listdir(mdetr_text_dir_path)
    imgs = []
    for file in files:
        # print(f"On file {file}")
        mdetr_text_file_path = f"{mdetr_text_dir_path}/{file}"
        # print(f"Reading text from file.")
        with open(mdetr_text_file_path, "r") as f:
            mdetr_train_text = f.read()
        captions = mdetr_train_text.split('\n')
        # print(f"There are total {len(captions)} captions in the file.")
        for c in captions:
            if c == '':
                continue
            else:
                total_datapoints += 1
            text = c.split(',')
            img = text[0]
            caption = text[1]
            if not TASK4_KNOWN_CLASSES:
                imgs.append(img)
            else:
                flag = False
                for cls in TASK4_KNOWN_CLASSES:
                    if cls in caption:
                        flag = True
                        break
                # flag=True -> caption contains unknown class, flag=False -> caption doesn't contain unknown class
                if flag:
                    filtered_datapoints += 1
                    imgs.append(img)

    print(f"Total Datapoints: {total_datapoints}")
    print(f"Filtered Datapoints: {filtered_datapoints}")
    print(f"There are total {len(imgs)} captions in the dataset.")
    print(f"There are total {len(set(imgs))} images in the dataset.")

    unique_imgs = set(imgs)
    coco_image = 0
    for i in unique_imgs:
        if 'COCO' in i:
            coco_image += 1
    print(f"Total images from coco datasets are: {coco_image}")


if __name__ == "__main__":
    main()
