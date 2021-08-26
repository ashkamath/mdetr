import os
import pickle

dino_dir_path = "/home/maaz/PycharmProjects/VOC_EVAL/dets_from_diff_methods/dino/dino_boxes"
all_task_imgs_dir_path = "/home/maaz/PycharmProjects/VOC_EVAL/all_task_images"


if __name__ == "__main__":
    total_boxes = 0
    files = os.listdir(all_task_imgs_dir_path)
    total_files = len(files)
    for file in files:
        dino_file_path = f"{dino_dir_path}/{file.split('.')[0]}.pkl"
        with open(dino_file_path, "rb") as f:
            boxes = pickle.load(f)
        for key in boxes.keys():
            total_boxes += len(boxes[key])
    print(f"Total Files: {total_files}\nTotal Boxes:: {total_boxes}\nBoxes per File: {total_boxes/total_files}")
