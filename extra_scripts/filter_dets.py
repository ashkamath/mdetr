import os
import shutil

all_task_test_images_path = "/home/maaz/PycharmProjects/VOC_EVAL/all_task_images"
all_dets_path = "/home/maaz/PycharmProjects/VOC_EVAL/dets_from_diff_methods/deep_mask/deep_mask_dets"
output_path = "/home/maaz/PycharmProjects/VOC_EVAL/dets_from_diff_methods/deep_mask/deep_mask_all_task_dets"
if not os.path.exists(output_path):
    os.makedirs(output_path)


if __name__ == "__main__":
    images = os.listdir(all_task_test_images_path)
    for image in images:
        det_path = f"{all_dets_path}/{image.split('.')[0]}.txt"
        out_path = f"{output_path}/{image.split('.')[0]}.txt"
        shutil.copy(det_path, out_path)
