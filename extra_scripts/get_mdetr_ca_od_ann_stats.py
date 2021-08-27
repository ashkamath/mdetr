import os
import json
from pycocotools.coco import COCO
import cv2


def draw_boxes(boxes):
    image_path = "/home/maaz/PycharmProjects/mdetr/data/2351314.jpg"
    image = cv2.imread(image_path)
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1
    for b in boxes:
        cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[0])+int(b[2]), int(b[1])+int(b[3])), color, thickness)
    cv2.imwrite("/home/maaz/PycharmProjects/mdetr/data/COCO_train2014_000000385646_draw.jpg", image)


def main():
    count = 0
    file_path = "/home/maaz/PycharmProjects/mdetr/data/mdetr_ca_od_train.json"
    file_contents = COCO(file_path)
    image_to_ann = {}
    for key in file_contents.imgs.keys():
        image = file_contents.imgs[key]
        ann_ids = file_contents.getAnnIds(imgIds=image['id'])
        target = file_contents.loadAnns(ann_ids)
        if len(target) > 50:
            count += 1
        image_to_ann[image["file_name"]] = len(target)

    print(count)


if __name__ == "__main__":
    main()
