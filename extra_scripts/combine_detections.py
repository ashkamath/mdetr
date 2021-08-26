import os
import numpy as np

ann_dir_path = "/home/maaz/PycharmProjects/VOC_EVAL/dets_from_diff_methods/mdetr_dets_tq"
text_queries = ['all_small_objects', 'all_medium_objects', 'all_large_objects']
output_dir = "/home/maaz/PycharmProjects/VOC_EVAL/dets_from_diff_methods/mdetr_dets_tq/combine_11"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
nms_iou_threshold = 0.5


def nms(dets, scores, thresh):
    """
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep], scores[keep]


def class_agnostic_nms(boxes, scores, iou=0.7):
    if len(boxes) > 1:
        # boxes = non_max_suppression_fast(np.array(boxes), iou)
        boxes, scores = nms(np.array(boxes), np.array(scores), iou)
        return list(boxes), list(scores)
    else:
        return boxes, scores


def parse_det_txt(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
        boxes = []
        scores = []
        for line in lines:
            content = line.rstrip().split(' ')
            bbox = content[2:]
            boxes.append([int(b) for b in bbox])
            scores.append(content[1])
        return boxes, scores
    else:
        return [], []


def main():
    det_files = os.listdir(f"{ann_dir_path}/{text_queries[0]}")
    for i, file in enumerate(det_files):
        if i % 1000 == 0:
            print(f"On file no. {i}")
        out_file_path = f"{output_dir}/{file}"
        all_boxes = []
        all_scores = []
        for q in text_queries:
            file_path = f"{ann_dir_path}/{q}/{file}"
            boxes, scores = parse_det_txt(file_path)
            all_boxes += boxes
            all_scores += scores
        filtered_boxes, filtered_scores = class_agnostic_nms(all_boxes, all_scores, nms_iou_threshold)
        with open(out_file_path, "w") as f:
            for b, s in zip(filtered_boxes, filtered_scores):
                f.write(f"{0} {s} {int(b[0])} {int(b[1])} {int(b[2])} {int(b[3])}\n")


if __name__ == "__main__":
    main()
