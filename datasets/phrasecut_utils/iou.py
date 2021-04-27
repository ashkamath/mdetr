# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Direct import from https://github.com/ChenyunWu/PhraseCutDataset/blob/b15fb71a1ba692ea3186498f1390e8854b681a66/utils/iou.py
"""

from .data_transfer import *


# IoU function
def iou_box(box1, box2, xywh=True, ioubp=False):
    # each box is of [x1, y1, w, h]
    if not xywh:
        [box1, box2] = xyxy_to_xywh([box1, box2])

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    s1 = box1[2] * box1[3]
    s2 = box2[2] * box2[3]
    union = s1 + s2 - inter
    if not ioubp:
        return float(inter) / union
    else:
        return float(inter) / union, float(inter) / s1, float(inter) / s2


def iou_boxes(boxes1, boxes2, w=0, h=0, xywh=True, ioubp=False, iandu=False):
    if w == 0 or h == 0:
        region = boxes_region(list(boxes1) + list(boxes2), xywh)
        w = int(region[2] + 1)
        h = int(region[3] + 1)
    m1 = boxes_to_mask(boxes1, w, h, xywh)
    m2 = boxes_to_mask(boxes2, w, h, xywh)

    i = np.sum((np.logical_and(m1, m2) > 0), axis=None)
    u = np.sum((m1 + m2) > 0, axis=None)
    if i == 0:
        iou = 0
    else:
        iou = i * 1.0 / u
    if not ioubp and not iandu:
        return iou
    out = [iou]
    if ioubp:
        if i == 0:
            out += [0, 0]
        else:
            b = np.sum(m1 > 0, axis=None)
            p = np.sum(m2 > 0, axis=None)
            out += [i * 1.0 / b, i * 1.0 / p]
    if iandu:
        out += [i, u]
    return out


def iou_boxes_polygons(boxes, polygons, w=0, h=0, xywh=True, ioubp=False):
    # tic = time.time()
    if w * h == 0:
        p_boxes = [polygon_to_box(p) for p in polygons]
        region = boxes_region(p_boxes + list(boxes))
        w = int(region[2] + 1)
        h = int(region[3] + 1)

    p_mask = polygons_to_mask(polygons, w, h)

    b_mask = boxes_to_mask(boxes, w, h, xywh)

    i = np.sum((np.logical_and(p_mask, b_mask) > 0), axis=None)
    u = np.sum((p_mask + b_mask) > 0, axis=None)
    # toc = time.time()
    # print('iou_time', toc - tic)
    if not ioubp:
        if i == 0:
            return 0
        else:
            return i * 1.0 / u
    else:
        if i == 0:
            return 0, 0, 0
        else:
            b = np.sum(b_mask > 0, axis=None)
            p = np.sum(p_mask > 0, axis=None)
            return i * 1.0 / u, i * 1.0 / b, i * 1.0 / p


def iou_mask(m1, m2, ioubp=False):
    assert m1.shape == m2.shape
    i = np.sum((np.logical_and(m1, m2) > 0), axis=None)
    u = np.sum(np.logical_or(m1, m2) > 0, axis=None)
    if not ioubp:
        if i == 0:
            return 0
        else:
            return i * 1.0 / u
    else:
        if i == 0:
            return 0, 0, 0
        else:
            b = np.sum(m1 > 0, axis=None)
            p = np.sum(m2 > 0, axis=None)
            return i * 1.0 / u, i * 1.0 / b, i * 1.0 / p


def iou_boxes_mask(boxes, mask, xywh=True, ioubp=False):
    w, h = mask.shape
    b_mask = boxes_to_mask(boxes, w, h, xywh)
    i = np.sum((np.logical_and(mask, b_mask) > 0), axis=None)
    u = np.sum((mask + b_mask) > 0, axis=None)
    if not ioubp:
        if i == 0:
            return 0
        else:
            return i * 1.0 / u
    else:
        if i == 0:
            return 0, 0, 0
        else:
            b = np.sum(b_mask > 0, axis=None)
            p = np.sum(mask > 0, axis=None)
            return i * 1.0 / u, i * 1.0 / b, i * 1.0 / p


def iou_polygons(ps1, ps2, w=0, h=0, ioubp=False):
    if w * h == 0:
        xyxy = boxes_region([polygon_to_box(p) for p in ps1 + ps2])
        w = int(xyxy[2] + 1)
        h = int(xyxy[3] + 1)
    m1 = polygons_to_mask(ps1, w, h)
    m2 = polygons_to_mask(ps2, w, h)
    i = np.sum((np.logical_and(m1, m2) > 0), axis=None)
    u = np.sum((m1 + m2) > 0, axis=None)
    if not ioubp:
        if i == 0:
            return 0
        else:
            return i * 1.0 / u
    else:
        if i == 0:
            return 0, 0, 0
        else:
            b = np.sum(m1 > 0, axis=None)
            p = np.sum(m2 > 0, axis=None)
            return i * 1.0 / u, i * 1.0 / b, i * 1.0 / p


def iou_polygons_masks(ps, masks, ioubp=False, iandu=False, gt_size=False):
    h, w = masks[0].shape
    mps = polygons_to_mask(ps, w, h)
    mask = np.sum(masks, axis=0)
    i = np.sum((np.logical_and(mps, mask) > 0), axis=None)
    u = np.sum((mps + mask) > 0, axis=None)
    if i == 0:
        iou = 0
    else:
        iou = i * 1.0 / u
    if not ioubp and not iandu and not gt_size:
        return iou

    out = [iou]
    if ioubp:
        if i == 0:
            out += [0, 0]
        else:
            b = np.sum(mps > 0, axis=None)
            p = np.sum(mask > 0, axis=None)
            out += [i * 1.0 / b, i * 1.0 / p]
    if iandu:
        out += [i, u]
    if gt_size:
        b = np.sum(mps > 0, axis=None)
        s = b * 1.0 / (w * h)
        out.append(s)
    return out
