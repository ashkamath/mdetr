# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Utilities to manipulate and convert boxes"""
from collections import defaultdict
from typing import Any, Dict

import torch
from torchvision.ops.boxes import box_iou

from .unionfind import UnionFind


def obj_to_box(obj: Dict[str, Any]):
    """Extract the bounding box of a given object as a list"""
    return [obj["x"], obj["y"], obj["w"], obj["h"]]


def region_to_box(obj: Dict[str, Any]):
    """Extract the bounding box of a given region as a list"""
    return [obj["x"], obj["y"], obj["width"], obj["height"]]


def get_boxes_equiv(orig_boxes, iou_threshold):
    """Given a set of boxes, returns a dict containing clusters of boxes that are highly overlapping.
    For optimization, return None if none of the boxes are overlapping
    A high overlap is characterized by the iou_threshold
    Boxes are expected as [top_left_x, top_left_y, width, height]
    """
    boxes = torch.as_tensor(orig_boxes, dtype=torch.float)
    # Convert to (x,y,x,y) format
    boxes[:, 2:] += boxes[:, :2]
    ious = box_iou(boxes, boxes)
    uf = UnionFind(len(boxes))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if ious[i][j] >= iou_threshold:
                uf.unite(i, j)
    if len(orig_boxes) == uf.nb_compo:
        # We didn't found any opportunity for merging, returning as-is
        # print("no merging")
        return None, None
    # print("merging")
    compo2boxes = defaultdict(list)
    compo2id = defaultdict(list)

    for i in range(len(boxes)):
        compo2boxes[uf.find(i)].append(boxes[i])
        compo2id[uf.find(i)].append(i)
    assert len(compo2boxes) == uf.nb_compo
    return compo2boxes, compo2id


def xyxy_to_xywh(boxes: torch.Tensor):
    """Converts a set of boxes in [top_left_x, top_left_y, bottom_right_x, bottom_right_y] format to
    [top_left_x, top_left_y, width, height] format"""
    assert boxes.shape[-1] == 4
    converted = boxes.clone()
    converted[..., 2:] -= converted[..., :2]
    return converted


def combine_boxes(orig_boxes, iou_threshold=0.7):
    """Given a set of boxes, returns the average of all clusters of boxes that are highly overlapping.
    A high overlap is characterized by the iou_threshold
    Boxes are expected as [top_left_x, top_left_y, width, height]
    """
    compo2boxes, _ = get_boxes_equiv(orig_boxes, iou_threshold)
    if compo2boxes is None:
        return orig_boxes
    result_boxes = []
    for box_list in compo2boxes.values():
        result_boxes.append(xyxy_to_xywh(torch.stack(box_list, 0).mean(0)).tolist())
    return result_boxes


def box_iou_helper(b1, b2):
    """returns the iou matrix between two sets of boxes
    The boxes are expected in the format [top_left_x, top_left_y, w, h]
    """
    boxes_r1 = torch.as_tensor(b1, dtype=torch.float)
    # Convert to (x,y,x,y) format
    boxes_r1[:, 2:] += boxes_r1[:, :2]
    boxes_r2 = torch.as_tensor(b2, dtype=torch.float)
    # Convert to (x,y,x,y) format
    boxes_r2[:, 2:] += boxes_r2[:, :2]
    return box_iou(boxes_r1, boxes_r2)
