# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Directly imported from
https://github.com/ChenyunWu/PhraseCutDataset/blob/b15fb71a1ba692ea3186498f1390e8854b681a66/utils/data_transfer.py
"""
import numpy as np
from PIL import Image, ImageDraw


# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    if boxes is None or len(boxes) == 0:
        return boxes
    boxes = np.array(boxes)
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    if boxes is None or len(boxes) == 0:
        return boxes
    boxes = np.array(boxes)
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def boxes_region(boxes, xywh=True):
    """
    :return: [x_min, y_min, x_max, y_max] of all boxes
    """
    if xywh:
        boxes = xywh_to_xyxy(boxes)
    boxes = np.array(boxes)
    min_xy = np.min(boxes[:, :2], axis=0)
    max_xy = np.max(boxes[:, 2:], axis=0)
    return [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]


def polygon_to_box(polygon):
    x1 = 1e8
    y1 = 1e8
    x2 = y2 = 0
    for point in polygon:
        x1 = point[0] if point[0] < x1 else x1
        x2 = point[0] if point[0] > x2 else x2
        y1 = point[1] if point[1] < y1 else y1
        y2 = point[1] if point[1] > y2 else y2
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    return [x1, y1, w, h]


def polygons_to_mask(polygons, w, h):
    p_mask = np.zeros((h, w))
    for polygon in polygons:
        if len(polygon) < 2:
            continue
        p = []
        for x, y in polygon:
            p.append((int(x), int(y)))
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).polygon(p, outline=1, fill=1)
        mask = np.array(img)
        p_mask += mask
    p_mask = p_mask > 0
    return p_mask


def boxes_to_mask(boxes, w, h, xywh=True):
    b_mask = np.zeros((h, w))
    if xywh:
        boxes = xywh_to_xyxy(boxes)
    for box in boxes:
        x1, y1, x2, y2 = box
        b_mask[int(y1) : int(y2), int(x1) : int(x2)] = 1
    return b_mask


def polygon_in_box(polygon, box, xywh=True):
    """
    Output polygon is the intersect region of given polygon and box
    """

    def point_in_box(p, b):
        # b: xyxy
        return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]

    def point_in_polygon(point, polygon):
        pb = polygon_to_box(polygon)
        pb = xywh_to_xyxy([pb])[0]
        if not point_in_box(point, pb):
            return False
        p_mask = polygons_to_mask([polygon], int(pb[0] + pb[2] + 1), int(pb[1] + pb[3] + 1))
        is_in = p_mask[point[0], point[1]]
        almost_in = np.mean(p_mask.astype(float)[point[0] - 1 : point[0] + 1, point[1] - 1 : point[1] + 1])
        return max(is_in, almost_in)

    def points_in_line(p1, p2):
        return p1[0] == p2[0] or p1[1] == p2[1]

    def points_intersect_box(p0, p1, b):
        def point_in_seg(p0, p1, pt):
            if not min(p0[0], p1[0]) <= pt[0] <= max(p0[0], p1[0]):
                return False
            if not min(p0[1], p1[1]) <= pt[1] <= max(p0[1], p1[1]):
                return False
            # if p0==pt or p1 == pt:
            #     return False
            return True

        # b: xyxy
        valid_ps = []
        if p0[1] != p1[1]:
            k = (p0[0] - p1[0]) * 1.0 / (p0[1] - p1[1])
            x1 = k * (b[1] - p0[1]) + p0[0]
            vp = [x1, b[1]]
            if point_in_box(vp, b) and point_in_seg(p0, p1, vp):
                valid_ps.append(vp)
            x2 = k * (b[3] - p0[1]) + p0[0]
            vp = [x2, b[3]]
            if point_in_box(vp, b) and point_in_seg(p0, p1, vp):
                valid_ps.append(vp)
        if p0[0] != p1[0]:
            k = (p0[1] - p1[1]) * 1.0 / (p0[0] - p1[0])
            y1 = k * (b[0] - p0[0]) + p0[1]
            vp = [b[0], y1]
            if point_in_box(vp, b) and point_in_seg(p0, p1, vp):
                valid_ps.append(vp)
            y2 = k * (b[2] - p0[0]) + p0[1]
            vp = [b[2], y2]
            if point_in_box(vp, b) and point_in_seg(p0, p1, vp):
                valid_ps.append(vp)
        if len(valid_ps) > 1:
            valid_ps.sort(key=lambda q: abs(q[0] - p0[0]))
        return valid_ps

    if xywh:
        box = xywh_to_xyxy([box])[0]

    start_i = -1
    for i, p in enumerate(polygon):
        if point_in_box(p, box):
            start_i = i
            break
    if start_i < 0:
        return None

    polygon_n = []
    out_ps = []
    p_out = None
    for p in polygon[start_i:] + polygon[: start_i + 1]:
        if point_in_box(p, box):
            if not p_out:
                polygon_n.append(p)
            else:
                inter_ps = points_intersect_box(out_ps[-1], p, box)
                if len(inter_ps) < 1:
                    print(box)
                    print(out_ps)
                    print(p)
                    print(inter_ps)
                assert len(inter_ps) >= 1
                p_in = inter_ps[0]
                if points_in_line(p_in, p_out):
                    polygon_n += [p_out, p_in]
                else:
                    to_add = []
                    has_in = False
                    for pt in [[box[0], box[1]], [box[0], box[3]], [box[2], box[1]], [box[2], box[3]]]:
                        is_in = point_in_polygon(pt, out_ps + [p_in, p_out])
                        assert 0 <= is_in <= 1
                        if is_in == 1:
                            has_in = True
                        if is_in > 0:
                            s = 0
                            if points_in_line(pt, p_out):
                                s = -1
                            if points_in_line(pt, p_in):
                                s = 1
                            to_add.append((pt, s, is_in))
                    if has_in:
                        to_add = [x for x in to_add if x[2] == 1]
                        to_add.sort(key=lambda x: x[1])
                    else:
                        to_add.sort(key=lambda x: -x[2])
                        to_add = [to_add[0]]
                    if not to_add:
                        print("to_add", box)
                        print("to_add", out_ps + [p_in, p_out])
                        print("to_add", to_add)
                    assert to_add
                    polygon_n += [p_out] + [x[0] for x in to_add] + [p_in]
                #     print('to_add', to_add)
                # print('p_out', p_out)
                # print('p_in', p_in)
                # print('out_ps', out_ps)
                # print('polygon_n', polygon_n)
                polygon_n.append(p)
                p_out = None
                out_ps = []
        else:
            if not p_out:
                inter_ps = points_intersect_box(polygon_n[-1], p, box)
                if len(inter_ps) < 1:
                    print(box)
                    print(polygon_n)
                    print(p)
                    print(inter_ps)
                assert len(inter_ps) >= 1
                p_out = inter_ps[-1]
                out_ps.append(p)
            else:
                inter_ps = points_intersect_box(out_ps[-1], p, box)
                if inter_ps:
                    if len(inter_ps) != 2:
                        print(box)
                        print(out_ps)
                        print(p)
                        print(inter_ps)
                    assert len(inter_ps) == 2
                    polygon_n += inter_ps
                    out_ps = [p]
                    p_out = inter_ps[-1]
                else:
                    out_ps.append(p)

    return polygon_n[:-1]
