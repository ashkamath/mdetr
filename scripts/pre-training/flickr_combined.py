# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
This script it a utility to convert the Flickr30k entities dataset to a coco format suitable for this repository.
Note that each image-caption pair will yield a single data-point, hence there will be 5x more datapoint than images.

Data pre-processing modified from https://github.com/jnhwkim/ban-vqa/blob/master/dataset.py and
https://github.com/BryanPlummer/flickr30k_entities/blob/68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b/flickr30k_entities_utils.py

"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.etree.ElementTree import parse

import numpy as np
import torch
import xmltodict
from torchvision.ops.boxes import box_area
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--flickr_path",
        required=True,
        type=str,
        help="Path to the flickr dataset",
    )
    parser.add_argument(
        "--out_path",
        default="",
        type=str,
        help="Path where to export the resulting dataset.",
    )

    parser.add_argument(
        "--merge_ground_truth",
        action="store_true",
        help="Whether to follow Bryan Plummer protocol and merge ground truth. By default, all the boxes for an entity are kept separate",
    )

    return parser.parse_args()


def box_xywh_to_xyxy(x):
    """Accepts a list of bounding boxes in coco format (xmin,ymin, width, height)
    Returns the list of boxes in pascal format (xmin,ymin,xmax,ymax)

    The boxes are expected as a numpy array
    """
    result = x.copy()
    result[..., 2:] += result[..., :2]
    return result


def xyxy2xywh(box: List):
    """Accepts a list of bounding boxes in pascal format (xmin,ymin,xmax,ymax)
    Returns the list of boxes in coco format (xmin,ymin, width, height)
    """
    xmin, ymin, xmax, ymax = box
    h = ymax - ymin
    w = xmax - xmin
    return [xmin, ymin, w, h]


#### The following loading utilities are imported from
#### https://github.com/BryanPlummer/flickr30k_entities/blob/68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b/flickr30k_entities_utils.py
# Changelog:
#    - Added typing information
#    - Completed docstrings


def get_sentence_data(filename) -> List[Dict[str, Any]]:
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      filename - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(filename, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data["phrases"].append(
                {"first_word_index": index, "phrase": phrase, "phrase_id": p_id, "phrase_type": p_type}
            )

        annotations.append(sentence_data)

    return annotations


## END of import from flickr tools

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


#### End of import of box utilities


class UnionFind:
    """Optimized union find structure"""

    def __init__(self, n):
        """Initialize a union find with n components"""
        self.compo = list(range(n))
        self.weight = [1] * n
        self.nb_compo = n

    def get_nb_compo(self):
        return self.nb_compo

    def find(self, x):
        if self.compo[x] == x:
            return x
        self.compo[x] = self.find(self.compo[x])
        return self.compo[x]

    def unite(self, a, b):
        fa = self.find(a)
        fb = self.find(b)
        if fa != fb:
            self.nb_compo -= 1
            if self.weight[fb] > self.weight[fa]:
                fa, fb = fb, fa
            self.compo[fb] = fa
            self.weight[fa] += self.weight[fb]


def get_equivalent_boxes(all_boxes, iou_threshold=0.95):
    """Find clusters of highly overlapping boxes
    Parameters:
        - all_boxes: a list of boxes in [center_x, center_y, w, h] format
        - iou_threshold: threshold at which we consider two boxes to be the same

    Returns a dict where the keys are an arbitrary id, and the values are the equivalence lists
    """
    if len(all_boxes) == 0:
        return {0: []}
    uf = UnionFind(len(all_boxes))

    xy_boxes = box_xywh_to_xyxy(np.asarray(all_boxes))
    iou = box_iou(xy_boxes, xy_boxes)
    for i, j in zip(*np.where(iou >= iou_threshold)):
        uf.unite(i, j)
    compo = defaultdict(list)
    for i in range(len(all_boxes)):
        compo[uf.find(i)].append(i)
    return compo


def convert(
    subset: str, flickr_path: Path, output_path: Path, merge_ground_truth: bool, next_img_id: int = 0, next_id: int = 0
):

    with open(flickr_path / f"{subset}.txt") as fd:
        ids = [int(l.strip()) for l in fd]

    multibox_entity_count = 0

    categories = [{"supercategory": "object", "id": 1, "name": "object"}]
    annotations = []
    images = []

    print(f"Exporting {subset}...")
    global_phrase_id = 0
    global_phrase_id2phrase = {}

    for img_id in tqdm(ids):

        with open(flickr_path / "Annotations" / f"{img_id}.xml") as xml_file:
            annotation = xmltodict.parse(xml_file.read())["annotation"]

        anno_file = os.path.join(flickr_path, "Annotations/%d.xml" % img_id)

        # Parse Annotation
        root = parse(anno_file).getroot()
        obj_elems = root.findall("./object")
        target_bboxes = {}

        for elem in obj_elems:
            if elem.find("bndbox") == None or len(elem.find("bndbox")) == 0:
                continue
            xmin = float(elem.findtext("./bndbox/xmin"))
            ymin = float(elem.findtext("./bndbox/ymin"))
            xmax = float(elem.findtext("./bndbox/xmax"))
            ymax = float(elem.findtext("./bndbox/ymax"))
            assert 0 < xmin and 0 < ymin

            h = ymax - ymin
            w = xmax - xmin

            coco_box = [xmin, ymin, w, h]

            for name in elem.findall("name"):
                entity_id = int(name.text)
                assert 0 < entity_id
                if not entity_id in target_bboxes:
                    target_bboxes[entity_id] = []
                else:
                    multibox_entity_count += 1
                # Dict from entity_id to list of all the bounding boxes
                target_bboxes[entity_id].append(coco_box)

        if merge_ground_truth:
            merged_bboxes = defaultdict(list)
            for eid, bbox_list in target_bboxes.items():
                boxes_xyxy = box_xywh_to_xyxy(torch.as_tensor(bbox_list, dtype=torch.float))
                gt_box_merged = [
                    min(boxes_xyxy[:, 0]).item(),
                    min(boxes_xyxy[:, 1]).item(),
                    max(boxes_xyxy[:, 2]).item(),
                    max(boxes_xyxy[:, 3]).item(),
                ]
                merged_bboxes[eid] = [xyxy2xywh(gt_box_merged)]  # convert back to xywh for coco format

            target_bboxes = merged_bboxes

        sents = get_sentence_data(flickr_path / "Sentences" / f"{img_id}.txt")
        for sent_id, sent in enumerate(sents):

            spans = {}  # global phrase ID to span in sentence
            phraseid2entityid = {}
            entityid2phraseid = defaultdict(list)
            sentence = sent["sentence"]
            entity_ids = [int(p["phrase_id"]) for p in sent["phrases"]]

            for global_phrase_id, phrase in enumerate(sent["phrases"]):
                phraseid2entityid[global_phrase_id] = int(phrase["phrase_id"])
                entityid2phraseid[int(phrase["phrase_id"])].append(global_phrase_id)
                first_word = phrase["first_word_index"]
                beg = sum([len(x) for x in sentence.split()[:first_word]]) + first_word
                spans[global_phrase_id] = (beg, beg + len(phrase["phrase"]))
                assert sentence[beg : beg + len(phrase["phrase"])] == phrase["phrase"]

            all_boxes_in_sent = []
            for ent_id in entity_ids:
                if ent_id in target_bboxes:
                    for bb in target_bboxes[ent_id]:
                        all_boxes_in_sent.append({"ent_id": int(ent_id), "coords": bb})

            equivalences = get_equivalent_boxes([b["coords"] for b in all_boxes_in_sent], 0.95)

            tokens_positive_eval = []
            for gpid, span in spans.items():
                if phraseid2entityid[gpid] in target_bboxes:
                    tokens_positive_eval.append([span])

            cur_img = {
                "file_name": annotation["filename"],
                "height": annotation["size"]["height"],
                "width": annotation["size"]["width"],
                "id": next_img_id,
                "caption": sentence,
                "dataset_name": "flickr",
                "tokens_negative": [(0, len(sentence))],
                "sentence_id": sent_id,
                "original_img_id": img_id,
                "tokens_positive_eval": tokens_positive_eval,
            }

            for equiv in equivalences.values():
                if len(equiv) == 0:
                    continue
                cur_entids = set([all_boxes_in_sent[bid]["ent_id"] for bid in equiv])
                token_spans = []
                for entid in cur_entids:
                    token_spans += [spans[gid] for gid in entityid2phraseid[entid]]
                xmin, ymin, w, h = all_boxes_in_sent[equiv[-1]]["coords"]

                cur_obj = {
                    "area": h * w,
                    "iscrowd": 0,
                    "image_id": next_img_id,
                    "category_id": 1,
                    "id": next_id,
                    "bbox": [xmin, ymin, w, h],
                    "tokens_positive": token_spans,
                }
                next_id += 1
                annotations.append(cur_obj)

            next_img_id += 1
            images.append(cur_img)
    ds = {"info": [], "licenses": [], "images": images, "annotations": annotations, "categories": categories}

    if merge_ground_truth:
        filename = f"final_flickr_mergedGT_{subset}.json"
    else:
        filename = f"final_flickr_separateGT_{subset}.json"

    with open(output_path / filename, "w") as j_file:
        json.dump(ds, j_file)
    return next_img_id, next_id


def main(args):
    flickr_path = Path(args.flickr_path)
    output_path = Path(args.out_path)
    os.makedirs(str(output_path), exist_ok=True)

    next_img_id, next_id = convert("train", flickr_path, output_path, args.merge_ground_truth)
    next_img_id, next_id = convert("val", flickr_path, output_path, args.merge_ground_truth, next_img_id, next_id)
    next_img_id, next_id = convert("test", flickr_path, output_path, args.merge_ground_truth, next_img_id, next_id)


if __name__ == "__main__":
    main(parse_args())
