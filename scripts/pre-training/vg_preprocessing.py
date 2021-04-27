# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import argparse
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple
import sys
PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import torch
from tqdm import tqdm
from utils.boxes import box_iou_helper, combine_boxes, get_boxes_equiv, obj_to_box, region_to_box, xyxy_to_xywh
from utils.dump import Annotation, Datapoint
from utils.spans import (
    PreprocessError,
    consolidate_spans,
    get_canonical_spans,
    span_intersect_spanlist,
    spanlist_intersect_spanlist,
)
from utils.text import get_root_and_nouns, normalize_sentence, normalize_whitespace, simplify_punctuation
from utils.unionfind import UnionFind


def parse_args():
    parser = argparse.ArgumentParser("Visual Genome conversion script")

    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        help="Path to the VG dataset. Should contain region graphs",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        type=str,
        help="Path where to export the resulting dataset.",
    )
    parser.add_argument(
        "--nb_process",
        default=1,
        type=str,
        help="Number of concurrent processes to use to dump the data",
    )

    return parser.parse_args()


def preprocess_region(region):
    filtered_region = {
        "caption": simplify_punctuation(normalize_whitespace(region["phrase"])),
        "original_image_id": region["image_id"],
        "original_region_id": region["region_id"],
        "boxes": [],
        "tokens_positive": [],
        "found_objects": False,
    }
    if len(filtered_region["caption"]) < 3:
        raise PreprocessError("caption too short, skipping" + filtered_region["caption"])
    _, _, root_spans, negative_spans = get_root_and_nouns(filtered_region["caption"].lower(), False)

    # Filter objects that have multiple synsets, they are likely to be spurious
    obj_synsets = set([o["synsets"][0] for o in region["objects"] if len(o["synsets"]) == 1])
    synsets_count = Counter([s["synset_name"] for s in region["synsets"]])
    # Filter synsets that occur multiple times, since we don't have mapping to objects
    all_synsets = set([s["synset_name"] for s in region["synsets"] if synsets_count[s["synset_name"]] == 1])
    authorized_synsets = obj_synsets.intersection(all_synsets)
    syn2span: Dict[str, Tuple[int, int]] = {
        s["synset_name"]: (s["entity_idx_start"], s["entity_idx_end"])
        for s in region["synsets"]
        if s["synset_name"] in authorized_synsets
    }

    synlist, spanlist = [], []
    for k, s in syn2span.items():
        synlist.append(k)
        spanlist.append([s])

    # the spans positions may have been altered by the whitespace removal, so we recompute here
    spanlist, new_caption = get_canonical_spans(spanlist, region["phrase"], whitespace_only=True)
    if new_caption.lower().strip() != filtered_region["caption"].lower().strip():
        raise PreprocessError(f"Inconsistent whitespace removal: '{new_caption}' vs '{filtered_region['caption']}'")

    assert len(synlist) == len(spanlist)
    syn2span = {k: v[0] for k, v in zip(synlist, spanlist)}

    root_objs = []
    other_objs: Dict[Tuple[int, int], List[List[int]]] = {}
    for obj in region["objects"]:
        if len(obj["synsets"]) == 1 and obj["synsets"][0] in authorized_synsets:
            cur_span = syn2span[obj["synsets"][0]]
            if span_intersect_spanlist(cur_span, root_spans):
                root_objs.append(obj_to_box(obj))
                filtered_region["found_objects"] = True
            else:
                if cur_span not in other_objs:
                    other_objs[cur_span] = []
                    negative_spans.append(cur_span)
                other_objs[cur_span].append(obj_to_box(obj))
                filtered_region["found_objects"] = True

    if len(root_objs) == 0:
        # If we don't have a box for the root of the sentence, we use the box of the region itself.
        root_objs.append(region_to_box(region))

    dedup_root_objs = combine_boxes(root_objs)
    filtered_region["boxes"] += dedup_root_objs
    root_spans = consolidate_spans(root_spans, filtered_region["caption"])
    filtered_region["tokens_positive"] += [root_spans for _ in range(len(dedup_root_objs))]

    for span, objs in other_objs.items():
        dedup_objs = combine_boxes(objs)
        filtered_region["boxes"] += dedup_objs
        cur_spans = consolidate_spans([span], filtered_region["caption"])
        filtered_region["tokens_positive"] += [cur_spans for _ in range(len(dedup_objs))]

    filtered_region["tokens_negative"] = consolidate_spans(negative_spans, filtered_region["caption"])
    return filtered_region


def deduplicate_regions(regions, iou_threshold=0.5):
    """This functions accepts pre-processed region descriptions for a given image, and removes regions that are redundant.
    Two regions are deemed redundant if 1) the text is closely matching 2) the IOU between region boxes is > iou_threshold
    A cleaned description is returned.
    """

    def helper_merge(regions):
        if len(regions) <= 1:
            return regions
        uf = UnionFind(len(regions))
        for r in regions:
            spans, txt2 = get_canonical_spans(r["tokens_positive"], r["caption"])
            if txt != txt2:
                raise PreprocessError(f"inconsistent canonicalization fct. Mismatch: '{txt}' and '{txt2}'")
            r["cano_tokens"] = spans

        for r1 in range(len(regions)):
            for r2 in range(r1 + 1, len(regions)):
                compatible = True
                assert len(regions[r1]["boxes"]) == len(regions[r1]["cano_tokens"])
                assert len(regions[r2]["boxes"]) == len(regions[r2]["cano_tokens"])
                ious = box_iou_helper(regions[r1]["boxes"], regions[r2]["boxes"])
                for b1 in range(len(regions[r1]["cano_tokens"])):
                    for b2 in range(len(regions[r2]["cano_tokens"])):
                        if (len(regions[r1]["cano_tokens"][b1]) == 0 or len(regions[r2]["cano_tokens"][b2]) == 0) or (
                            spanlist_intersect_spanlist(regions[r1]["cano_tokens"][b1], regions[r2]["cano_tokens"][b2])
                            and ious[b1][b2] < iou_threshold
                        ):
                            compatible = False
                            break
                    if not compatible:
                        break
                if compatible:
                    uf.unite(r1, r2)
        compo2regions = defaultdict(list)
        for i, r in enumerate(regions):
            compo2regions[uf.find(i)].append(r)

        final_regions = []
        for reg_list in compo2regions.values():
            if len(reg_list) == 1:
                final_regions.append(reg_list[0])
            else:
                # We pick as representative of this cluster the region with the most boxes
                sorted_regions = sorted([(len(r["boxes"]), i) for i, r in enumerate(reg_list)], reverse=True)
                reg_ids = [sr[1] for sr in sorted_regions]
                # We need to put the boxes and token spans in buckets
                cano_spans_buckets = []
                orig_spans_buckets = []
                boxes_buckets = []
                for idx in reg_ids:
                    for b in range(len(reg_list[idx]["boxes"])):
                        # find the bucket
                        bucket = -1
                        for j in range(len(cano_spans_buckets)):
                            if spanlist_intersect_spanlist(reg_list[idx]["cano_tokens"][b], cano_spans_buckets[j]):
                                bucket = j
                                break
                        if bucket == -1:
                            # bucket not found, creating one.
                            if idx != reg_ids[0]:
                                # This shouldn't happen. But if it does, we give up on the merging
                                return regions
                                assert idx == reg_ids[0], (
                                    "TODO: if this triggers, it means another regions has token spans than aren't covered by the main region."
                                    + "We need to create a new token span, which involve finding the span in the original sentencen of the main region. Don't forget to update the negative tokens"
                                )

                            bucket = len(orig_spans_buckets)
                            orig_spans_buckets.append(reg_list[idx]["tokens_positive"][b])
                            cano_spans_buckets.append(reg_list[idx]["cano_tokens"][b])
                            boxes_buckets.append([reg_list[idx]["boxes"][b]])
                        else:
                            boxes_buckets[bucket].append(reg_list[idx]["boxes"][b])
                assert len(orig_spans_buckets) == len(boxes_buckets)
                merged_region = deepcopy(reg_list[reg_ids[0]])
                merged_region["tokens_positive"] = []
                merged_region["boxes"] = []
                for i in range(len(boxes_buckets)):
                    dedup_objs = combine_boxes(boxes_buckets[i], iou_threshold=0.5)
                    merged_region["boxes"] += dedup_objs
                    merged_region["tokens_positive"] += [orig_spans_buckets[i] for _ in range(len(dedup_objs))]
                final_regions.append(merged_region)
        for r in final_regions:
            del r["cano_tokens"]
        return final_regions

    txt2region = defaultdict(list)
    for r in regions:
        txt2region[normalize_sentence(r["caption"])].append(r)

    stupid_sentence_set = set(["wall", "side", "building"])
    final_regions = []
    for txt, regions in txt2region.items():
        # Edge case, we remove the sentences like "the wall on the side of the building" which are uninformative and have spurious boxes
        if "wall" in txt and set(txt.strip().split(" ")).issubset(stupid_sentence_set):
            continue
        if len(regions) == 1:
            final_regions.append(deepcopy(regions[0]))
        else:
            # print(txt)

            regions_with_boxes = [r for r in regions if r["found_objects"]]
            all_boxes = sum([r["boxes"] for r in regions_with_boxes], [])
            # print("regions with boxes", len(regions_with_boxes))

            regions_without_boxes = []
            for r in regions:
                if not r["found_objects"]:
                    # we consider than one of the region with boxes will be better suited and drop this one
                    # if there is a positive iou. Otherwise, we have to keep it
                    if len(regions_with_boxes) == 0 or box_iou_helper(all_boxes, r["boxes"]).max().item() < 0.1:
                        regions_without_boxes.append(r)

            # print("regions without boxes", len(regions_without_boxes))

            try:
                new_regions_with_boxes = helper_merge(regions_with_boxes)
            except PreprocessError as e:
                print("skipping", e)
                # Ouch, hit a cornercase, we give up on the merge
                new_regions_with_boxes = regions_with_boxes
            try:
                new_regions_without_boxes = helper_merge(regions_without_boxes)
            except PreprocessError as e:
                print("skipping", e)
                # Ouch, hit a cornercase, we give up on the merge
                new_regions_without_boxes = regions_without_boxes

            # now collapse into one big region. We do it only when the captions are exactly matching, otherwise it's a nightmare to recompute spans
            capt2region = defaultdict(list)
            for r in new_regions_with_boxes + new_regions_without_boxes:
                capt2region[r["caption"]].append(r)
            for capt, reg_list in capt2region.items():
                all_boxes = sum([r["boxes"] for r in reg_list], [])
                all_tokens = sum([r["tokens_positive"] for r in reg_list], [])
                compo2boxes, compo2id = get_boxes_equiv(all_boxes, iou_threshold=0.75)
                final_boxes = []
                final_tokens = []
                if compo2boxes is not None:
                    for compo in compo2boxes.keys():
                        box_list = compo2boxes[compo]
                        id_list = compo2id[compo]
                        final_boxes.append(xyxy_to_xywh(torch.stack(box_list, 0).mean(0)).tolist())
                        final_tokens.append(consolidate_spans(sum([all_tokens[i] for i in id_list], []), capt))
                else:
                    final_boxes = all_boxes
                    final_tokens = all_tokens

                merged_region = {
                    "caption": capt,
                    "original_image_id": reg_list[0]["original_image_id"],
                    "original_region_id": reg_list[0]["original_region_id"],
                    "boxes": final_boxes,
                    "tokens_positive": final_tokens,
                    "tokens_negative": consolidate_spans(sum([r["tokens_negative"] for r in reg_list], []), capt),
                    "found_objects": False,
                }
                final_regions.append(merged_region)

    return final_regions


def _get_all_datapoints(output_path: Path, img_list, proc_id: int):
    # image2ann_map = defaultdict(lambda: defaultdict(list))
    print(f"process {proc_id} got job queue of", len(img_list))
    all_datapoints: List[Datapoint] = []
    for i, data in enumerate(tqdm(img_list)):
        # print(f"status {i}/{len(img_list)}")
        all_regions = []
        for r in data["regions"]:
            try:
                all_regions.append(preprocess_region(r))
            except PreprocessError as e:
                print("Dropping region, preprocess failed", e)
        all_regions = deduplicate_regions(all_regions)

        # all_regions = deduplicate_regions([preprocess_region(r) for r in data["regions"]])

        for region in all_regions:
            cur_datapoint = Datapoint(
                image_id=data["image_id"],
                dataset_name="VG",
                tokens_negative=region["tokens_negative"],
                original_id=region["original_region_id"],
                caption=region["caption"],
                annotations=[],
            )
            assert len(region["boxes"]) == len(region["tokens_positive"])
            converted_bbox = torch.as_tensor(region["boxes"], dtype=torch.float)
            areas = converted_bbox[:, -1] * converted_bbox[:, -2]
            # Convert to (x,y,x,y) format
            converted_bbox[:, 2:] += converted_bbox[:, :2]
            for i in range(len(region["boxes"])):
                cur_ann = Annotation(
                    area=float(areas[i]),
                    iscrowd=0,
                    category_id=1,
                    bbox=region["boxes"][i],
                    giou_friendly_bbox=converted_bbox[i].tolist(),
                    tokens_positive=region["tokens_positive"][i],
                )
                cur_datapoint.annotations.append(cur_ann)
            all_datapoints.append(cur_datapoint)

    print(f"Process {proc_id} dumping...")
    pickle.dump(all_datapoints, open(output_path / f"vg_train_dump_{proc_id}.pkl", "wb"))
    print(f"Process {proc_id} done.")
    del all_datapoints
    return None
    # return image2ann_map


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_all_datapoints(dataset_path: Path, output_path: Path, nb_proc: int):

    print("loading region graphs....")
    with open(dataset_path / "region_graphs.json", "r") as f:
        VG_region_graph = json.load(f)

    print("loading success!")

    # return _get_image2ann_mapping(VG_region_graph)
    chunks = list(chunk_list(VG_region_graph, math.ceil(len(VG_region_graph) / (18 * nb_proc))))
    # sub_part = sum(chunks[:3], [])
    # chunks = list(chunk_list(sub_part, math.ceil(len(sub_part) / nb_proc)))
    proc_id = list(range(len(chunks)))
    # assert len(chunks) == nb_proc
    with Pool(nb_proc) as p:
        p.starmap(partial(_get_all_datapoints, output_path), zip(chunks, proc_id))

    return None


def main(args):
    vg_path = Path(args.dataset_path)

    output_path = Path(args.out_path) if args.out_path is not None else vg_path

    os.makedirs(str(output_path), exist_ok=True)

    get_all_datapoints(vg_path, output_path, int(args.nb_process))


if __name__ == "__main__":
    main(parse_args())
