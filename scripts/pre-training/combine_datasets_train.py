# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
Script used to combine data from different sources.
all_data_path : path containing the dicts from refexp, vg and gqa

"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torchvision.ops.boxes import box_iou
from tqdm import tqdm
import sys
PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils.dump import Annotation, Datapoint, convert2dataset_combined
from utils.spans import consolidate_spans, shift_spans


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")
    parser.add_argument(
        "--all_data_path", type=str, default=""
    )
    parser.add_argument(
        "--combine_datasets", "--list", nargs="+", help="List of datasets to combine", default=["refexp", "gqa", "vg"]
    )

    parser.add_argument(
        "--out_path",
        default="",
        type=str,
        help="Path where to export the resulting dataset.",
    )

    parser.add_argument(
        "--coco_path",
        required=True,
        type=str,
        help="Path to coco 2014 dataset.",
    )

    parser.add_argument(
        "--vg_img_data_path",
        required=True,
        type=str,
        help="Path to image meta data for VG"
    )
    return parser.parse_args()


def set_last_char(text):
    if text[-1] == "." or text[-1] == "?":
        return text
    else:
        return text + "."


def rescale_boxes(old_datapoint: Datapoint, old_size: Tuple[int, int], new_size: Tuple[int, int]):
    new_datapoint = Datapoint(
        image_id=old_datapoint.image_id,
        dataset_name=old_datapoint.dataset_name,
        tokens_negative=old_datapoint.tokens_negative,
        original_id=old_datapoint.original_id,
        caption=old_datapoint.caption,
        annotations=[],
    )
    old_h, old_w = old_size
    new_h, new_w = new_size
    for an in old_datapoint.annotations:
        assert len(an.bbox) == 4
        assert len(an.giou_friendly_bbox) == 4
        new_ann = Annotation(
            area=an.area,
            iscrowd=an.iscrowd,
            category_id=an.category_id,
            tokens_positive=an.tokens_positive,
            bbox=[
                an.bbox[0] / old_w * new_w,
                an.bbox[1] / old_h * new_h,
                an.bbox[2] / old_w * new_w,
                an.bbox[3] / old_h * new_h,
            ],
            giou_friendly_bbox=[
                an.giou_friendly_bbox[0] / old_w * new_w,
                an.giou_friendly_bbox[1] / old_h * new_h,
                an.giou_friendly_bbox[2] / old_w * new_w,
                an.giou_friendly_bbox[3] / old_h * new_h,
            ],
        )
        new_datapoint.annotations.append(new_ann)

    return new_datapoint


def combine_dataset_datapoints(
    dataset_dicts: Dict[str, List[Datapoint]], vg_imid2data: Dict[int, Dict], coco_imid2data: Dict[str, Dict], coco_path: str,
) -> Tuple[Dict[str, List[Datapoint]], Dict[str, List[Datapoint]]]:
    """This functions accepts a dict from the 'dataset_name' to the list of datapoints we have for this dataset.
    It splits the data points based on whether we have a coco id or a vg id for the images.
    Note that the images that are both in VG and COCO are considered to be coco images.

    vg_imid2data is a dict where the keys are VG image ID, the values are the image info (in particular, it contains the coco_id if it exists for this image)

    It returns two dictionaries:
    coco2datapoints:  the keys are the coco ids, the values are the list of datapoint for this image
    vg2datapoints:  the keys are the vg, the values are the list of datapoint for this image

    """

    coco_all_unsafe = set()
    vg_all_unsafe = set()

    # Extract all val/test ids
    with open(f"{coco_path}/annotations/instances_val2014.json", "r") as f:
        coco_val = json.load(f)
    coco_val_ids = []
    for item in coco_val["images"]:
        coco_val_ids.append(item["id"])

    with open("phrase_cut_unsafe_ids.pkl", "rb") as f:
        pc_unsafe = pickle.load(f)

    with open("refexp_all_unsafe_ids.pkl", "rb") as f:
        refexp_unsafe = pickle.load(f)

    with open("gqa_unsafe_ids.pkl", "rb") as f:
        gqa_unsafe = pickle.load(f)

    vg_image_unsafe = set.union(set(pc_unsafe), set(gqa_unsafe))

    # Divide into coco and vg images
    for id in vg_image_unsafe:
        if str(id)[0] == "n":
            continue
        if vg_imid2data[int(id)]["coco_id"] is not None:
            coco_all_unsafe.add(str(vg_imid2data[int(id)]["coco_id"]))
        else:
            vg_all_unsafe.add(str(id))

    for id in refexp_unsafe:
        coco_all_unsafe.add(str(id))

    for id in coco_val_ids:
        coco_all_unsafe.add(str(id))

    coco_all = defaultdict(list)
    vg_all = defaultdict(list)
    # Divide the data into coco and vg images
    for (dname, datapoint_list) in dataset_dicts.items():
        if dname == "gqa" or dname == "vg":
            for datapoint in datapoint_list:
                image_id = int(datapoint.image_id)
                if vg_imid2data[image_id]["coco_id"] is not None:
                    image_id = str(vg_imid2data[image_id]["coco_id"])
                    if image_id not in coco_all_unsafe:
                        coco_all[image_id].append(
                            rescale_boxes(
                                datapoint,
                                (
                                    vg_imid2data[int(datapoint.image_id)]["height"],
                                    vg_imid2data[int(datapoint.image_id)]["width"],
                                ),
                                (coco_imid2data[image_id]["height"], coco_imid2data[image_id]["width"]),
                            )
                        )
                else:
                    image_id = str(image_id)
                    if image_id not in vg_all_unsafe:
                        vg_all[image_id].append(datapoint)

        else:
            for datapoint in datapoint_list:
                image_id = str(datapoint.image_id)
                if image_id not in coco_all_unsafe:
                    coco_all[image_id].append(datapoint)

    return coco_all, vg_all


def get_refexp_groups(im2datapoint: Dict[str, List[Datapoint]]) -> List[Datapoint]:
    """This functions accepts a dictionary that contains all the datapoints from a given id.
    These datapoints are assumed to come from the same image subset (vg or coco)

    For each image, given the list of datapoints, we try to combine several datapoints together.
    The combination simply concatenates the captions for the combined datapoints, as well as the list of boxes.
    For a combination to be deemed acceptable, we require that the boxes are not overlapping too much.
    This ensures that only one part of the combined caption is referring to a particular object in the image.
    To achieve this combination, we use a greedy graph-coloring algorithm.

    This function returns a flat list of all the combined datapoints that were created.
    """
    combined_datapoints: List[Datapoint] = []

    for image_id, all_datapoints in tqdm(im2datapoint.items()):
        # get all the referring expressions for this image
        refexps = [datapoint.caption for datapoint in all_datapoints]

        # Create a graph where there is an edge between two datapoints iff they are NOT compatible
        adj_list = {i: [] for i in range(len(refexps))}

        # Get the list of all boxes (in "giou_friendly" format, aka [top_left_x, top_left_y, bottom_right_x, bottom_right_y]) for each datapoint
        all_boxes = []
        for datapoint in all_datapoints:
            if len(datapoint.annotations) > 0:
                all_boxes.append(
                    torch.stack([torch.as_tensor(ann.giou_friendly_bbox) for ann in datapoint.annotations])
                )
            else:
                all_boxes.append(torch.zeros(0, 4))

        # To find which referring expressions to combine into a single instance, we apply a graph coloring step
        # First we build the graph of refexps such that nodes correspond to refexps and and edge occurs between
        # two nodes when max giou between ANY boxes in the annotations > 0.5. This implies they are both referring
        # to the same box and hence should not be combined into one example.
        for i in range(len(all_datapoints)):
            for j in range(i + 1, len(all_datapoints)):
                giou = box_iou(all_boxes[i], all_boxes[j])
                if giou.numel() > 0 and torch.max(giou).item() > 0.5:
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        # Here we build the colored graph corresponding to the adjacency list given by adj_list
        colored_graph: Dict[int, int] = {}  # Color of each vertex
        nodes_degree = [(len(v), k) for k, v in adj_list.items()]
        nodes_sorted = sorted(nodes_degree, reverse=True)
        global_colors = [0]  # Colors used so far
        color_size = defaultdict(int)  # total length of the captions assigned to each color

        def get_color(admissible_color_set, new_length):
            admissible_color_list = sorted(list(admissible_color_set))
            for color in admissible_color_list:
                if color_size[color] + new_length + 2 <= 250:
                    return color
            return None

        # Loop over all nodes and color with the lowest color that is compatible
        # We add the constraint that the sum of the lengths of all the captions assigned to a given color is less than 250 (our max sequence length)
        for _, node in nodes_sorted:
            used_colors = set()
            # Gather the colors of the neighbours
            for adj_node in adj_list[node]:
                if adj_node in colored_graph:
                    used_colors.add(colored_graph[adj_node])
            if len(used_colors) < 1:
                # Neighbours are uncolored, we take the smallest color
                curr_color = get_color(global_colors, len(all_datapoints[node].caption))
            else:
                # Find the smallest unused color
                curr_color = get_color(set(global_colors) - set(used_colors), len(all_datapoints[node].caption))
            if curr_color is None:
                # Couldn't find a suitable color, creating one
                global_colors.append(max(global_colors) + 1)
                curr_color = global_colors[-1]
            colored_graph[node] = curr_color
            color_size[curr_color] += len(all_datapoints[node].caption)

        # Collect the datapoints that all have the same color
        color2datapoints: Dict[int, List[Datapoint]] = defaultdict(list)
        for node, color in colored_graph.items():
            color2datapoints[color].append(all_datapoints[node])

        # Make sure we have a valid coloring by checking that adjacent nodes have different colors
        for k, v in adj_list.items():
            for node in v:
                assert colored_graph[k] != colored_graph[node]

        for cur_datapoint_list in color2datapoints.values():
            if len(cur_datapoint_list) == 0:
                continue
            # collect the captions, and maybe add a trailing punctuation mark if there is not already
            all_captions = [set_last_char(datapoint.caption) for datapoint in cur_datapoint_list]
            combined_caption = " ".join(all_captions) + " "

            # compute the combined (offsetted) negative span
            cur_offset = 0
            combined_tokens_negative: List[Tuple[int, int]] = []
            for i, datapoint in enumerate(cur_datapoint_list):
                combined_tokens_negative += shift_spans(datapoint.tokens_negative, cur_offset)
                cur_offset += len(all_captions[i]) + 1  # 1 for space
            assert cur_offset == len(combined_caption)

            cur_combined_datapoint = Datapoint(
                image_id=image_id,
                dataset_name="mixed",
                tokens_negative=consolidate_spans(combined_tokens_negative, combined_caption),
                original_id=-1,
                caption=combined_caption.rstrip(),
                annotations=[],
            )

            # compute the offsetted positive span and append all annotations
            cur_offset = 0
            for data_id, datapoint in enumerate(cur_datapoint_list):
                for ann_id, ann in enumerate(datapoint.annotations):
                    new_annotation = Annotation(
                        area=ann.area,
                        iscrowd=ann.iscrowd,
                        category_id=ann.category_id,
                        bbox=ann.bbox,
                        giou_friendly_bbox=[],  # We don't need that anymore
                        tokens_positive=consolidate_spans(
                            shift_spans(ann.tokens_positive, cur_offset), combined_caption
                        ),
                    )
                    cur_combined_datapoint.annotations.append(new_annotation)
                cur_offset += len(all_captions[data_id]) + 1
            assert cur_offset == len(combined_caption)

            combined_datapoints.append(cur_combined_datapoint)

    return combined_datapoints


def main(args):

    output_path = Path(args.out_path)
    os.makedirs(str(output_path), exist_ok=True)

    dset_dicts: Dict[str, List[Datapoint]] = {}
    for i, dset in enumerate(args.combine_datasets):
        dset_path = Path(args.all_data_path + dset + "_dict.pkl")
        with open(dset_path, "rb") as f:
            print(f"Loading {args.combine_datasets[i]}")
            dset_dicts[args.combine_datasets[i]] = pickle.load(f)

    with open(f"{args.vg_img_data_path}/image_data.json", "r") as f:
        vg_image_data = json.load(f)
    vg_imid2data = {x["image_id"]: x for x in vg_image_data}

    with open(f"{args.coco_path}/annotations/instances_train2014.json", "r") as f:
        coco_image_data = json.load(f)
    imid2data_coco = {str(x["id"]): x for x in coco_image_data["images"]}

    combined_dict_coco, combined_dict_vg = combine_dataset_datapoints(dset_dicts, vg_imid2data, imid2data_coco, coco_path=args.coco_path)
    datapoint_list_coco = get_refexp_groups(combined_dict_coco)
    datapoint_list_vg = get_refexp_groups(combined_dict_vg)

    with open(output_path / "final_dict_coco.pkl", "wb") as f:
        pickle.dump(datapoint_list_coco, f)

    with open(output_path / "final_dict_vg.pkl", "wb") as f:
        pickle.dump(datapoint_list_vg, f)

    imid2data_vg = {str(x["image_id"]): x for x in vg_image_data}

    convert2dataset_combined(datapoint_list_coco, datapoint_list_vg, imid2data_coco, imid2data_vg, output_path)


if __name__ == "__main__":
    main(parse_args())
