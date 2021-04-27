# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
This script it a utility to convert the CLEVR dataset to a coco format suitable for this repository.
Specifically, we want the bounding boxes that are mentionned in the CLEVR question.
To retrieve the set of objects of interest, we simply execute the provided functional program,
recording the relevant objects along the way.
To retrieve bounding boxes, we rely on bounding boxes provided by https://github.com/StanfordVL/ReferringRelationships
Note that each image-question pair will yield a single data-point.
"""
import argparse
import itertools
import json
import os
import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# import cv2
# import numpy as np
# from pycocotools import mask as coco_mask
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--clevr_path",
        default="",
        type=str,
        help="Path to the CLEVR dataset. Should contain questions and scenes",
    )
    parser.add_argument(
        "--refclevr_path",
        default="",
        type=str,
        help="Path to the REFCLEVR dataset. This is required to get boxes if --use_refclevr_imgs is passed",
    )
    parser.add_argument(
        "--clevr_box_path",
        default="",
        type=str,
        help="Path to the boxes for the CLEVR dataset.",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        type=str,
        help="Path where to export the resulting dataset. Leave it to None to use the same path as above",
    )

    parser.add_argument(
        "--use_refclevr_imgs",
        action="store_true",
        help="If true, we assume that refclevr images will be used for training, and provide box/segmentation mask annotation accordingly",
    )

    parser.add_argument(
        "--no_caption", action="store_true", help="If true, output all the boxes and ignore questions all together"
    )

    parser.add_argument(
        "--medium",
        action="store_true",
        help="If true, filter some degenerate questions, to make sure a box correspond to exactly one token group",
    )

    parser.add_argument(
        "--cogent",
        action="store_true",
        help="If true, assume the dataset at hand is CoGenT",
    )
    return parser.parse_args()


# Synonyms extracted from https://github.com/facebookresearch/clevr-dataset-gen/blob/master/question_generation/synonyms.json
SYNONYMS = {
    "thing": ["thing", "object"],
    "sphere": ["sphere", "ball"],
    "cube": ["cube", "block"],
    "large": ["large", "big"],
    "small": ["small", "tiny"],
    "metal": ["metallic", "metal", "shiny"],
    "rubber": ["rubber", "matte"],
    "left": ["left of", "to the left of", "on the left side of"],
    "right": ["right of", "to the right of", "on the right side of"],
    "behind": ["behind"],
    "front": ["in front of"],
    "above": ["above"],
    "below": ["below"],
}

# Sets of word representing the various attributes
SIZE_SET = [s for current in ["small", "large"] for s in (SYNONYMS[current] if current in SYNONYMS else [current])]
COLOR_SET = [
    s
    for current in ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    for s in (SYNONYMS[current] if current in SYNONYMS else [current])
]
SHAPE_SET = [
    s
    for current in ["cube", "sphere", "cylinder", "thing"]
    for s in (SYNONYMS[current] if current in SYNONYMS else [current])
]
MATERIAL_SET = [s for current in ["rubber", "metal"] for s in (SYNONYMS[current] if current in SYNONYMS else [current])]

# Compiled regexes for various parts of the templates
OPTIONAL_REGEX = re.compile(r"\[([^\[]*)\]")
ALL_SIZES = "|".join(SIZE_SET)
ALL_COLORS = "|".join(COLOR_SET)
ALL_RELATIONS = "|".join(
    s
    for current in ["left", "right", "behind", "front"]
    for s in (SYNONYMS[current] if current in SYNONYMS else [current])
)
ALL_SHAPES = "|".join(SHAPE_SET)
ALL_MATERIALS = "|".join(MATERIAL_SET)
SIZE_REGEX = re.compile(r"<Z\d?>")
COLOR_REGEX = re.compile(r"<C\d?>")
RELATION_REGEX = re.compile(r"<R\d?>")
SHAPE_REGEX = re.compile(r"<S\d?>")
MATERIAL_REGEX = re.compile(r"<M\d?>")
PLURAL_REGEX = re.compile(r"\)\)\?s(\\s|\\\?)")
OTHER_QUES = re.compile(r"(?:Is|Are) there ((?:anything else|any other thing)s?) that")
OTHER_COUNT_QUES = re.compile(r"(?:How many|What number of) (other (?:thing|object)s?)")


def get_synonyms(word, include_plural=False):
    """Get the possible CLEVR synonyms for a given word"""
    if include_plural:
        syns = get_synonyms(word, False)
        return syns + [s + "s" for s in syns]
    return SYNONYMS[word] if word in SYNONYMS else [word]


def build_regex(text):
    """Transform a CLEVR text template into a regex that matches it"""
    # Match the final question mark
    text = re.sub(r"\?", "\?", text)
    # Because of optinal expensions, we need to be lenient on space matching. This will allow to skip some spaces, double spaces,...
    text = re.sub(r"\s", "\\\s*", text)
    # Hack, because the templates in the dataset somehow don't match the templates exactly
    text = re.sub("another", "(?:another|a)", text)
    text = re.sub("other", "(?:other)?", text)
    # Replace all attributes by their possibilities, possibly in a group
    text = SIZE_REGEX.sub(f"(?:{ALL_SIZES})?", text)
    text = COLOR_REGEX.sub(f"(?:{ALL_COLORS})?", text)
    text = MATERIAL_REGEX.sub(f"(?:{ALL_MATERIALS})?", text)
    text = SHAPE_REGEX.sub(f"(?:{ALL_SHAPES})?", text)
    text = RELATION_REGEX.sub(f"(?:{ALL_RELATIONS})?", text)
    text = OPTIONAL_REGEX.sub(r"(?:\1)?", text)
    return re.compile(text)


def load_templates():
    """Loads the CLEVR templates from disk"""
    num_loaded_templates = 0
    templates = {}
    for fn in os.listdir("CLEVR_1.0_templates"):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join("CLEVR_1.0_templates", fn), "r") as f:
            base = os.path.splitext(fn)[0]
            for i, template in enumerate(json.load(f)):
                num_loaded_templates += 1
                key = (fn, i)
                template["regexes"] = [build_regex(t) for t in template["text"]]
                templates[key] = template
    print("Read %d templates from disk" % num_loaded_templates)
    return templates


class Item:
    def __init__(self, color, size, shape, material, pos=None, id=None):
        self.id = id
        self.color = color
        self.size = size
        self.shape = shape
        self.material = material
        self.pos = pos

        self.union_branches = []  # For tracking union filters

    def is_size(self, size):
        return self.size == size

    def is_color(self, color):
        return self.color == color

    def is_material(self, material):
        return self.material == material

    def is_shape(self, shape):
        return self.shape == shape

    def is_same(self, other):
        return (
            self.color == other.color
            and self.shape == other.shape
            and self.size == other.size
            and self.material == other.material
        )

    def get_cat(self):
        return f"{self.color}_{self.material}_{self.shape}"

    def __str__(self):
        return f"object{self.id}: {self.size} {self.color} {self.material} {self.shape}"

    def __repr__(self):
        return f'Item("{self.color}", "{self.size}", "{self.shape}", "{self.material}")'


class FilterUnion:
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def append(self, e):
        self.f1.append(e)
        self.f2.append(e)


class ItemCollection:
    def __init__(self, scene):
        self.objects = [
            Item(o["color"], o["size"], o["shape"], o["material"], o["pixel_coords"], i)
            for i, o in enumerate(scene["objects"])
        ]
        self.scene = scene
        self.filters = []

    def __len__(self):
        return len(self.objects)

    def filter_material(self, material, id=-1):
        if id == -1:
            self.filters.append(material)
        self.objects = [o for o in self.objects if o.is_material(material) and o.id != id]
        return self

    def filter_shape(self, shape, id=-1):
        if id == -1:
            self.filters.append(shape)
        self.objects = [o for o in self.objects if o.is_shape(shape) and o.id != id]
        return self

    def filter_color(self, color, id=-1):
        if id == -1:
            self.filters.append(color)
        self.objects = [o for o in self.objects if o.is_color(color) and o.id != id]
        return self

    def filter_size(self, size, id=-1):
        if id == -1:
            self.filters.append(size)
        self.objects = [o for o in self.objects if o.is_size(size) and o.id != id]
        return self

    def filter_ids(self, ids):
        self.objects = [o for o in self.objects if o.id in ids]
        return self

    def filter_direction(self, obj, dir):
        kept = self.scene["relationships"][dir][obj.id]
        self.objects = [o for o in self.objects if o.id in kept]
        return self

    def get_unique(self):
        assert len(self.objects) == 1, "Error, the set is not a unique element"
        return deepcopy(self.objects[0])

    def __str__(self):
        string = f"Set of {len(self.objects)} objects:"
        for o in self.objects:
            string += "\n\t " + o.__str__()
        return string


def find_matching_template(question, templates):
    """Return the template that corresponds to the given question"""
    for t in templates.values():
        for r in t["regexes"]:
            if r.fullmatch(question["question"]) is not None:
                return t
    assert False, "No template found"


def find_node_id_in_template(program, idx, template):
    """Given a node idx in the given program, try to find the corresponding node in the template"""
    ind_prog = ind_template = 0

    def is_same_fun(fn1, fn2):
        """The function names in programs and templates don't always match 1:1, so we redefine equality"""
        if fn2 == "filter" and fn1[: len("filter_")] == "filter_":
            return True
        if fn1 == "relate" and fn2 == "relate_filter":
            return True
        prefixes = ["", "filter_", "relate_filter_"]
        for p in prefixes:
            if p + fn1 == fn2 or fn1 == p + fn2:
                return True
        return False

    # First heuristic: if both functions are unique in the template and the program, return as is
    count_in_prog = [1 if is_same_fun(p["function"], program[idx]["function"]) else 0 for p in program]
    count_in_template = [1 if is_same_fun(p["type"], program[idx]["function"]) else 0 for p in template["nodes"]]
    if sum(count_in_prog) == sum(count_in_template) == 1:
        return count_in_template.index(1)

    # Second heuristic: We go through the template and the nodes, iterating through matching functions
    while ind_prog <= idx and ind_template < len(template["nodes"]):
        if ind_prog == idx:
            assert is_same_fun(
                program[ind_prog]["function"], template["nodes"][ind_template]["type"]
            ), "Wrong function type"
            return ind_template
        if is_same_fun(program[ind_prog]["function"], template["nodes"][ind_template]["type"]) and not is_same_fun(
            program[ind_prog + 1]["function"], template["nodes"][ind_template]["type"]
        ):
            ind_template += 1
        ind_prog += 1

    assert False, "shouldn't get here"
    return None


def find_tokens(question, template: Dict, node_id, backtrack=True) -> List[List[Tuple[int, int]]]:
    def backtrack_previous_nodes(cur_id, is_root=True):
        """Find previous filtering nodes that might have text associated.
        The full filter may contain UNION nodes, which means that there might be several set of candidate filters, one for each branch
        This function returns a list of list, representing the list of token for each possible branch.
        """
        cur_tokens = []
        function = template["nodes"][cur_id]["type"]
        if function[: len("same_")] == "same_":
            # This node finds objects which have a property in common with a target one
            # The text is always constructed as "same {attribute}" where attribute can be size, color, material or shape.
            # To retrieve the corresponding tokens, we simply look for that pattern in the question
            pos = question["question"].find(function.replace("_", " "))
            assert pos != -1
            cur_tokens.append((pos, pos + len(function)))
        if (
            (function in ["filter", "relate", "relate_filter"] or function[: len("filter_")] == "filter_")
            and cur_id != node_id
            and function != "filter_unique"
        ):
            # "relate" filters object based on their physical position relative to another one (eg. "to the right of")
            # "filter" filters by attributes
            # Both types are actually pattern-based filters, so we use the find_tokens function to retrieve the token
            cur_tokens = find_tokens(question, template, cur_id, False)
            assert len(cur_tokens) == 1, "Error, the relate filter is expected to yield only one possibility"
            cur_tokens = cur_tokens[0]
        if (
            is_root
            or function[: len("same_")] == "same_"
            or function == "intersect"
            or function[: len("filter_")] == "filter_"
        ) and (is_root or function != "filter_unique"):
            # If a node verifies this condition, we are interested in its ancestors.
            # We get a list of list for each ancestors, and we need to do the "cartesian concatenation" of all of them
            # Eg, if we get [[a],[b]] from ancestor 1 and [[c]] from ancestor 2, then the result is [[a,c],[b,c]]
            # Then we append cur_tokens and return
            ancestors = [backtrack_previous_nodes(a, False) for a in template["nodes"][cur_id]["inputs"]]
            return [
                cur_tokens + list(itertools.chain.from_iterable(element)) for element in itertools.product(*ancestors)
            ]
        if function == "union":
            # This the union node. We get filters from both branches and return both
            return [
                b for branch in template["nodes"][cur_id]["inputs"] for b in backtrack_previous_nodes(branch, False)
            ]
        return [cur_tokens]

    if "side_inputs" not in template["nodes"][node_id]:
        assert template["nodes"][node_id]["type"] in ["exist", "count"]
        # In the case of an exist node, we need to backtrack through the previous nodes to understand what led us here.
        tokens = backtrack_previous_nodes(node_id)

        for reg in [OTHER_QUES, OTHER_COUNT_QUES]:
            other_match = reg.match(question["question"])
            if other_match is not None:
                tokens = [t + [other_match.span(1)] for t in tokens]
        return tokens
    targets = template["nodes"][node_id]["side_inputs"]

    if (
        template["nodes"][node_id]["type"] not in ["relate_filter_count", "relate_filter_exist", "relate_filter_unique"]
        and backtrack
    ):
        tokens = backtrack_previous_nodes(node_id)
    else:
        tokens = [[]]

    def add_group(choices, match):
        """Given a match of an attribute token (eg a color <C3>), this function replaces it
        with a regex matching all the possible values of that attribute. If the given attribute
        token is part of the input of the current node_id, this regex is wrapped in a group."""
        return f"((?:{choices}))?" if match.group(0) in targets else f"(?:{choices})?"

    def build_custom_regex(text):
        """This build a regex that will match questions from the given text template.
        Additionally, tokens that are input of the node_id will be put in a regex group for easy retrieval"""

        # Match the final question mark
        text = re.sub(r"\?", "\?", text)
        # Because of optinal expensions, we need to be lenient on space matching. This will allow to skip some spaces
        text = re.sub(r"\s", "\\\s*", text)
        # Hack, because the templates in the dataset somehow don't match the templates exactly
        text = re.sub("another", "(?:another|a)", text)
        text = re.sub("other", "(?:other)?", text)
        # Replace all attributes by their possibilities, possibly in a group
        text = SIZE_REGEX.sub(partial(add_group, ALL_SIZES), text)
        text = COLOR_REGEX.sub(partial(add_group, ALL_COLORS), text)
        text = MATERIAL_REGEX.sub(partial(add_group, ALL_MATERIALS), text)
        text = SHAPE_REGEX.sub(partial(add_group, ALL_SHAPES), text)
        text = RELATION_REGEX.sub(partial(add_group, ALL_RELATIONS), text)
        # Optional text
        text = OPTIONAL_REGEX.sub(r"(?:\1)?", text)
        # To match plurals in our groups, we detect -s suffixes
        text = PLURAL_REGEX.sub(r")s)?\1", text)
        return re.compile(text)

    regexes = [build_custom_regex(t) for t in template["text"]]
    for r in regexes:
        match = r.fullmatch(question["question"])
        if match is not None:
            for i in range(len(match.groups())):
                if match.group(i + 1) is not None:
                    tokens = [t + [match.span(i + 1)] for t in tokens]
            return tokens
    assert False, "not found"
    return None


class SkipException(Exception):
    pass


def parse_prog(scene, question, templates, medium=False, verbose=False):
    """Parse the program of the given question and return the objects, along with the list of tokens they correspond to from the scene necessary to compute the answer"""

    id_to_output = {}  # contains intermediate results, with the step number as key
    out_obj = []
    answer = None
    template = find_matching_template(question, templates)
    if verbose:
        print(question)
        print(template)
    for i, fn in enumerate(question["program"]):
        if fn["function"] == "scene":
            id_to_output[i] = ItemCollection(scene)
        elif fn["function"] == "filter_size":
            assert len(fn["value_inputs"]) == 1
            id_to_output[i] = deepcopy(id_to_output[fn["inputs"][0]]).filter_size(fn["value_inputs"][0])
        elif fn["function"] == "filter_material":
            assert len(fn["value_inputs"]) == 1
            id_to_output[i] = deepcopy(id_to_output[fn["inputs"][0]]).filter_material(fn["value_inputs"][0])
        elif fn["function"] == "filter_color":
            assert len(fn["value_inputs"]) == 1
            id_to_output[i] = deepcopy(id_to_output[fn["inputs"][0]]).filter_color(fn["value_inputs"][0])
        elif fn["function"] == "filter_shape":
            assert len(fn["value_inputs"]) == 1
            id_to_output[i] = deepcopy(id_to_output[fn["inputs"][0]]).filter_shape(fn["value_inputs"][0])

        elif fn["function"] == "unique":
            single_obj = id_to_output[fn["inputs"][0]]

            node_id = find_node_id_in_template(question["program"], i, template)
            tokens = find_tokens(question, template, node_id)
            assert len(tokens) == 1, f"{tokens}"
            tokens = tokens[0]
            new_obj = single_obj.get_unique()
            if medium:
                for o in out_obj:
                    if o[0].id == new_obj.id:
                        raise SkipException
            out_obj.append((new_obj, tokens))
            id_to_output[i] = out_obj[-1][0]

        elif fn["function"] == "same_shape":
            id_to_output[i] = ItemCollection(scene).filter_shape(
                id_to_output[fn["inputs"][0]].shape, id_to_output[fn["inputs"][0]].id
            )
        elif fn["function"] == "same_color":
            id_to_output[i] = ItemCollection(scene).filter_color(
                id_to_output[fn["inputs"][0]].color, id_to_output[fn["inputs"][0]].id
            )
        elif fn["function"] == "same_size":
            id_to_output[i] = ItemCollection(scene).filter_size(
                id_to_output[fn["inputs"][0]].size, id_to_output[fn["inputs"][0]].id
            )
        elif fn["function"] == "same_material":
            id_to_output[i] = ItemCollection(scene).filter_material(
                id_to_output[fn["inputs"][0]].material, id_to_output[fn["inputs"][0]].id
            )

        elif fn["function"] == "query_shape":
            answer = id_to_output[fn["inputs"][0]].shape
            id_to_output[i] = answer
        elif fn["function"] == "query_color":
            answer = id_to_output[fn["inputs"][0]].color
            id_to_output[i] = answer
        elif fn["function"] == "query_size":
            answer = id_to_output[fn["inputs"][0]].size
            id_to_output[i] = answer
        elif fn["function"] == "query_material":
            answer = id_to_output[fn["inputs"][0]].material
            id_to_output[i] = answer

        elif fn["function"] == "relate":
            assert len(fn["value_inputs"]) == 1
            id_to_output[i] = ItemCollection(scene).filter_direction(
                id_to_output[fn["inputs"][0]], fn["value_inputs"][0]
            )

        elif fn["function"] == "count":
            assert len(fn["value_inputs"]) == 0
            cur_set = id_to_output[fn["inputs"][0]]
            answer = len(cur_set)
            tokens = []
            if len(cur_set.objects) > 0:
                # find associated tokens only if the set is non-empty
                node_id = find_node_id_in_template(question["program"], i, template)
                tokens = find_tokens(question, template, node_id)

            for o in cur_set.objects:
                if medium:
                    for oo in out_obj:
                        if oo[0].id == o.id:
                            raise SkipException
                if isinstance(cur_set.filters, FilterUnion):
                    assert len(tokens) == 2
                    cur_tok = []
                    for b in o.union_branches:
                        cur_tok += tokens[b]
                    out_obj.append((deepcopy(o), list(set(cur_tok))))
                else:
                    assert len(tokens) == 1
                    out_obj.append((deepcopy(o), tokens[0]))
            id_to_output[i] = answer

        elif fn["function"][: len("equal_")] == "equal_":
            assert len(fn["inputs"]) == 2
            answer = "yes" if id_to_output[fn["inputs"][0]] == id_to_output[fn["inputs"][1]] else "no"
            id_to_output[i] = answer

        elif fn["function"] == "greater_than":
            assert len(fn["inputs"]) == 2
            answer = "yes" if id_to_output[fn["inputs"][0]] > id_to_output[fn["inputs"][1]] else "no"
            id_to_output[i] = answer
        elif fn["function"] == "less_than":
            assert len(fn["inputs"]) == 2
            answer = "yes" if id_to_output[fn["inputs"][0]] < id_to_output[fn["inputs"][1]] else "no"
            id_to_output[i] = answer

        elif fn["function"] == "union":
            assert len(fn["inputs"]) == 2
            set1 = {o.id for o in id_to_output[fn["inputs"][0]].objects}
            set2 = {o.id for o in id_to_output[fn["inputs"][1]].objects}
            if medium and len(set1.intersection(set2)) != 0:
                raise SkipException
            id_to_output[i] = ItemCollection(scene).filter_ids(set1.union(set2))
            for o in id_to_output[i].objects:
                if o.id in set1:
                    o.union_branches.append(0)
                if o.id in set2:
                    o.union_branches.append(1)
            id_to_output[i].filters = FilterUnion(
                id_to_output[fn["inputs"][0]].filters, id_to_output[fn["inputs"][1]].filters
            )

        elif fn["function"] == "intersect":
            assert len(fn["inputs"]) == 2
            set1 = {o.id for o in id_to_output[fn["inputs"][0]].objects}
            set2 = {o.id for o in id_to_output[fn["inputs"][1]].objects}
            id_to_output[i] = ItemCollection(scene).filter_ids(set1.intersection(set2))
            id_to_output[i].filters = list(
                set(id_to_output[fn["inputs"][0]].filters + id_to_output[fn["inputs"][1]].filters)
            )

        elif fn["function"] == "exist":
            cur_set = id_to_output[fn["inputs"][0]]
            answer = "no" if len(cur_set) == 0 else "yes"
            node_id = find_node_id_in_template(question["program"], i, template)
            tokens = find_tokens(question, template, node_id)
            assert len(tokens) == 1
            tokens = tokens[0]
            if len(cur_set) != 0:
                for o in cur_set.objects:
                    if medium:
                        for oo in out_obj:
                            if oo[0].id == o.id:
                                raise SkipException
                    out_obj.append((deepcopy(o), tokens))
            id_to_output[i] = answer
        else:
            raise RuntimeError(f"Unimplemented function {fn['function']}")

    if str(answer) != question["answer"]:
        raise RuntimeError(f"Wrong answer found. Expected {question['answer']} but found {answer}")

    # sometimes several parts of the question refer to the same box. We consolidate here.
    obj_by_id = {}
    for o in out_obj:
        if o[0].id in obj_by_id:
            obj_by_id[o[0].id] = (obj_by_id[o[0].id][0], list(set(obj_by_id[o[0].id][1] + o[1])))
        else:
            obj_by_id[o[0].id] = deepcopy(o)
    out_obj = list(obj_by_id.values())

    if verbose:
        print("Scene", ItemCollection(scene))
        print("Caption", question["question"])
        for o in out_obj:
            print("\t matched", o[0], " with ", ", ".join(["'" + question["question"][i:j] + "'" for (i, j) in o[1]]))
    return out_obj


def retrieve_boxes_and_masks(scene, objs):
    """Retrieve bounding boxes and segmentation masks associated to given objs in given scene
    This assumes REFCLEVR annotations
    """
    if len(objs) == 0:
        return [], [], []

    boxes = [scene["objects"][o.id]["bbox"] for o, _ in objs]
    tokens = [tok for _, tok in objs]
    return boxes, None, tokens

    # The dataset use Run-Length-Encoding for the segmentation mask, but unfortunately it is transposed.
    # Here we use coco tools to decode, transpose, re-encode then extract bounding boxes
    raw_rles = [
        {"counts": json.loads("[" + scene["obj_mask"][str(scene["objects"][o.id]["idx"])] + "]"), "size": [480, 320]}
        for o, _ in objs
    ]
    tokens = [tok for _, tok in objs]
    rles = coco_mask.frPyObjects(raw_rles, 320, 480)
    masks = coco_mask.decode(rles).transpose(1, 0, 2)

    rles = coco_mask.encode(np.asfortranarray(masks))
    boxes = coco_mask.toBbox(rles)

    all_seg = []
    for mask in masks.transpose(2, 0, 1):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        all_seg.append(segmentation)
    return boxes, all_seg, tokens


def retrieve_boxes(scene, objs, all_bboxes, cat2obj):
    """Retrieve bounding boxes associated to given objs in given scene.

    This assumes non-REFCLEVR annotations (and thus can't provide a mask)
    """
    all_bbox = {(tuple(c["object"]["bbox"]), c["object"]["category"]) for c in all_bboxes[scene["image_filename"]]}
    all_bbox = [(list(b), c) for b, c in all_bbox]

    assert len(all_bbox) == len(scene["objects"]), "Error, number of boxes doesn't match number of objects"

    def cost(bbox, obj):
        coord, cat = bbox
        ymin, ymax, xmin, xmax = coord
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        return abs(obj.pos[0] - xc) + abs(obj.pos[1] - yc)

    # Retrieve correspondances between boxes and objects
    final_bbox = []
    for o, tokens in objs:
        best_idx = 0
        best_val = 1e9
        for i, b in enumerate(all_bbox):
            if cat2obj[b[1]] != o.get_cat():
                continue
            cur_cost = cost(b, o)
            if cur_cost < best_val:
                best_val = cur_cost
                best_idx = i
        assert cat2obj[all_bbox[best_idx][1]] == o.get_cat(), "Wrong category"
        final_bbox.append((all_bbox[best_idx][0], tokens))
    return convert_bounding_boxes(final_bbox)


def convert_bounding_boxes(bboxes):
    """Accepts a list of bounding boxes in format (xmin,ymin,xmax,ymax)
    Returns the list of boxes in coco format (xmin,ymin, width, height)
    """
    coco_boxes = []
    all_tokens = []
    for b, tokens in bboxes:
        ymin, ymax, xmin, xmax = b
        h = ymax - ymin
        w = xmax - xmin
        coco_boxes.append([xmin, ymin, w, h])
        all_tokens.append(tokens)

    assert len(coco_boxes) == len(all_tokens)
    return coco_boxes, all_tokens


def convert(
    subset: str,
    clevr_path: Path,
    clevr_box_path: Optional[Path],
    refclevr_path: Optional[Path],
    output_path: Path,
    no_caption: bool,
    medium: bool,
    templates,
    next_img_id: int = 0,
    next_id: int = 0,
):
    """Do the heavy lifting on the given subset (eg 'train')"""

    print(f"Exporting {subset}...")

    questions = None
    if not no_caption:
        print("Loading questions...")
        with open(clevr_path / f"questions/CLEVR_{subset}_questions.json") as f:
            questions = json.load(f)["questions"]

    scenes = []
    use_refclevr = refclevr_path is not None
    all_bboxes = []
    cat2obj = []
    if use_refclevr:
        print("Loading scenes (using refclevr annotations)...")
        if subset != "test":
            with open(refclevr_path / f"CLEVR_{subset}_scenes.json") as f:
                scenes = json.load(f)["scenes"]
    else:
        print("Loading scenes...")
        if subset != "test":
            with open(clevr_path / f"scenes/CLEVR_{subset}_scenes.json") as f:
                scenes = json.load(f)["scenes"]

        print("Loading boxes...")
        if subset == "val" or subset == "train":
            # In the bounding box dataset, they renamed val to test for some reason
            sset = "train" if subset == "train" else "test"
            with open(clevr_box_path / f"annotations_{sset}.json") as f:
                all_bboxes = json.load(f)
        else:
            # We don't have bounding boxes for the actual test set.
            all_bboxes = []

        with open(clevr_box_path / "cat2obj.json") as f:
            cat2obj = json.load(f)

    categories = [{"supercategory": "object", "id": 1, "name": "object"}]
    annotations = []
    images = []

    if subset == "test":
        # We don't have scenes nor bounding boxes, simply create image information
        for id in tqdm(range(15000)):
            cur_img = {
                "file_name": "CLEVR_test_{id:06}.png",
                "height": 320,
                "width": 480,
                "id": next_img_id,
            }
            next_img_id += 1
            images.append(cur_img)
    elif no_caption:

        for scene in tqdm(scenes):
            all_objects = ItemCollection(scene).objects
            all_seg = None
            if use_refclevr:
                bboxes, all_seg, _ = retrieve_boxes_and_masks(scene, all_objects)
            else:
                bboxes, _ = retrieve_boxes(scene, all_objects, all_bboxes, cat2obj)
            cur_img = {
                "file_name": scene["image_filename"],
                "height": 320,
                "width": 480,
                "id": next_img_id,
            }
            for i, b in enumerate(bboxes):
                xmin, ymin, w, h = b
                cur_obj = {
                    "area": h * w,
                    "iscrowd": 0,
                    "image_id": next_img_id,
                    "category_id": 1,
                    "id": next_id,
                    "bbox": [xmin, ymin, w, h],
                }
                if all_seg is not None:
                    cur_obj["segmentation"] = all_seg[i]
                next_id += 1
                annotations.append(cur_obj)

            next_img_id += 1
            images.append(cur_img)

    else:
        for qid, question in enumerate(tqdm(questions)):
            scid = question["image_index"]
            scene = scenes[scid]

            try:
                all_objects = parse_prog(scene, question, templates, medium)
            except SkipException:
                # print("skipping", qid)
                continue

            all_seg = None
            if use_refclevr:
                bboxes, all_seg, tokens = retrieve_boxes_and_masks(scene, all_objects)
            else:
                bboxes, tokens = retrieve_boxes(scene, all_objects, all_bboxes, cat2obj)
            cur_img = {
                "file_name": scene["image_filename"],
                "height": 320,
                "width": 480,
                "id": next_img_id,
                "caption": question["question"],
                "answer": question["answer"],
            }
            for i, (b, t) in enumerate(zip(bboxes, tokens)):
                xmin, ymin, w, h = b
                cur_obj = {
                    "area": h * w,
                    "iscrowd": 0,
                    "image_id": next_img_id,
                    "category_id": 1,
                    "id": next_id,
                    "bbox": [xmin, ymin, w, h],
                    "tokens": t,
                }
                if all_seg is not None:
                    cur_obj["segmentation"] = all_seg[i]
                next_id += 1
                annotations.append(cur_obj)

            next_img_id += 1
            images.append(cur_img)

    ds = {"info": [], "licenses": [], "images": images, "annotations": annotations, "categories": categories}
    with open(output_path / f"{subset}.json", "w") as j_file:
        json.dump(ds, j_file)
    return next_img_id, next_id


def main(args):
    clevr_path = Path(args.clevr_path)

    refclevr_path, clevr_box_path = None, None
    if args.use_refclevr_imgs:
        refclevr_path = Path(args.refclevr_path)
    else:
        clevr_box_path = Path(args.clevr_box_path)

    templates = load_templates()

    output_path = Path(args.out_path) if args.out_path is not None else clevr_path

    os.makedirs(str(output_path), exist_ok=True)

    next_img_id, next_id = 0, 0
    sets = ["trainA", "valA", "valB"] if args.cogent else ["val", "train"]
    for s in sets:
        next_img_id, next_id = convert(
            s,
            clevr_path,
            clevr_box_path,
            refclevr_path,
            output_path,
            args.no_caption,
            args.medium,
            templates,
            next_img_id,
            next_id,
        )


if __name__ == "__main__":
    main(parse_args())
