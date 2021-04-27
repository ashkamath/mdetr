# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Subset utils from the VG Phrasecut API
Adapted from https://github.com/ChenyunWu/PhraseCutDataset/blob/b15fb71a1ba692ea3186498f1390e8854b681a66/utils/subset.py

Changelog:
- Formatting (black)
- Make it a class and properly handle paths
"""
"""
Split the dataset into subsets for detailed diagnose. 
Supported subsets:

'all': All the data

'c_coco': phrases with categories from the 80 coco categories.

'c20', 'c100', 'c500', 'c500+', 'c21-100', 'c101-500':
Rank all instance categories by their frequencies in our dataset.
'c100' means all tasks where the described instances are ranked top 100;
'c500+' means ranked after 500;
'c21-100' means ranked between 21 and 100.

'i_single', 'i_multi', 'i_many':
The number of instances described by each phrase.
'i_single': 1 instance;
'i_multi': more than 1 instances;
'i_many': more than 4 instances.

'p_name', 'p_att', 'p_att+', 'p_rel', 'p_rel+', 'p_verbose', 'p_attm', 'p_relm', 'p_att_rel':
Phrase structure when the phrase is generated based on Visual Genome (VG) original boxes.
[Note: instance / mask annotations from our dataset are NOT used to decide these subsets.]
'p_name': Only the "category name" is enough to distinguish against other VG boxes.
'p_att': The phrase contains attributes.
'p_att+': Attributes are needed to distinguish against other VG boxes with the same category.
'p_rel': The phrase contains relationship descriptions.
'p_rel+': Relationship descriptions are needed to distinguish against other VG boxes with the same category.
'p_verbose': Cannot distinguish against other VG boxes.
Used all the annotations of the box to generate the phrase.
'p_attm': multiple atts in one phrase
'p_relm': multiple rels in one phrase
'p_att_rel': one phrase contains both att and rel

's_stuff', 's_obj': Whether the phrase describes stuff or objects.

's_small', 's_mid', 's_large': size of target region relative to image size (< 2%, 2-20%, >20%)

'a20', 'a100', 'a200', 'a21-100', 'a101-200', 'a200+': frequency rank of att, similar as those for category

'a_color', 'a_shape', 'a_material', 'a_texture', 'a_state', 'a_adj', 'a_noun', 'a_loc', 'a_count', 'a_bad':
types of attributes. We have pre-define vocabulary for each type.

"""

import json
import os

subsets = [
    "all",
    "c_coco",
    "c20",
    "c100",
    "c500",
    "c21-100",
    "c101-500",
    "c500+",
    "i_single",
    "i_multi",
    "i_many",
    "p_name",
    "p_att",
    "p_att+",
    "p_rel",
    "p_rel+",
    "p_verbose",
    "p_attm",
    "p_relm",
    "t_stuff",
    "t_obj",
    "s_small",
    "s_mid",
    "s_large",
    "a20",
    "a100",
    "a200",
    "a21-100",
    "a101-200",
    "a200+",
    "a_color",
    "a_shape",
    "a_material",
    "a_texture",
    "a_state",
    "a_adj",
    "a_noun",
    "a_loc",
    "a_count",
    "a_bad",
    "p_att_rel",
    "d_cocotv",
    "d_notcocotv",
]


class PhraseCutSubsets:
    def __init__(self, phrase_cut_path: str) -> None:
        with open(os.path.join(phrase_cut_path, "name_att_rel_count.json"), "r") as f:
            count_info = json.load(f)
        cat_count_list = count_info["cat"]  # list of (cat_name, count), count from high to low
        att_count_list = count_info["att"]
        rel_count_list = count_info["rel"]
        self.cat_sorted = [p[0] for p in cat_count_list]
        self.att_sorted = [p[0] for p in att_count_list]
        self.rel_sorted = [p[0] for p in rel_count_list]

        with open(os.path.join(phrase_cut_path, "image_data_split.json"), "r") as f:
            img_info = json.load(f)

        self.not_coco_trainval = set()
        for img in img_info:
            cid = img.get("coco_id", None)
            if not cid:
                self.not_coco_trainval.add(img["image_id"])

    def get_subset(self, image_id, phrase_structure, gt_boxes, gt_relative_size):
        cond = dict()
        for key in subsets:
            cond[key] = False
        cond["all"] = True

        # # people
        # cond['people'] = False
        # for name in people_names:
        #     if name in phrase:
        #         cond['people'] = True
        #         break
        # cond['non_people'] = not cond['people']

        # top_k: ICCV submission
        # top_k = 501  # top_k starts from 0
        # for ni, name in enumerate(vg500_names):
        #     if name in phrase:
        #         top_k = ni
        #         break

        # d_cocotv: coco trainval or not
        if image_id in self.not_coco_trainval:
            cond["d_notcocotv"] = True
        else:
            cond["d_cocotv"] = True

        # c_coco
        if phrase_structure["name"] in coco:
            cond["c_coco"] = True

        # cat freq ranking
        cat_topk = 501
        if phrase_structure["name"] in self.cat_sorted:
            cat_topk = self.cat_sorted.index(phrase_structure["name"])

        if cat_topk < 20:
            cond["c20"] = True
        elif cat_topk < 100:
            cond["c21-100"] = True
        elif cat_topk < 500:
            cond["c101-500"] = True
        else:
            cond["c500+"] = True

        if cat_topk < 100:
            cond["c100"] = True
        if cat_topk < 500:
            cond["c500"] = True

        # att freq ranking
        att_topk = 201
        for att in phrase_structure["attributes"]:
            if att in self.att_sorted:
                att_topk = min(self.att_sorted.index(att), att_topk)

        if att_topk < 20:
            cond["a20"] = True
        elif att_topk < 100:
            cond["a21-100"] = True
        elif att_topk < 200:
            cond["a101-200"] = True
        else:
            cond["a200+"] = True

        if att_topk < 100:
            cond["a100"] = True
        if att_topk < 200:
            cond["a200"] = True

        # phrase mode
        if phrase_structure:
            if len(phrase_structure["attributes"]) > 0:
                cond["p_att"] = True
            if len(phrase_structure["attributes"]) > 1:
                cond["p_attm"] = True
            if len(phrase_structure["relation_descriptions"]) > 0:
                cond["p_rel"] = True
            if len(phrase_structure["relation_descriptions"]) > 1:
                cond["p_relm"] = True
            if len(phrase_structure["attributes"]) > 0 and len(phrase_structure["relation_descriptions"]) > 0:
                cond["p_att_rel"] = True

            if phrase_structure["type"] == "name":
                cond["p_name"] = True
            if phrase_structure["type"] == "attribute":
                cond["p_att+"] = True
            if phrase_structure["type"] == "relation":
                cond["p_rel+"] = True
            if phrase_structure["type"] == "verbose":
                cond["p_verbose"] = True

        # instance count
        if len(gt_boxes) == 1:
            cond["i_single"] = True
        elif 5 > len(gt_boxes) > 1:
            cond["i_multi"] = True
        elif len(gt_boxes) >= 5:
            cond["i_many"] = True

        # gt size
        if gt_relative_size < 0.02:
            cond["s_small"] = True
        elif gt_relative_size > 0.2:
            cond["s_large"] = True
        else:
            cond["s_mid"] = True

        # stuff or not
        is_stuff = False
        for name in stuff_names:
            # if name in phrase:  # iccv submission
            if name in phrase_structure["name"]:
                is_stuff = True
                break
        if is_stuff:
            cond["t_stuff"] = True
        else:
            cond["t_obj"] = True

        # att type
        if phrase_structure:
            if phrase_structure["attributes"]:
                for att in phrase_structure["attributes"]:
                    if att in att_color:
                        cond["a_color"] = True
                    if att in att_shape:
                        cond["a_shape"] = True
                    if att in att_material:
                        cond["a_material"] = True
                    if att in att_texture:
                        cond["a_texture"] = True
                    if att in att_state:
                        cond["a_state"] = True
                    if att in att_adj:
                        cond["a_adj"] = True
                    if att in att_noun:
                        cond["a_noun"] = True
                    if att in att_loc:
                        cond["a_loc"] = True
                    if att in att_count:
                        cond["a_count"] = True
                    if att in att_bad:
                        cond["a_bad"] = True
        return cond


people_names = [
    "person",
    "people",
    "man",
    "men",
    "woman",
    "women",
    "kid",
    "kids",
    "baby",
    "boy",
    "boys",
    "girl",
    "girls",
    "child",
    "children",
    "lady",
    "player",
    "players",
    "guy",
    "skier",
    "crowd",
    "skateboarder",
    "tennis player",
    "snowboarder",
    "spectators",
    "baseball player",
    "male",
    "skiers",
    "he",
    "passengers",
]
stuff_names = [
    "water",
    "waterdrops",
    "sea",
    "river",
    "fog",
    "ground",
    "field",
    "platform",
    "rail",
    "pavement",
    "road",
    "gravel",
    "mud",
    "dirt",
    "snow",
    "sand",
    "solid",
    "hill",
    "mountain",
    "stone",
    "rock",
    "wood",
    "sky",
    "cloud",
    "vegetation",
    "straw",
    "moss",
    "branch",
    "leaf",
    "leaves",
    "bush",
    "tree",
    "grass",
    "forest",
    "railing",
    "net",
    "cage",
    "fence",
    "building",
    "roof",
    "tent",
    "bridge",
    "skyscraper",
    "house",
    "food",
    "vegetable",
    "salad",
    "textile",
    "banner",
    "blanket",
    "curtain",
    "cloth",
    "napkin",
    "towel",
    "mat",
    "rug",
    "stairs",
    "light",
    "counter",
    "mirror",
    "cupboard",
    "cabinet",
    "shelf",
    "table",
    "desk",
    "door",
    "window",
    "floor",
    "carpet",
    "ceiling",
    "wall",
    "brick",
    "metal",
    "plastic",
    "paper",
    "cardboard",
    "street",
    "snow",
    "shadow",
    "sidewalk",
    "plant",
    "wave",
    "reflection",
    "ocean",
    "beach",
]  # 'flower', 'fruit', 'pillow'

att_color = [
    "white",
    "black",
    "blue",
    "green",
    "brown",
    "red",
    "yellow",
    "gray",
    "grey",
    "silver",
    "orange",
    "dark",
    "pink",
    "tan",
    "purple",
    "beige",
    "bright",
    "gold",
    "colorful",
    "blonde",
    "light brown",
    "light blue",
    "colored",
    "multicolored",
    "maroon",
    "dark blue",
    "dark brown",
    "golden",
    "dark green",
    "black and white",
    "blond",
    "evergreen",
    "light colored",
    "dark grey",
    "multi-colored",
    "light skinned",
    "dark colored",
    "multi colored",
    "blue and white",
    "light green",
    "bright blue",
    "red and white",
    "dark gray",
    "cream colored",
    "light grey",
    "teal",
    "navy blue",
    "turquoise",
    "murky",
    "navy",
]

att_shape = [
    "large",
    "small",
    "tall",
    "long",
    "big",
    "short",
    "round",
    "grassy",
    "little",
    "thick",
    "square",
    "thin",
    "sliced",
    "curved",
    "rectangular",
    "flat",
    "high",
    "wide",
    "stacked",
    "arched",
    "chain link",
    "circular",
    "bent",
    "cut",
    "huge",
    "metallic",
    "cream",
    "pointy",
    "extended",
    "curly",
    "skinny",
    "pointed",
    "narrow",
    "piled",
    "tiny",
    "vertical",
    "oval",
    "curled",
    "row",
    "straight",
    "smaller",
    "triangular",
    "horizontal",
    "crossed",
    "sharp",
    "upside down",
    "pointing",
    "chopped",
    "slice",
    "rectangle",
    "shallow",
    "wispy",
    "rounded",
    "piece",
    "scattered",
    "giant",
    "slanted",
    "tied",
    "sparse",
    "circle",
    "patchy",
    "tilted",
    "fat",
    "upright",
    "larger",
]

att_material = [
    "wooden",
    "metal",
    "wood",
    "brick",
    "cloudy",
    "glass",
    "concrete",
    "plastic",
    "stone",
    "tiled",
    "cement",
    "dirt",
    "sandy",
    "leafy",
    "fluffy",
    "rocky",
    "snowy",
    "leather",
    "steel",
    "paper",
    "chocolate",
    "tile",
    "ceramic",
    "grass",
    "furry",
    "iron",
    "water",
    "stainless steel",
    "hardwood",
    "marble",
    "khaki",
    "cardboard",
    "porcelain",
    "snow covered",
    "asphalt",
    "chrome",
    "rock",
    "wicker",
    "rubber",
    "denim",
    "muddy",
    "foamy",
    "granite",
    "bricked",
    "gravel",
    "snow-covered",
    "clay",
    "sand",
    "red brick",
]

att_texture = [
    "clear",
    "wet",
    "striped",
    "dirty",
    "paved",
    "shiny",
    "painted",
    "dry",
    "plaid",
    "clean",
    "blurry",
    "hazy",
    "floral",
    "rusty",
    "splashing",
    "cloudless",
    "worn",
    "smooth",
    "checkered",
    "spotted",
    "patterned",
    "reflecting",
    "wrinkled",
    "reflective",
    "shining",
    "choppy",
    "rough",
    "reflected",
    "rusted",
    "lined",
    "fuzzy",
    "blurred",
    "faded",
    "printed",
    "foggy",
    "dusty",
    "glazed",
    "rippled",
    "transparent",
    "frosted",
]

att_state = [
    "standing",
    "open",
    "sitting",
    "walking",
    "parked",
    "hanging",
    "playing",
    "closed",
    "empty",
    "on",
    "looking",
    "watching",
    "flying",
    "eating",
    "skiing",
    "covered",
    "surfing",
    "skateboarding",
    "full",
    "jumping",
    "holding",
    "close",
    "leaning",
    "running",
    "riding",
    "folded",
    "waiting",
    "moving",
    "laying",
    "grazing",
    "off",
    "talking",
    "parking",
    "calm",
    "posing",
    "crashing",
    "melted",
    "skating",
    "seated",
    "raised",
    "playing tennis",
    "sleeping",
    "opened",
    "broken",
    "resting",
    "dried",
    "snowboarding",
    "crouching",
    "driving",
    "fried",
    "swinging",
    "cracked",
    "drinking",
    "burnt",
    "kneeling",
    "stopped",
    "rolling",
    "sitting down",
    "trimmed",
    "breaking",
    "crouched",
    "bending",
    "dressed",
    "standing up",
    "wrapped",
    "attached",
    "floating",
    "rolled up",
    "lying",
    "squatting",
    "held",
    "cutting",
    "outstretched",
    "illuminated",
    "reading",
    "turned",
    "swimming",
    "turning",
]

att_adj = [
    "young",
    "old",
    "smiling",
    "bare",
    "light",
    "part",
    "dead",
    "cooked",
    "framed",
    "pictured",
    "overcast",
    "leafless",
    "beautiful",
    "stuffed",
    "growing",
    "decorative",
    "electrical",
    "electric",
    "bald",
    "older",
    "lit",
    "fresh",
    "lush",
    "wire",
    "happy",
    "puffy",
    "sunny",
    "ripe",
    "male",
    "palm",
    "shirtless",
    "female",
    "asian",
    "hairy",
    "ornate",
    "bushy",
    "deep",
    "wavy",
    "toasted",
    "barefoot",
    "potted",
    "short sleeved",
    "edge",
    "wild",
    "busy",
    "decorated",
    "double decker",
    "long sleeved",
    "partial",
    "soft",
    "flat screen",
    "healthy",
    "floppy",
    "plain",
    "filled",
    "modern",
    "long sleeve",
    "overgrown",
    "displayed",
    "digital",
    "cast",
    "airborne",
    "delicious",
    "hard",
    "carpeted",
    "heavy",
    "new",
    "grilled",
    "sleeveless",
    "pale",
    "pretty",
    "different",
    "american",
    "nice",
    "fake",
    "designed",
    "cute",
    "manicured",
    "written",
]

att_noun = [
    "tennis",
    "baseball",
    "man's",
    "baby",
    "train",
    "woman's",
    "pine",
    "tree",
    "street",
    "passenger",
    "traffic",
    "computer",
    "adult",
    "ski",
    "man",
    "wine",
    "burgundy",
    "stop",
    "snow",
    "bathroom",
    "city",
    "teddy",
    "kitchen",
    "patch",
    "nike",
    "woman",
    "wall",
    "fire",
    "clock",
    "window",
    "straw",
    "flower",
    "ground",
    "pizza",
    "apple",
    "power",
    "coffee",
    "tennis player",
    "toy",
    "ocean",
]

att_loc = [
    "distant",
    "background",
    "back",
    "behind",
    "in background",
    "side",
    "up",
    "rear",
    "down",
    "top",
    "far",
    "overhead",
    "low",
    "above",
    "outdoors",
    "in distance",
    "in the background",
    "inside",
    "outdoor",
    "bottom",
    "in air",
]

att_count = [
    "one",
    "three",
    "four",
    "many",
    "pair",
    "several",
    "double",
    "grouped",
    "group",
    "together",
    "bunch",
    "alone",
    "set",
    "single",
]

att_bad = ["here", "present", "wearing", "in the picture", "some", "daytime", "existing", "ready", "made", "in picture"]

coco = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
