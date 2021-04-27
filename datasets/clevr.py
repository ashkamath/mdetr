# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from transformers import RobertaTokenizerFast

import datasets.transforms as T

from .coco import ConvertCocoPolysToMask, create_positive_map

ALL_ATTRIBUTES = [
    "small",
    "large",
    "gray",
    "red",
    "blue",
    "green",
    "brown",
    "purple",
    "cyan",
    "yellow",
    "cube",
    "sphere",
    "cylinder",
    "rubber",
    "metal",
]


def _encode_answer(target, answer):
    if answer in ["yes", "no"]:
        target["answer_type"] = torch.as_tensor(0, dtype=torch.long)
        target["answer_binary"] = torch.as_tensor(0.0 if answer == "no" else 1.0)
        target["answer_attr"] = torch.as_tensor(-100, dtype=torch.long)
        target["answer_reg"] = torch.as_tensor(-100, dtype=torch.long)
    elif answer in ALL_ATTRIBUTES:
        target["answer_type"] = torch.as_tensor(1, dtype=torch.long)
        target["answer_binary"] = torch.as_tensor(0.0)
        target["answer_attr"] = torch.as_tensor(ALL_ATTRIBUTES.index(answer), dtype=torch.long)
        target["answer_reg"] = torch.as_tensor(-100, dtype=torch.long)
    else:
        target["answer_type"] = torch.as_tensor(2, dtype=torch.long)
        target["answer_binary"] = torch.as_tensor(0.0)
        target["answer_attr"] = torch.as_tensor(-100, dtype=torch.long)
        target["answer_reg"] = torch.as_tensor(int(answer), dtype=torch.long)
    return target


class ClevrDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, return_tokens, tokenizer, do_qa):
        super(ClevrDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens)
        self.tokenizer = tokenizer
        self.return_tokens = return_tokens
        self.do_qa = do_qa

    def __getitem__(self, idx):
        img, target = super(ClevrDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        img_data = self.coco.loadImgs(image_id)[0]
        caption = img_data["caption"] if "caption" in img_data else None
        target = {"image_id": image_id, "annotations": target, "caption": caption}
        img, target = self.prepare(img, target)
        if self.do_qa:
            answer = img_data["answer"]
            target = _encode_answer(target, answer)

        if self.return_tokens:
            assert len(target["boxes"]) == len(target["tokens_positive"])
            tokenized = self.tokenizer(caption, return_tensors="pt")
            # construct a map such that positive_map[i,j] = True iff box i is associated to token j
            target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ClevrQuestion(torch.utils.data.Dataset):
    """Dataset for eval only. Provides the question and the image"""

    def __init__(self, img_folder, ann_file, transforms):
        super(ClevrQuestion, self).__init__()
        self.transforms = transforms
        self.root = img_folder
        with open(ann_file, "r") as f:
            self.questions = json.load(f)["questions"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        img = Image.open(os.path.join(self.root, question["image_filename"])).convert("RGB")
        target = {
            "questionId": question["question_index"] if "question_index" in question else idx,
            "caption": question["question"],
        }
        if "answer" in question:
            target = _encode_answer(target, question["answer"])

        if self.transforms is not None:
            img, _ = self.transforms(
                img,
                {
                    "boxes": torch.zeros(0, 4),
                    "labels": torch.zeros(0),
                    "iscrowd": torch.zeros(0),
                    "positive_map": torch.zeros(0),
                },
            )
        return img, target


def make_clevr_transforms(image_set, cautious=False):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [256, 288, 320, 352, 384]

    if image_set == "train":
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=512),
                    T.Compose(
                        [
                            T.RandomResize([320, 352, 384]),
                            T.RandomSizeCrop(256, 512, respect_boxes=cautious),
                            T.RandomResize(scales, max_size=512),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                # T.RandomResize([480], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(dataset_file, image_set, args):

    if dataset_file == "clevr_question":
        if args.clevr_variant == "humans":
            assert args.no_detection, "CLEVR-Humans doesn't have boxes, please disable detection"
            im_set = image_set
            if args.test:
                im_set = "test"
            ann_file = Path(args.clevr_ann_path) / f"CLEVR-Humans-{im_set}.json"
            img_dir = Path(args.clevr_img_path) / f"{im_set}"
            image_set = "train" if im_set == "train" else "val"
        elif args.clevr_variant == "cogent":
            assert image_set != "train", "Please train CoGenT with 'clevr' dataset, not 'clevr_question'"
            im_set = args.cogent_set
            ann_file = Path(args.clevr_ann_path) / f"CLEVR_{im_set}_questions.json"
            img_dir = Path(args.clevr_img_path) / f"{im_set}"
            image_set = "train" if im_set == "train" else "val"
        elif args.clevr_variant == "normal":
            im_set = image_set
            if args.test:
                im_set = "test"

            ann_file = Path(args.clevr_ann_path) / f"CLEVR_{im_set}_questions.json"
            img_dir = Path(args.clevr_img_path) / f"{im_set}"
            image_set = "train" if im_set == "train" else "val"
        else:
            assert False, f"Unknown clevr variant {args.clevr_variant}"
        print("loading ", img_dir, ann_file)
        return ClevrQuestion(
            img_dir,
            ann_file,
            transforms=make_clevr_transforms(image_set, cautious=True),
        )
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)

    img_dir = Path(args.clevr_img_path) / f"{image_set}"
    ann_file = Path(args.clevr_ann_path) / f"{image_set}.json"

    if args.clevr_variant == "cogent":
        im_set = "trainA" if image_set == "train" else "valA"
        img_dir = Path(args.clevr_img_path) / f"{image_set}A"

    dataset = ClevrDetection(
        img_dir,
        ann_file,
        transforms=make_clevr_transforms(image_set, cautious=True),
        return_masks=False,
        return_tokens=True,
        tokenizer=tokenizer,
        do_qa=args.do_qa,
    )

    return dataset
