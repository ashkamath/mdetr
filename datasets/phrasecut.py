# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
Data class for the PhraseCut dataset. The task considered is referring expression segmentation.
"""

from pathlib import Path

from transformers import RobertaTokenizerFast

from .coco import ModulatedDetection, make_coco_transforms


class PhrasecutDetection(ModulatedDetection):
    pass


def build(image_set, args):

    img_dir = Path(args.vg_img_path)
    if image_set == "val":
        # We validate on the minival for efficiency
        image_set = "miniv"

    if image_set == "miniv":
        ann_file = Path(args.phrasecut_ann_path) / f"finetune_phrasecut_miniv.json"
        image_set = "val"
    else:
        ann_file = Path(args.phrasecut_ann_path) / f"finetune_phrasecut_{image_set}.json"

    if args.test:
        ann_file = Path(args.phrasecut_ann_path) / f"finetune_phrasecut_test.json"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = PhrasecutDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=args.masks,
        return_tokens=True,  # args.contrastive_align_loss,
        tokenizer=tokenizer,
    )
    return dataset
