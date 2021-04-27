# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
from pathlib import Path

from transformers import RobertaTokenizerFast

import datasets.transforms as T

from .coco import ModulatedDetection, make_coco_transforms


class LvisModulatedDetection(ModulatedDetection):
    pass


def build(image_set, args):

    img_dir = Path(args.coco2017_path)
    if args.lvis_subset is None or int(args.lvis_subset) == 100:
        ann_file = Path(args.modulated_lvis_ann_path) / f"finetune_lvis_{image_set}.json"
    else:
        ann_file = Path(args.modulated_lvis_ann_path) / f"finetune_lvis{args.lvis_subset}_{image_set}.json"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = LvisModulatedDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=False,
        return_tokens=True,  # args.contrastive_align_loss,
        tokenizer=tokenizer,
    )
    return dataset
