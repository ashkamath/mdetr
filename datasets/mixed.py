# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets.vision import VisionDataset
from transformers import RobertaTokenizerFast

from .coco import ConvertCocoPolysToMask, make_coco_transforms


class CustomCocoDetection(VisionDataset):
    """Coco-style dataset imported from TorchVision.
    It is modified to handle several image sources


    Args:
        root_coco (string): Path to the coco images
        root_vg (string): Path to the vg images
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root_coco: str,
        root_vg: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(CustomCocoDetection, self).__init__(root_coco, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root_coco = root_coco
        self.root_vg = root_vg

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        dataset = img_info["data_source"]

        cur_root = self.root_coco if dataset == "coco" else self.root_vg
        img = Image.open(os.path.join(cur_root, path)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)


class MixedDetection(CustomCocoDetection):
    """Same as the modulated detection dataset, except with multiple img sources"""

    def __init__(self, img_folder_coco, img_folder_vg, ann_file, transforms, return_masks, return_tokens, tokenizer):
        super(MixedDetection, self).__init__(img_folder_coco, img_folder_vg, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer)

    def __getitem__(self, idx):
        img, target = super(MixedDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        caption = self.coco.loadImgs(image_id)[0]["caption"]
        target = {"image_id": image_id, "annotations": target, "caption": caption}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def build(image_set, args):
    vg_img_dir = Path(args.vg_img_path)
    coco_img_dir = Path(args.coco_path) / f"{image_set}2014"
    assert vg_img_dir.exists(), f"provided VG img path {vg_img_dir} does not exist"
    assert coco_img_dir.exists(), f"provided coco img path {coco_img_dir} does not exist"

    ann_file = Path(args.gqa_ann_path) / f"final_mixed_{image_set}.json"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = MixedDetection(
        coco_img_dir,
        vg_img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=args.masks,
        return_tokens=True,
        tokenizer=tokenizer,
    )

    return dataset
