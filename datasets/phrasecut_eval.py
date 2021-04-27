# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import os
from collections import defaultdict
from typing import Dict, List

import util.dist as dist

from .phrasecut_utils.evaluator import Evaluator
from .phrasecut_utils.refvg_loader import RefVGLoader
from .phrasecut_utils.subset import PhraseCutSubsets


class PhrasecutEvaluator(object):
    def __init__(self, split, ann_folder, output_dir="phrasecut_eval", eval_mask=False):
        subset = PhraseCutSubsets(ann_folder)
        loader = RefVGLoader(ann_folder, subset, split=split)
        if dist.is_main_process():
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        self.output_dir = output_dir
        self.evaluator = Evaluator(loader, summary_path=output_dir)
        self.eval_mask = eval_mask
        self.predictions = []

    def update(self, predictions):
        self.predictions += predictions

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        if dist.is_main_process():
            imgid2pred: Dict[str, List] = defaultdict(list)
            for p in self.predictions:
                imgid2pred[p["original_id"]].append(p)

            for img_id, pred in imgid2pred.items():
                im_pred_dict = {p["task_id"]: p for p in pred}
                self.evaluator.eval_single_img(
                    img_id,
                    im_pred_dict,
                    pred_mask_tag="masks" if self.eval_mask else None,
                    pred_boxes_tag="boxes",
                    verbose=False,
                )

            mask_box = ["box"]
            if self.eval_mask:
                mask_box.append("mask")
            results = self.evaluator.analyze_stats(mask_box, exp_name_in_summary=None, save_result_to_path=None)

            results = results["all"]["pred_box_acc"]
            return {f"Precision@{k}": v for k, v in results.items()}
        return None
