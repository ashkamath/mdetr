# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""This script allows to dump the model's prediction on an arbitrary split of CLEVR/CoGenT/CLEVR-Humans"""
import argparse
import json
import os
import random
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.utils
from torch.utils.data import DataLoader, DistributedSampler

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import main as detection
import util.dist as dist
import util.misc as utils
from datasets import build_dataset
from datasets.clevr import ALL_ATTRIBUTES
from engine import evaluate
from models import build_model
from util.metrics import MetricLogger


def local_get_args_parser():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser(
        "Get predictions for clevr and dump to file", parents=[detection_parser], add_help=False
    )
    parser.add_argument("--split", type=str, default="val", choices=("val", "test", "testA", "testB", "valA", "valB"))
    parser.add_argument("--clevr_eval_path", type=str, default="")
    return parser


def main(args):
    utils.init_distributed_mode(args)

    # Update dataset specific configs
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    if args.mask_model != "none":
        args.masks = True

    args.clevr_ann_path = args.clevr_eval_path
    print(args)

    device = torch.device(args.device)

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_deterministic(True)

    if args.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location="cpu")

    model_args = checkpoint["args"]
    model_args.device = args.device

    model_args.combine_datasets = ["clevr_question"]
    for a in vars(args):
        if a not in vars(model_args):
            vars(model_args)[a] = vars(args)[a]

    model, _, _, _, _ = build_model(model_args)
    if "ema" in args and args.ema:
        assert "model_ema" in checkpoint
        model.load_state_dict(checkpoint["model_ema"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    dataset = build_dataset("clevr_question", image_set=args.split, args=args)
    sampler = (
        DistributedSampler(dataset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dataset)
    )
    dataloader = DataLoader(
        dataset,
        args.batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=partial(utils.collate_fn, False),
        num_workers=args.num_workers,
    )

    evaluator_list = [CLEVREvaluator()]

    evaluate(
        model,
        dataloader,
        evaluator_list,
        device,
        args.output_dir,
        args,
    )

    return


class CLEVREvaluator:
    def __init__(self):
        self.predictions = []

    def update(self, res: List[Dict]):
        self.predictions += res

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        self.predictions = sum(all_predictions, [])
        self.predictions = sorted(self.predictions, key=lambda x: x["questionId"])
        self.predictions = [p["prediction"] for p in self.predictions]

    def dump_result(self, path):

        if dist.is_main_process():
            with open(path, "w") as f:
                for p in self.predictions:
                    f.write(p)
                    f.write("\n")


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    evaluator_list,
    device,
    output_dir,
    args,
):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    clevr_evaluator = evaluator_list[0]

    for batch_dict in metric_logger.log_every(data_loader, 10, header):
        samples = batch_dict["samples"].to(device)
        targets = batch_dict["targets"]
        captions = [t["caption"] for t in targets]

        memory_cache = model(samples, captions, encode_and_save=True)
        outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
        answers = []
        answer_types = outputs["pred_answer_type"].argmax(-1)
        answer_types = [x.item() for x in answer_types]
        for i, ans_type in enumerate(answer_types):
            if ans_type == 0:
                answers.append("yes" if outputs["pred_answer_binary"][i].sigmoid() > 0.5 else "no")
            elif ans_type == 1:
                answers.append(ALL_ATTRIBUTES[outputs["pred_answer_attr"][i].argmax(-1).item()])
            elif ans_type == 2:
                answers.append(str(outputs["pred_answer_reg"][i].argmax(-1).item()))
            else:
                assert False, "must be one of the answer types"

        res = [{"questionId": target["questionId"], "prediction": answer} for target, answer in zip(targets, answers)]
        clevr_evaluator.update(res)

    clevr_evaluator.synchronize_between_processes()

    results_path = os.path.join(output_dir, f"{args.split}_predictions.txt")
    clevr_evaluator.dump_result(results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Get predictions for CLEVR and dump to file", parents=[local_get_args_parser()], add_help=False
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
