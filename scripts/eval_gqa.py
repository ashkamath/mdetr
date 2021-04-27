# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import argparse
import json
import os
import random
import sys
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import main as detection
import util.dist as dist
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
from util.metrics import MetricLogger


def get_args_parser():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser(
        "Get predictions for GQA and dump to file", parents=[detection_parser], add_help=False
    )
    parser.add_argument("--split", type=str, default="testdev", choices=("testdev", "test", "challenge", "submission"))
    return parser


def main(args):
    dist.init_distributed_mode(args)

    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    print("git:\n  {}\n".format(utils.get_sha()))
    if args.mask_model != "none":
        args.masks = True

    print(args)

    device = torch.device(args.device)

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_deterministic(True)

    model, criterion, _, _, _ = build_model(args)
    model.to(device)

    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    if len(args.combine_datasets_val) == 0:
        raise RuntimeError("Please provide at leas one validation dataset")

    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])

    val_tuples = []
    for dset_name in args.combine_datasets_val:
        dset = build_dataset(dset_name, image_set=args.split, args=args)
        sampler = (
            DistributedSampler(dset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dset)
        )
        dataloader = DataLoader(
            dset,
            args.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(utils.collate_fn, False),
            num_workers=args.num_workers,
        )
        base_ds = get_coco_api_from_dataset(dset)
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=dataloader, base_ds=base_ds, evaluator_list=None))

    if args.load:
        print("loading from", args.load)
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        if "model_ema" in checkpoint:
            model_without_ddp.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        model_ema = deepcopy(model_without_ddp)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if args.ema:
            if "model_ema" not in checkpoint:
                print("WARNING: ema model not found in checkpoint, resetting to current model")
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])

    with open(Path(args.gqa_ann_path) / "gqa_answer2id.json", "r") as f:
        answer2id = json.load(f)
    with open(Path(args.gqa_ann_path) / "gqa_answer2id_by_type.json", "r") as f:
        answer2id_by_type = json.load(f)

    id2answer = {v: k for k, v in answer2id.items()}

    id2answerbytype = {}
    for type, answer_dict in answer2id_by_type.items():
        curr_reversed_dict = {v: k for k, v in answer2id_by_type[type].items()}
        id2answerbytype[type] = curr_reversed_dict

    print("Running evaluation")

    test_model = model_ema if model_ema is not None else model
    for i, item in enumerate(val_tuples):
        evaluator_list = []
        evaluator_list.append(GQAEvaluator())
        item = item._replace(evaluator_list=evaluator_list)

        evaluate(
            test_model,
            criterion,
            item.dataloader,
            item.evaluator_list,
            device,
            args.output_dir,
            args,
            id2answer,
            id2answerbytype,
        )

    return


class GQAEvaluator:
    def __init__(self):
        self.predictions = []

    def update(self, res: List[Dict]):
        self.predictions += res

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def dump_result(self, path):

        if dist.is_main_process():
            with open(path, "w") as f:
                json.dump(self.predictions, f, indent=4, sort_keys=True)


@torch.no_grad()
def evaluate(
    model,
    criterion,
    data_loader,
    evaluator_list,
    device,
    output_dir,
    args,
    id2answer,
    id2answerbytype,
):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    gqa_evaluator = evaluator_list[0]

    for batch_dict in metric_logger.log_every(data_loader, 10, header):
        samples = batch_dict["samples"].to(device)
        targets = batch_dict["targets"]

        captions = [t["caption"] for t in targets]

        memory_cache = model(samples, captions, encode_and_save=True)
        outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

        if not args.split_qa_heads:
            preds = outputs["pred_answer"].argmax(-1)
            answers = [id2answer[x.item()] for x in preds]

        else:
            answers = []
            answer_types = outputs["pred_answer_type"].argmax(-1)
            answer_types = [x.item() for x in answer_types]
            for i, ans_type in enumerate(answer_types):
                if ans_type == 0:
                    answers.append(id2answerbytype["answer_obj"][outputs["pred_answer_obj"][i].argmax(-1).item()])
                elif ans_type == 1:
                    answers.append(id2answerbytype["answer_attr"][outputs["pred_answer_attr"][i].argmax(-1).item()])
                elif ans_type == 2:
                    answers.append(id2answerbytype["answer_rel"][outputs["pred_answer_rel"][i].argmax(-1).item()])
                elif ans_type == 3:
                    answers.append(id2answerbytype["answer_global"][outputs["pred_answer_global"][i].argmax(-1).item()])
                elif ans_type == 4:
                    answers.append(id2answerbytype["answer_cat"][outputs["pred_answer_cat"][i].argmax(-1).item()])
                else:
                    assert False, "must be one of the answer types"

        res = [{"questionId": target["questionId"], "prediction": answer} for target, answer in zip(targets, answers)]
        gqa_evaluator.update(res)

    gqa_evaluator.synchronize_between_processes()

    results_path = os.path.join(output_dir, f"{args.split}_predictions.json")
    gqa_evaluator.dump_result(results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Get predictions for GQA and dump to file", parents=[get_args_parser()], add_help=False
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Note: remember to choose between all/ balanced using --gqa_split_type in the config

    main(args)
