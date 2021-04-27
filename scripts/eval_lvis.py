# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import argparse
import json
import os
import random
import re
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import main as detection
import util.dist as dist
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.lvis_eval import LvisDumper, LvisEvaluatorFixedAP
from engine import evaluate
from models import build_model
from models.postprocessors import build_postprocessors
from util.metrics import MetricLogger


def get_args_parser():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser("Evaluate MDETR on LVIS detection", parents=[detection_parser], add_help=False)
    parser.add_argument("--lvis_minival_path", type=str, default="")
    return parser


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    args.batch_size = 1
    print(args)

    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    # fix the seed for reproducibility
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location="cpu")

    model_args = checkpoint["args"]
    for a in vars(args):
        if a not in vars(model_args):
            vars(model_args)[a] = vars(args)[a]

    model_args.device = args.device
    model, _, _, _, _ = build_model(model_args)
    model.to(device)
    with open(Path(args.lvis_minival_path) / "lvis_v1_minival.json", "r") as f:
        lvis_val = json.load(f)
    id2cat = {c["id"]: c for c in lvis_val["categories"]}
    all_cats = sorted(list(id2cat.keys()))

    # model_without_ddp = model
    # if args.distributed:
    #    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    dset = build_dataset("lvis", image_set="minival", args=args)
    sampler = DistributedSampler(dset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dset)
    dataloader = DataLoader(
        dset,
        args.batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=partial(utils.collate_fn, False),
        num_workers=args.num_workers,
    )

    if args.test:
        evaluator = LvisDumper(fixed_ap=True, out_path=os.path.join(args.output_dir, "lvis_eval"))
    else:
        evaluator = LvisEvaluatorFixedAP(dset.lvis, fixed_ap=True)

    postprocessor = build_postprocessors(args, "lvis")
    model.load_state_dict(checkpoint["model_ema"], strict=False)
    model.eval()
    label_set = torch.as_tensor(list(all_cats))
    print("label_set", len(label_set))
    text_memories = []
    print("encoding text...")
    splits = torch.split(label_set, 32)
    for split in tqdm(splits):
        captions = [f"{clean_name(id2cat[l]['name'])}" for l in split.tolist()]
        tokenized = model.transformer.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(
            device
        )
        encoded_text = model.transformer.text_encoder(**tokenized)

        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = model.transformer.resizer(text_memory)

        text_memories.append((text_attention_mask, text_memory_resized, tokenized))

    for batch_dict in tqdm(dataloader):

        samples = batch_dict["samples"].to(device)
        targets = batch_dict["targets"]
        targets = [
            {
                k: v.to(device) if k not in ["tokens_positive", "tokens", "dataset_name"] else v
                for k, v in t.items()
                if k != "caption"
            }
            for t in targets
        ]

        assert len(targets) == 1
        t = targets[0]

        with torch.no_grad():
            features, orig_pos = model.backbone(samples)
            orig_src, orig_mask = features[-1].decompose()
            res = []
            for i in range(len(text_memories)):
                # captions = [f"{clean_name(id2cat[l]['name'])}" for l in split.tolist()]
                bs = len(splits[i])
                src = orig_src.repeat(bs, 1, 1, 1)
                mask = orig_mask.repeat(bs, 1, 1, 1)
                pos = deepcopy(orig_pos)
                pos[0] = pos[0].repeat(bs, 1, 1, 1)

                memory_cache = model.transformer(
                    model.input_proj(src),
                    mask,
                    model.query_embed.weight,
                    pos[-1],
                    text_memories[i],  # captions,
                    encode_and_save=True,
                    text_memory=None,
                    img_memory=None,
                    text_attention_mask=None,
                )
                out = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
                orig_target_sizes = torch.stack([t["orig_size"] for _ in range(bs)], dim=0)
                results = postprocessor["bbox"](out, orig_target_sizes)

                assert len(results) == len(splits[i])
                for j in range(len(results)):
                    results[j]["labels"] *= splits[i][j].item()

                for output in results:
                    res.append((t["image_id"].item(), output))
            evaluator.update(res)
    evaluator.synchronize_between_processes()
    evaluator.summarize()


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LVIS evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
