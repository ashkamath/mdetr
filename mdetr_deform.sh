#!/usr/bin/env bash

set -x

EXP_DIR=exps/mdetr_deformable_detr
PY_ARGS=${@:1}

python -u main.py --output-dir ${EXP_DIR} --dataset_config configs/pretrain.json --ema \
      --backbone timm_tf_efficientnet_b5_ns  --lr_backbone 5e-5 --transformer "Deformable-DETR" ${PY_ARGS}
