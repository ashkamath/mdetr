export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export CUBLAS_WORKSPACE_CONFIG=:16:8

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --dataset_config configs/pretrain.json --ema --backbone timm_tf_efficientnet_b3_ns --lr_backbone 5e-5 --batch_size 4 --output-dir ./mdetr/pretrain_batch_4
