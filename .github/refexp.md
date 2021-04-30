# Referring Expression Comprehension

### RefCOCO 

| Backbone | Pre-training Image Data | Val | TestA  | TestB | url | size |
|----------|---------|---------|-----------|----------|-----------|-----------|
| Resnet-101| COCO+VG+Flickr | 86.75   |  89.58   |   81.41  | [model](https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth?download=1)   |  3GB   |   
| EfficientNet-B3| COCO+VG+Flickr |  87.51  | 90.40  | 82.67 | [model](https://zenodo.org/record/4721981/files/refcoco_EB3_checkpoint.pth?download=1)  |  2.4GB   | 

### RefCOCO+

| Backbone | Pre-training Image Data | Val | TestA  | TestB | url | size |
|----------|---------|---------|-----------|----------|-----------|-----------|
| Resnet-101| COCO+VG+Flickr | 79.52   |  84.09  |   70.62  | [model](https://zenodo.org/record/4721981/files/refcoco%2B_resnet101_checkpoint.pth?download=1)   |  3GB  |   
| EfficientNet-B3| COCO+VG+Flickr |  81.13  | 85.52  | 72.96 | [model](https://zenodo.org/record/4721981/files/refcoco%2B_EB3_checkpoint.pth?download=1)   | 2.4GB   | 

### RefCOCOg

| Backbone | Pre-training Image Data | Val | Test  |  url | size |
|----------|---------|---------|-----------|----------|-----------|
| Resnet-101| COCO+VG+Flickr | 81.64 | 80.89    | [model](https://zenodo.org/record/4721981/files/refcocog_resnet101_checkpoint.pth?download=1)   |   3GB  |   
| EfficientNet-B3| COCO+VG+Flickr |  83.35  | 83.31  | [model](https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth?download=1)  | 2.4GB   | 



### Data preparation
There are three datasets which have the same structure, to be used with three config files: configs/refcoco, configs/refcoco+ and configs/refcocog. Here we show instructions for refcoco but the same applies for the others.
The config for this dataset can be found in configs/refcoco.json and is also shown below:

```json
{
    "combine_datasets": ["refexp"],
    "combine_datasets_val": ["refexp"],
    "refexp_dataset_name": "refcoco",
    "coco_path": "",
    "refexp_ann_path": "mdetr_annotations/",
}
```

The images for this dataset come from the COCO 2014 train split which can be downloaded from : [Coco train2014](http://images.cocodataset.org/zips/train2014.zip). Update the "coco_path" to the folder containing the downloaded images.
Download our [pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1) and place them in a folder called "mdetr_annotations". The `refexp_ann_path` should point to this folder.

### Script to reproduce results

Model weights (can also be loaded directly from url): 
1. [refcoco_resnet101_checkpoint.pth](https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth?download=1)
2. [refcoco_EB3_checkpoint.pth](https://zenodo.org/record/4721981/files/refcoco_EB3_checkpoint.pth?download=1)
3. [refcoco+_resnet101_checkpoint.pth](https://zenodo.org/record/4721981/files/refcoco%2B_resnet101_checkpoint.pth?download=1)
4. [refcoco+_EB3_checkpoint.pth](https://zenodo.org/record/4721981/files/refcoco%2B_EB3_checkpoint.pth?download=1)
5. [refcocog_resnet101_checkpoint.pth](https://zenodo.org/record/4721981/files/refcocog_resnet101_checkpoint.pth?download=1)
6. [refcocog_EB3_checkpoint.pth](https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth?download=1)

#### RefCOCO

The script to run the evaluation using the correct checkpoints are given below.
For test results, pass --test and --test_type test or testA or testB according to the dataset.

MDETR-Resnet101:
```
python run_with_submitit.py --dataset_config configs/refcoco.json  --batch_size 4  --resume https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth  --ngpus 1 --nodes 2 --ema --eval
```

Alternatively to run on a single node with 2 gpus
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/refcoco.json --batch_size 4  --resume https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth  --ema --eval
```


MDETR-EB3:
```
python run_with_submitit.py --dataset_config configs/refcoco.json  --backbone "timm_tf_efficientnet_b3_ns" --batch_size 4  --resume https://zenodo.org/record/4721981/files/refcoco_EB3_checkpoint.pth  --ngpus 1 --nodes 2 --ema --eval
```

Alternatively to run on a single node with 2 gpus
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/refcoco.json --batch_size 4 --backbone "timm_tf_efficientnet_b3_ns"  --resume https://zenodo.org/record/4721981/files/refcoco_EB3_checkpoint.pth  --ema --eval
```


#### RefCOCO+

MDETR-Resnet101:
```
python run_with_submitit.py --dataset_config configs/refcoco+.json  --batch_size 4 --resume https://zenodo.org/record/4721981/files/refcoco%2B_resnet101_checkpoint.pth  --ngpus 1 --nodes 2 --ema --eval
```

Alternatively to run on a single node with 2 gpus
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/refcoco+.json --batch_size 4  --resume https://zenodo.org/record/4721981/files/refcoco%2B_resnet101_checkpoint.pth  --ema --eval
```

MDETR-EB3:
```
python run_with_submitit.py --dataset_config configs/refcoco+.json  --backbone "timm_tf_efficientnet_b3_ns" --batch_size 4 --resume https://zenodo.org/record/4721981/files/refcoco%2B_EB3_checkpoint.pth  --ngpus 1 --nodes 2 --ema --eval
```

Alternatively to run on a single node with 2 gpus
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/refcoco+.json --batch_size 4 --backbone "timm_tf_efficientnet_b3_ns"  --resume https://zenodo.org/record/4721981/files/refcoco%2B_EB3_checkpoint.pth --ema --eval
```


#### RefCOCOg

MDETR-Resnet101:
```
python run_with_submitit.py --dataset_config configs/refcocog.json  --batch_size 4 --resume https://zenodo.org/record/4721981/files/refcocog_resnet101_checkpoint.pth  --ngpus 1 --nodes 2 --ema --eval
```

Alternatively to run on a single node with 2 gpus
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/refcocog.json --batch_size 4  --resume https://zenodo.org/record/4721981/files/refcocog_resnet101_checkpoint.pth  --ema --eval
```


MDETR-EB3:
```
python run_with_submitit.py --dataset_config configs/refcocog.json  --backbone "timm_tf_efficientnet_b3_ns"  --batch_size 4  --resume https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth --ngpus 1 --nodes 2 --ema --eval
```

Alternatively to run on a single node with 2 gpus
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/refcocog.json --batch_size 4 --backbone "timm_tf_efficientnet_b3_ns"  --resume https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth --ema --eval
```



### Finetuning instructions

To finetune on one of these datasets from the pre-trained checkpoint, use the following (example shown for refcoco):

```
python run_with_submitit.py --dataset_config configs/refcoco.json  --batch_size 4  --load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1  --ngpus 1 --nodes 2 --ema --text_encoder_lr 1e-5 --lr 5e-5
```

Alternatively to run on a single node with 2 gpus
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/refcoco.json --batch_size 4  --load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1  --ema --text_encoder_lr 1e-5 --lr 5e-5
```
