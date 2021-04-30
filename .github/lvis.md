# Long-tailed detection on LVIS
| Data | AP | AP 50 |  AP r | APc | AP f | url | size
|----------|---------|---------|-----------|----------|---------|---------|---------|
| 1%| 16.7 | 25.8 | 11.2  | 14.6  | 19.5  | [model](https://zenodo.org/record/4721981/files/lvis1_checkpoint.pth?download=1) | 3GB
| 10%| 24.2 | 38.0 | 20.9   | 24.9 | 24.3 | [model](https://zenodo.org/record/4721981/files/lvis10_checkpoint.pth?download=1) | 3GB
| 100%| 22.5 | 35.2 | 7.4 |22.7 | 25.0 | [model](https://zenodo.org/record/4721981/files/lvis100_checkpoint.pth?download=1) | 3GB

In this section, we show-case how the vision+language pre-training can be leveraged to perform regular detection, with an emphasis on the performance on the rare categories.

## Data preparation
The config for this dataset can be found in configs/lvis.json and is also shown below:

```json
{
    "combine_datasets": ["modulated_lvis"],
    "combine_datasets_val": ["modulated_lvis"],
    "coco2017_path": "",
    "modulated_lvis_ann_path": "mdetr_annotations/",
    "lvis_subset": 10
}
```
The `lvis_subset` controls the percentage of the full dataset to be used. Valid values are 1, 10 and 100.

* You need the images from [COCO 2017](https://cocodataset.org/#download) and update the "coco2017_path" to the folder containing the images.
* Download our pre-processed annotations that are converted to coco format (all datasets present in the same zip folder for MDETR annotations): [Pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1) and update the "modulated_lvis_ann_path" to this folder with pre-processed annotations. The script to reproduce the annotations is located in scripts/fine-tuning/lvis_coco_format.py


## Evaluating pre-trained models


The evaluation involve querying MDETR for *all* the LVIS classes on each image. As a result, it is pretty slow, and we recommend evaluating on several gpus. Note that the LVIS-v1 val set contains some image from COCO train, hence some of these images may potentially have been seen during pre-training. To avoid all contamination, we report on the subset of the val that doesn't intersect with COCO train, which we call minival. This subset is available [here](https://nyu.box.com/shared/static/2yk9x8az9pnlsy2v8gd95yncwn2q7vj6.zip) (without segmentation).

To eval on a single node with 2 gpus

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env scripts/eval_lvis.py --dataset_config configs/lvis.json --resume https://zenodo.org/record/4721981/files/lvis10_checkpoint.pth --lvis_minival /path/to/lvisminival
```

Alternatively, you can evaluate using submitit

```
python run_with_submitit_eval_lvis.py --dataset_config configs/lvis.json --resume https://zenodo.org/record/4721981/files/lvis10_checkpoint.pth --lvis_minival /path/to/lvisminival  --ngpus 8 --nodes 4
```


## Fine-Tuning

We fine-tuned on 64 gpus. The number of epoch depends on the size of the subset.

To train on a single node with 2 gpus:
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/lvis.json --load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth --ema --epochs 150 --lr_drop 120 --eval_skip 5
```

Alternatively, you can use submitit

```
python run_with_submitit.py --dataset_config configs/lvis.json --load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth --ema --epochs 150 --lr_drop 120 --eval_skip 5 --ngpus 8 --nodes 4
```
