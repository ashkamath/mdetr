# Referring expression segmentation on PhraseCut

| Backbone | M-IoU | Precision @0.5 | Precision @0.7 | Precision @0.9  |  url | size |
|----------|---------|---------|-----------|----------|-----------|-----------|
| Resnet-101| 53.1 | 56.1 | 38.9    | 11.9   | [model](https://zenodo.org/record/4721981/files/phrasecut_resnet101_checkpoint.pth?download=1)   |  1.5GB    |   
| EfficientNet-B3| 53.7| 57.5|  39.9  | 11.9 | [model](https://zenodo.org/record/4721981/files/phrasecut_EB3_checkpoint.pth?download=1)   | 1.2GB  | 


In this section, we showcase that MDETR can be extended to perform segmentation.


## Data preparation
The config for this dataset can be found in configs/phrasecut.json and is also shown below:

```json
{
    "combine_datasets": ["phrasecut"],
    "combine_datasets_val": ["phrasecut"],
    "vg_img_path": "",
    "phrasecut_ann_path": "mdetr_annotations/",
    "phrasecut_orig_ann_path": "VGPhraseCut_v0/"
}
```

* Download the VisualGenome images. For consistency with the other datasets, we use the ones from GQA, available at [GQA images](https://nlp.stanford.edu/data/gqa/images.zip). Update `vg_img_path` to point to the folder containing the images.
* Download our pre-processed annotations that are converted to coco format (all datasets present in the same zip folder for MDETR annotations): [Pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1) and update the `phrasecut_ann_path` to this folder with pre-processed annotations.
* Download the original [PhraseCut annotations](https://people.cs.umass.edu/~chenyun/publication/phrasecut/) and update the `phrasecut_orig_ann_path` with the path to the folder


### Evaluating pre-trained models

To evaluate on the test set, with a single node with 2 gpus, run

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/phrasecut.json --resume https://zenodo.org/record/4721981/files/phrasecut_resnet101_checkpoint.pth --ema --eval --mask_model smallconv --no_contrastive_align_loss --test
```

### Fine-tuning step1: Detector

We first fine-tune the detector for 10 epochs.

For the Resnet101:
```
python run_with_submitit.py --dataset_config configs/phrasecut.json --epochs 10 --lr_drop 11 --load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth --ema --ngpus 8 --nodes 4 --text_encoder_lr 1e-5 --lr 5e-5
```

For the EfficientNet-B3:
```
python run_with_submitit.py --dataset_config configs/phrasecut.json --backbone "timm_tf_efficientnet_b3_ns" --epochs 10 --lr_drop 11  --ema  --ngpus 8 --nodes 4  --text_encoder_lr 1e-5  --lr_backbone 1e-5 --lr 5e-5  --load https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth
```

### Fine-tuning step2: Segmentation

In a second step, we freeze the detector, and train a segmentation head for 35 epochs
For the Resnet101, assuming `detection_r101/checkpoint.pth` is the result of the first step
```
python run_with_submitit.py --dataset_config configs/phrasecut.json --ngpus 8 --nodes 4  --frozen_weights detection_r101/checkpoint.pth --mask_model smallconv --no_aux_loss --epochs 35 --lr_drop 25
```

For the EfficientNet-B3, assuming `detection_enb3/checkpoint.pth` is the result of the first step
```
python run_with_submitit.py --dataset_config configs/phrasecut.json --ngpus 8 --nodes 4  --frozen_weights detection_enb3/checkpoint.pth --mask_model smallconv --no_aux_loss --epochs 35 --lr_drop 25  --backbone "timm_tf_efficientnet_b3_ns"
```
