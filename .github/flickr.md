# Flickr30k

### AnyBox protocol
| Backbone | Pre-training Image Data | Val R@1 | Val R@5 | Val R@10 | Test R@1 | Test  R@5 | Test  R@10 | url | size |
|----------|---------|---------|-----------|----------|-----------|-----------|-----|------|---|
| Resnet-101| COCO+VG+Flickr | 82.5   |  92.9   |   94.9  |   83.4  |   93.5  |   95.3    | [model](https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1)    | 3GB      | 
| EfficientNet-B3| COCO+VG+Flickr | 82.9   | 93.2    | 95.2    |  84.0  | 93.8    |  95.6    | [model](https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth?download=1)    |  2.4GB     |
| EfficientNet-B5| COCO+VG+Flickr |83.6   | 93.4    | 95.1   |  84.3   | 93.9    |  95.8     | [model](https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth?download=1)    |  2.7GB     |

 ### MergedBox protocol
 | Backbone | Pre-training Image Data | Val R@1 | Val R@5 | Val R@10 | Test R@1 | Test  R@5 | Test  R@10 | url | size |
|----------|---------|---------|-----------|----------|-----------|-----------|-----|------|---|
| Resnet-101| COCO+VG+Flickr | 82.3   |  91.8   |   93.7  |   83.8  |   92.7  |   94.4    | [model](https://zenodo.org/record/4721981/files/flickr_merged_resnet101_checkpoint.pth?download=1)    |  3GB     | 



### Data preparation
The config for this dataset can be found in configs/flickr.json and is also shown below:

```json
{
  "combine_datasets": ["flickr"],
  "combine_datasets_val": ["flickr"],
  "GT_type" : "separate",
  "flickr_img_path" : "",
  "flickr_dataset_path" : "" ,
  "flickr_ann_path" : "mdetr_annotations/"
}
```

* Download the original Flickr30k image dataset from : [Flickr30K webpage](http://shannon.cs.illinois.edu/DenotationGraph/) and update the `flickr_img_path` to the folder containing the images.
* Download the original Flickr30k entities annotations from: [Flickr30k annotations](https://github.com/BryanPlummer/flickr30k_entities) and update the `flickr_dataset_path` to the folder with annotations.
* Download our pre-processed annotations that are converted to coco format (all datasets present in the same zip folder for MDETR annotations): [Pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1) and update the `flickr_ann_path` to this folder with pre-processed annotations.



### Script to reproduce results

Model weights (can also be loaded directly from url): 
1. [pretrained_resnet101_checkpoint.pth](https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1)
2. [flickr_merged_resnet101_checkpoint.pth](https://zenodo.org/record/4721981/files/flickr_merged_resnet101_checkpoint.pth?download=1)
3. [pretrained_EB3_checkpoint.pth](https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth?download=1)
4. [pretrained_EB5_checkpoint.pth](https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth?download=1)

#### For results using the AnyBox protocol, the pre-trained models are directly evaluated on the val/test set. 

The script to run the evaluation for the resnet-101 backbone pre-trained model is :
This command will run the evaluation on val. For test results, pass --test

MDEDTR-Resnet101:

```
python run_with_submitit.py --dataset_config configs/flickr.json  --resume https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth  --ngpus 1 --nodes 2  --ema  --eval 
```

To run on a single node with 2 gpus

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/flickr.json --resume https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth --ema --eval
```



MDETR-EB3:
```
python run_with_submitit.py --backbone "timm_tf_efficientnet_b3_ns" --dataset_config configs/flickr.json --resume https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth  --ngpus 1 --nodes 2 --ema  --eval 
```

To run on a single node with 2 gpus

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/flickr.json --backbone timm_tf_efficientnet_b3_ns --resume https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth --ema --eval
```



MDETR-EB5:
```
python run_with_submitit.py --backbone "timm_tf_efficientnet_b5_ns" --dataset_config configs/flickr.json --resume https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth  --ngpus 1 --nodes 2  --ema  --eval 
```

To run on a single node with 2 gpus
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/flickr.json --backbone timm_tf_efficientnet_b5_ns --resume https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth --ema --eval
```


#### For the MergedBox protocol, we provide the model fine-tuned on the merged ground truth. 

**Change the "GT_type" option in configs/flickr.json to "merged", and then run**:

```
python run_with_submitit.py --dataset_config configs/flickr.json --resume https://zenodo.org/record/4721981/files/flickr_merged_resnet101_checkpoint.pth  --ngpus 1 --nodes 2 --ema  --eval 
```

Similarly to the above, pass --test for test set evaluation.

To run on a single node with 2 gpus
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/flickr.json --resume https://zenodo.org/record/4721981/files/flickr_merged_resnet101_checkpoint.pth --ema --eval
```



