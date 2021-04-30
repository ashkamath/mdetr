# Pre-training

The models are summarized in the following table. Note that the performance reported is "raw", without any fine-tuning. For each dataset, we report the class-agnostic box AP@50, which measures how well the model finds the boxes mentioned in the text. All performances are reported on the respective validation sets of each dataset.
<table>
<thead>
  <tr>
    <th rowspan="2"></th>
    <th rowspan="2">Backbone</th>
    <th>GQA</th>
    <th colspan="2">Flickr</th>
    <th colspan="4">Refcoco</th>
    <th rowspan="2"> Url<br></th>
    <th rowspan="2">Size<br></th>
  </tr>
  <tr>
    <td>AP</td>
    <td>AP</td>
    <td>R@1</td>
    <td>AP</td>
    <td>Refcoco R@1</td>
    <td>Refcoco+ R@1</td>
    <td>Refcocog R@1</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td>R101</td>
    <td>58.9</td>
    <td>75.6</td>
    <td>82.5</td>
    <td>60.3</td>
    <td>72.1</td>
    <td>58.0</td>
    <td>55.7</td>
    <td><a href="https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1"> model</a></td>
    <td>3GB</td>
  </tr>
  <tr>
    <td>2</td>
    <td>ENB3</td>
    <td>59.5</td>
    <td>76.6</td>
    <td>82.9</td>
    <td>57.6</td>
    <td>70.2</td>
    <td>56.7</td>
    <td>53.8</td>
    <td><a href="https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth?download=1">model</a></td>
    <td>2.4GB</td>
  </tr>
  <tr>
    <td>3</td>
    <td>ENB5</td>
    <td>59.9</td>
    <td>76.4</td>
    <td>83.7</td>
    <td>61.8</td>
    <td>73.4</td>
    <td>58.8</td>
    <td>57.1</td>
    <td><a href="https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth?download=1">model</a></td>
    <td>2.7GB</td>
  </tr>
</tbody>
</table>


The config file for pretraining is configs/pretrain.json and looks like:

```json
{
    "combine_datasets": ["flickr", "mixed"],
    "combine_datasets_val": ["flickr", "gqa", "refexp"],
    "coco_path": "",
    "vg_img_path": "",
    "flickr_img_path": "",
    "refexp_ann_path": "mdetr_annotations/",
    "flickr_ann_path": "mdetr_annotations/",
    "gqa_ann_path": "mdetr_annotations/",
    "num_queries": 100,
    "refexp_dataset_name": "all",
    "GT_type": "separate",
    "flickr_dataset_path": ""
}
```

* Download the original Flickr30k image dataset from : [Flickr30K webpage](http://shannon.cs.illinois.edu/DenotationGraph/) and update the `flickr_img_path` to the folder containing the images.
* Download the original Flickr30k entities annotations from: [Flickr30k annotations](https://github.com/BryanPlummer/flickr30k_entities) and update the `flickr_dataset_path` to the folder with annotations.
* Download the gqa images at [GQA images](https://nlp.stanford.edu/data/gqa/images.zip) and update `vg_img_path` to point to the folder containing the images.
* Download COCO images [Coco train2014](http://images.cocodataset.org/zips/train2014.zip). Update the `coco_path` to the folder containing the downloaded images.
* Download our pre-processed annotations that are converted to coco format (all datasets present in the same zip folder for MDETR annotations): [Pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1) and update the `flickr_ann_path`, `gqa_ann_path` and `refexp_ann_path` to this folder with pre-processed annotations.

## Script to run pre-training

### Resnet101

This command will reproduce the training of the resnet 101.
```
python run_with_submitit.py --dataset_config configs/pretrain.json  --ngpus 8 --nodes 4 --ema
```

To run on a single node with 8 gpus

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_config configs/pretrain.json --ema
```


### Timm backbones

We provide an interface to the [Timm Library](https://github.com/rwightman/pytorch-image-models).
Most stride 32 models that support the "features_only" mode should be supported out of the box. That includes Resnet variants as well as EfficientNets. Simply use `--backbone timm_modelname` where the modelname is taken from the list of Timm model [here](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv).

In the paper, we train an efficientnet-b3 with noisy-student pre-training as follows (note that the learning rate of the backbone is slightly different):
```
python run_with_submitit.py --dataset_config configs/pretrain.json  --ngpus 8 --nodes 4 --ema --backbone timm_tf_efficientnet_b3_ns --lr_backbone 5e-5
```

To run on a single node with 8 gpus

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_config configs/pretrain.json --ema --backbone timm_tf_efficientnet_b3_ns --lr_backbone 5e-5
```
