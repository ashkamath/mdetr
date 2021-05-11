# Synthetic datasets

We show some results on synthetic data from CLEVR. These results use a significantly smaller model, and can be trained on a single GPU, allowing for easier reproducibility and idea validation.

- [CLEVR](#clevr)
- [CLEVR-Humans](#clevr-humans)
- [CLEVR-CoGenT](#clevr-cogent)

## CLEVR
<table>
<thead>
  <tr>
    <th>Overall Accuracy</th>
    <th>Count</th>
    <th>Exist<br></th>
    <th>Compare Number</th>
    <th>Query Attribute</th>
    <th>Compare Attribute</th>
    <th>Url</th>
    <th>Size</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>99.7</td>
    <td>99.3</td>
    <td>99.9</td>
    <td>99.4</td>
    <td>99.9</td>
    <td>99.9</td>
    <td><a href="https://zenodo.org/record/4721981/files/clevr_checkpoint.pth">model</a></td>
    <td>426MB</td>
  </tr>
</tbody>
</table>

The training is in two stages: first we pre-train the detection only on a subset of CLEVR, that we call CLEVR-Medium, which contains sentences that contain at most one reference to each object in the scene. Then we fine-tune on the full dataset, with the additional question answering loss.

The config for this dataset can be found in configs/clevr.json and is also shown below:

```json
{
    "combine_datasets": ["clevr"],
    "combine_datasets_val": ["clevr"],
    "clevr_img_path": "",
    "clevr_ann_path": "/path/to/clevr/clevr_annotations/full",
    "split_qa_heads": 1,
    "do_qa": 1,
    "clevr_variant": "normal"
}
```

* Download the original CLEVR images from : [CLEVR webpage](https://cs.stanford.edu/people/jcjohns/clevr/) and update the `clevr_img_path` to the folder containing the images.
* Download our pre-processed annotations that are converted to coco format: [Pre-processed annotations](https://zenodo.org/record/4721981/files/clevr_annotations.zip?download=1) and update the `clevr_ann_path` to this folder with pre-processed annotations. The scripts to reproduce these annotations are located in scripts/clevr.

### Evaluating pre-trained models

You can run an evaluation of our pre-train model on the val set as follows:
``` 
python main.py --batch_size 64 --dataset_config configs/clevr.json --num_queries 25 --text_encoder_type distilroberta-base --backbone resnet18  --resume https://zenodo.org/record/4721981/files/clevr_checkpoint.pth  --eval
```

Alternatively, you can also dump the model's predictions on the test (or val) set. For that you'll need the questions from the [CLEVR webpage](https://cs.stanford.edu/people/jcjohns/clevr/).
``` 
python scripts/eval_clevr.py --batch_size 64 --dataset_config configs/clevr.json  --resume https://zenodo.org/record/4721981/files/clevr_checkpoint.pth --clevr_eval_path /path/to/CLEVR_v1.0/questions/ --split test
``` 


### Training Step 1: CLEVR-Medium

The config for this step can be found in configs/clevr_pretrain.json.
Adjust `clevr_img_path` and `clevr_ann_path` according to your own path.


The training command for this step is the following (change the output dir if you need):
```
mkdir step1
python main.py --dataset_config configs/clevr_pretrain.json --backbone "resnet18" --num_queries 25 --batch_size 64  --schedule linear_with_warmup --text_encoder_type distilroberta-base --output_dir step1 --epochs 30 --lr_drop 20
```

### Training Step 2: CLEVR-Full


First, adjust the `clevr_ann_path` and `clevr_img_path` as in Step 1

The training command for this step is the following (change the output dir if you need):
```
mkdir step2
python main.py --dataset_config configs/clevr.json --backbone "resnet18" --num_queries 25 --batch_size 64  --schedule linear_with_warmup --text_encoder_type distilroberta-base --output_dir step2 --load step1/checkpoint.pth --epochs 30 --lr_drop 20
```

## CLEVR-Humans
<table>
<thead>
  <tr>
    <th>Before Fine-tuning</th>
    <th>After Fine-tuning</th>
    <th>Url</th>
    <th>Size</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>59.9</td>
    <td>81.7</td>
    <td><a href="https://zenodo.org/record/4721981/files/clevr_humans_checkpoint.pth">model</a></td>
    <td>426MB</td>
  </tr>
</tbody>
</table>

The CLEVR-Humans evaluates tests the generalization capabilities of the model to free-form, human-generated questions. We evaluate performance of the model trained on regular CLEVR, before and after fine-tuning.

The config for this dataset can be found in configs/clevr_humans.json and is also shown below:

```json
{
    "combine_datasets": ["clevr_question"],
    "combine_datasets_val": ["clevr_question"],
    "clevr_img_path": "",
    "clevr_ann_path": "/path/to/CLEVR-Humans/",
    "split_qa_heads": 1,
    "clevr_variant": "humans",
    "no_detection": 1,
    "do_qa": 1
}
```

The images are the same as regular CLEVR, but you need to download the annotations from the [CLEVR Humans webpage](https://cs.stanford.edu/people/jcjohns/iep/). Edit the `clevr_ann_path` accordingly.

### Evaluating pre-trained models

You can run an evaluation of our pre-train model on the val set as follows:
``` 
python main.py --batch_size 64 --dataset_config configs/clevr_humans.json --num_queries 25 --text_encoder_type distilroberta-base --backbone resnet18  --resume https://zenodo.org/record/4721981/files/clevr_humans_checkpoint.pth  --eval
```

Alternatively, you can also dump the model's predictions on the test set. 
``` 
python scripts/eval_clevr.py --batch_size 64 --dataset_config configs/clevr_humans.json  --resume https://zenodo.org/record/4721981/files/clevr_humans_checkpoint.pth --clevr_eval_path /path/to/CLEVR-Humans/ --split test
``` 

### Training

Assuming your CLEVR model is in `step2/checkpoint.pth`, you can fine-tune on CLEVR-Humans as follows:
```
mkdir humans
python main.py --dataset_config configs/clevr_human.json --backbone "resnet18" --num_queries 25 --batch_size 64  --schedule linear_with_warmup --text_encoder_type distilroberta-base --output_dir humans --load step2/checkpoint.pth --epochs 60 --lr_drop 40
```


## CLEVR-CoGenT
<table>
<thead>
  <tr>
    <th>TestA</th>
    <th>TestB</th>
    <th>Url</th>
    <th>Size</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>99.8</td>
    <td>76.7</td>
    <td><a href="https://zenodo.org/record/4721981/files/clevr_cogent_checkpoint.pth">model</a></td>
    <td>426MB</td>
  </tr>
</tbody>
</table>

The CLEVR-CoGenT evaluates tests the compositonal generalization capabilities of the model.

The config for this dataset can be found in configs/clevr_cogent.json and is also shown below:

```json
{
    "combine_datasets": ["clevr"],
    "combine_datasets_val": ["clevr"],
    "clevr_img_path": "",
    "clevr_ann_path": "/path/to/clevr/clevr_annotations/cogent_full",
    "split_qa_heads": 1,
    "do_qa": 1,
    "clevr_variant": "cogent"
}
```

Download the original CLEVR-CoGenT images from : [CLEVR webpage](https://cs.stanford.edu/people/jcjohns/clevr/) and update the `clevr_img_path` to the folder containing the images.

### Evaluating pre-trained models

You can run an evaluation of our pre-train model on the valA set as follows:
``` 
python main.py --batch_size 64 --dataset_config configs/clevr.json --num_queries 25 --text_encoder_type distilroberta-base --backbone resnet18  --resume https://zenodo.org/record/4721981/files/clevr_checkpoint.pth  --eval
```

Alternatively, you can also dump the model's predictions on the testB set. For that you'll need the questions from the [CLEVR webpage](https://cs.stanford.edu/people/jcjohns/clevr/).
``` 
python scripts/eval_clevr.py --batch_size 64 --dataset_config configs/clevr_cogent.json  --resume https://zenodo.org/record/4721981/files/clevr_cogent_checkpoint.pth --clevr_eval_path /path/to/CLEVR_CoGenT_v1.0/questions/ --split testA
``` 

Replace `testA` with `testB` to get the predictions on the other test set.

### Training

The training is similar to regular [CLEVR](#clevr). Follow the instructions there, using respectively configs/clevr_cogent_pretrain.json for step 1 and configs/clevr_cogent.json for step2.
