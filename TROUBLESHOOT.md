# Troubleshoot

1. Uable to install `pycocotools` due to missing `numpy` in new conda environment

    ```bash
    pip install numpy
    pip install -r requirements.txt
    ```

2. Download and extract data files

    ```bash
    wget https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz
    mkdir mdetr_annotations
    tar -xf mdetr_annotations.tar.gz -C mdetr_annotations
    ```

    ```bash
    wget http://images.cocodataset.org/zips/train2014.zip
    mkdir coco
    unzip http://images.cocodataset.org/zips/train2014.zip -d coco
    ```

3. `torch.set_deterministic` not found when `torch==1.10.0` installed

    Solution: Fixed pytorch and torchvision version in `requirements.txt`

## Run evaluation on RefCOCO validation

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main.py --dataset_config configs/refcoco.json  --batch_size 4  --resume https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth --eval
```

```
Dataset: refcoco - Precision @ 1, 5, 10: [0.863946834040982, 0.958094886468525, 0.969724940003692]
```

## Run evaluation on RefCOCO testA

```bash
UBLAS_WORKSPACE_CONFIG=:4096:8 python main.py --dataset_config configs/refcoco.json  --batch_size 4  --resume https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth --eval --test --test_type testA
```

```
Dataset: refcoco - Precision @ 1, 5, 10: [0.8951741205586, 0.9743680395969595, 0.9814389252253845]
```

## Run evaluation on RefCOCO testB

```bash
UBLAS_WORKSPACE_CONFIG=:4096:8 python main.py --dataset_config configs/refcoco.json  --batch_size 4  --resume https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth --eval --test --test_type testB
```

```
Dataset: refcoco - Precision @ 1, 5, 10: [0.8151128557409225, 0.937585868498528, 0.9540726202158979] 
```
