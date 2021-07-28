# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import torch

from models.backbone import Backbone, Joiner, TimmBackbone
from models.mdetr import MDETR
from models.position_encoding import PositionEmbeddingSine
from models.postprocessors import PostProcess, PostProcessSegm
from models.segmentation import DETRsegm
from models.transformer import Transformer

dependencies = ["torch", "torchvision"]


def _make_backbone(backbone_name: str, mask: bool = False):
    if backbone_name[: len("timm_")] == "timm_":
        backbone = TimmBackbone(
            backbone_name[len("timm_") :],
            mask,
            main_layer=-1,
            group_norm=True,
        )
    else:
        backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=False)

    hidden_dim = 256
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    return backbone_with_pos_enc


def _make_detr(
    backbone_name: str,
    num_queries=100,
    mask=False,
    qa_dataset=None,
    predict_final=False,
    text_encoder="roberta-base",
    contrastive_align_loss=True,
):
    hidden_dim = 256
    backbone = _make_backbone(backbone_name, mask)
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True, text_encoder_type=text_encoder)
    detr = MDETR(
        backbone,
        transformer,
        num_classes=255,
        num_queries=num_queries,
        qa_dataset=qa_dataset,
        predict_final=predict_final,
        contrastive_align_loss=contrastive_align_loss,
        contrastive_hdim=64,
    )
    if mask:
        return DETRsegm(detr)
    return detr


def mdetr_resnet101(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Pretrained on our combined aligned dataset of 1.3 million images paired with text.
    """

    model = _make_detr("resnet101")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB3(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Pretrained on our combined aligned dataset of 1.3 million images paired with text.
    """

    model = _make_detr("timm_tf_efficientnet_b3_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB5(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB5 with 6 encoder and 6 decoder layers.
    Pretrained on our combined aligned dataset of 1.3 million images paired with text.
    """

    model = _make_detr("timm_tf_efficientnet_b5_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_clevr(pretrained=False, return_postprocessor=False):
    """
    MDETR R18 with 6 encoder and 6 decoder layers.
    Trained on CLEVR, achieves 99.7% accuracy
    """

    model = _make_detr("resnet18", num_queries=25, qa_dataset="clevr", text_encoder="distilroberta-base")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/clevr_checkpoint.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_clevr_humans(pretrained=False, return_postprocessor=False):
    """
    MDETR R18 with 6 encoder and 6 decoder layers.
    Trained on CLEVR-Humans, achieves 81.7% accuracy
    """

    model = _make_detr("resnet18", num_queries=25, qa_dataset="clevr", text_encoder="distilroberta-base")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/clevr_humans_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_resnet101_gqa(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on GQA, achieves 61.99 on test-std
    """

    model = _make_detr("resnet101", qa_dataset="gqa", contrastive_align_loss=False)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/gqa_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB5_gqa(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB5 with 6 encoder and 6 decoder layers.
    Trained on GQA, achieves 61.99 on test-std
    """

    model = _make_detr("timm_tf_efficientnet_b5_ns", qa_dataset="gqa", contrastive_align_loss=False)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/gqa_EB5_checkpoint.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_resnet101_phrasecut(pretrained=False, threshold=0.5, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on Phrasecut, achieves 53.1 M-IoU on the test set
    """
    model = _make_detr("resnet101", mask=True, contrastive_align_loss=False)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/phrasecut_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, [PostProcess(), PostProcessSegm(threshold=threshold)]
    return model


def mdetr_efficientnetB3_phrasecut(pretrained=False, threshold=0.5, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Trained on Phrasecut, achieves 53.7 M-IoU on the test set
    """
    model = _make_detr("timm_tf_efficientnet_b3_ns", mask=True, contrastive_align_loss=False)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/phrasecut_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, [PostProcess(), PostProcessSegm(threshold=threshold)]
    return model


def mdetr_resnet101_refcoco(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on refcoco, achieves 86.75 val accuracy
    """
    model = _make_detr("resnet101")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB3_refcoco(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Trained on refcoco, achieves 86.75 val accuracy
    """
    model = _make_detr("timm_tf_efficientnet_b3_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcoco_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_resnet101_refcocoplus(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on refcoco+, achieves 79.52 val accuracy
    """
    model = _make_detr("resnet101")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcoco%2B_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB3_refcocoplus(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Trained on refcoco+, achieves 81.13 val accuracy
    """
    model = _make_detr("timm_tf_efficientnet_b3_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcoco%2B_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_resnet101_refcocog(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on refcocog, achieves 81.64 val accuracy
    """
    model = _make_detr("resnet101")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcocog_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB3_refcocog(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Trained on refcocog, achieves 83.35 val accuracy
    """
    model = _make_detr("timm_tf_efficientnet_b3_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model
