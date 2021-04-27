# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from timm.models import create_model
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": 0}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation], pretrained=True, norm_layer=FrozenBatchNorm2d
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, **kargs):
        super().__init__(num_groups, num_channels, **kargs)


class GroupNormBackbone(BackboneBase):
    """ResNet backbone with GroupNorm with 32 channels."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        name_map = {
            "resnet50-gn": ("resnet50", "/checkpoint/szagoruyko/imagenet/22014122/checkpoint.pth"),
            "resnet101-gn": ("resnet101", "/checkpoint/szagoruyko/imagenet/22080524/checkpoint.pth"),
        }
        backbone = getattr(torchvision.models, name_map[name][0])(
            replace_stride_with_dilation=[False, False, dilation], pretrained=False, norm_layer=GroupNorm32
        )
        checkpoint = torch.load(name_map[name][1], map_location="cpu")
        state_dict = {k[7:]: p for k, p in checkpoint["model"].items()}
        backbone.load_state_dict(state_dict)
        num_channels = 512 if name_map[name][0] in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


def replace_bn(m, name=""):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if isinstance(target_attr, torch.nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(target_attr.num_features)
            bn = getattr(m, attr_str)
            frozen.weight.data.copy_(bn.weight)
            frozen.bias.data.copy_(bn.bias)
            frozen.running_mean.data.copy_(bn.running_mean)
            frozen.running_var.data.copy_(bn.running_var)
            setattr(m, attr_str, frozen)
    for n, ch in m.named_children():
        replace_bn(ch, n)


class GN_8(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gn = torch.nn.GroupNorm(8, num_channels)

    def forward(self, x):
        return self.gn(x)


class TimmBackbone(nn.Module):
    def __init__(self, name, return_interm_layers, main_layer=-1, group_norm=False):
        super().__init__()
        backbone = create_model(name, pretrained=True, in_chans=3, features_only=True, out_indices=(1, 2, 3, 4))

        with torch.no_grad():
            replace_bn(backbone)
        num_channels = backbone.feature_info.channels()[-1]
        self.body = backbone
        self.num_channels =  num_channels
        self.interm = return_interm_layers
        self.main_layer = main_layer

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        if not self.interm:
            xs = [xs[self.main_layer]]
        out = OrderedDict()
        for i, x in enumerate(xs):
            mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).bool()[0]
            out[f"layer{i}"] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.backbone[: len("timm_")] == "timm_":
        backbone = TimmBackbone(
            args.backbone[len("timm_") :],
            return_interm_layers,
            main_layer=-1,
            group_norm=True,
        )
    elif args.backbone in ("resnet50-gn", "resnet101-gn"):
        backbone = GroupNormBackbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
