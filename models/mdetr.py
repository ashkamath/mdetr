# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

import util.dist as dist
from util import box_ops
from util.metrics import accuracy
from util.misc import NestedTensor, interpolate

from .backbone import build_backbone
from .matcher import build_matcher
from .postprocessors import build_postprocessors
from .segmentation import DETRsegm, dice_loss, sigmoid_focal_loss
from .transformer import build_transformer


class MDETR(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_loss=False,
        contrastive_align_loss=False,
        qa_dataset: Optional[str] = None,
        split_qa_heads=True,
        predict_final=False,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MDETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            contrastive_loss: If true, perform image-text contrastive learning
            contrastive_align_loss: If true, perform box - token contrastive learning
            qa_dataset: If not None, train a QA head for the target dataset (CLEVR or GQA)
            split_qa_heads: If true, use several head for each question type
            predict_final: If true, will predict if a given box is in the actual referred set.
                           Useful for CLEVR-Ref+ only currently.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.isfinal_embed = nn.Linear(hidden_dim, 1) if predict_final else None
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if qa_dataset is not None:
            nb_heads = 6 if qa_dataset == "gqa" else 4
            self.qa_embed = nn.Embedding(nb_heads if split_qa_heads else 1, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.contrastive_loss = contrastive_loss
        if contrastive_loss:
            self.contrastive_projection_image = nn.Linear(hidden_dim, contrastive_hdim, bias=False)
            self.contrastive_projection_text = nn.Linear(
                self.transformer.text_encoder.config.hidden_size, contrastive_hdim, bias=False
            )
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)

        self.qa_dataset = qa_dataset
        self.split_qa_heads = split_qa_heads
        if qa_dataset is not None:
            if split_qa_heads:
                self.answer_type_head = nn.Linear(hidden_dim, 5)
                # TODO: make this more general
                if qa_dataset == "gqa":
                    self.answer_rel_head = nn.Linear(hidden_dim, 1594)
                    self.answer_obj_head = nn.Linear(hidden_dim, 3)
                    self.answer_global_head = nn.Linear(hidden_dim, 111)
                    self.answer_attr_head = nn.Linear(hidden_dim, 403)
                    self.answer_cat_head = nn.Linear(hidden_dim, 678)
                elif qa_dataset == "clevr":
                    self.answer_type_head = nn.Linear(hidden_dim, 3)
                    self.answer_binary_head = nn.Linear(hidden_dim, 1)
                    self.answer_attr_head = nn.Linear(hidden_dim, 15)
                    self.answer_reg_head = MLP(hidden_dim, hidden_dim, 20, 3)
                else:
                    assert False, f"Invalid qa dataset {qa_dataset}"
            else:
                # TODO: make this more general
                assert qa_dataset == "gqa", "Clevr QA is not supported with unified head"
                self.answer_head = nn.Linear(hidden_dim, 1853)

    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            query_embed = self.query_embed.weight
            if self.qa_dataset is not None:
                query_embed = torch.cat([query_embed, self.qa_embed.weight], 0)
            memory_cache = self.transformer(
                self.input_proj(src),
                mask,
                query_embed,
                pos[-1],
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
            )

            if self.contrastive_loss:
                memory_cache["text_pooled_op"] = self.contrastive_projection_text(memory_cache["text_pooled_op"])
                memory_cache["img_pooled_op"] = self.contrastive_projection_image(memory_cache["img_pooled_op"])

            return memory_cache

        else:
            assert memory_cache is not None
            hs = self.transformer(
                mask=memory_cache["mask"],
                query_embed=memory_cache["query_embed"],
                pos_embed=memory_cache["pos_embed"],
                encode_and_save=False,
                text_memory=memory_cache["text_memory_resized"],
                img_memory=memory_cache["img_memory"],
                text_attention_mask=memory_cache["text_attention_mask"],
            )
            out = {}
            if self.qa_dataset is not None:
                if self.split_qa_heads:
                    if self.qa_dataset == "gqa":
                        answer_embeds = hs[0, :, -6:]
                        hs = hs[:, :, :-6]
                        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                        out["pred_answer_obj"] = self.answer_obj_head(answer_embeds[:, 1])
                        out["pred_answer_rel"] = self.answer_rel_head(answer_embeds[:, 2])
                        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                        out["pred_answer_cat"] = self.answer_cat_head(answer_embeds[:, 4])
                        out["pred_answer_global"] = self.answer_global_head(answer_embeds[:, 5])
                    elif self.qa_dataset == "clevr":
                        answer_embeds = hs[0, :, -4:]
                        hs = hs[:, :, :-4]
                        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                        out["pred_answer_binary"] = self.answer_binary_head(answer_embeds[:, 1]).squeeze(-1)
                        out["pred_answer_reg"] = self.answer_reg_head(answer_embeds[:, 2])
                        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                    else:
                        assert False, f"Invalid qa dataset {self.qa_dataset}"

                else:
                    answer_embeds = hs[0, :, -1]
                    hs = hs[:, :, :-1]
                    out["pred_answer"] = self.answer_head(answer_embeds)

            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update(
                {
                    "pred_logits": outputs_class[-1],
                    "pred_boxes": outputs_coord[-1],
                }
            )
            outputs_isfinal = None
            if self.isfinal_embed is not None:
                outputs_isfinal = self.isfinal_embed(hs)
                out["pred_isfinal"] = outputs_isfinal[-1]
            proj_queries, proj_tokens = None, None
            if self.contrastive_align_loss:
                proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
                proj_tokens = F.normalize(
                    self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1
                )
                out.update(
                    {
                        "proj_queries": proj_queries[-1],
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                    }
                )
            if self.aux_loss:
                if self.contrastive_align_loss:
                    assert proj_tokens is not None and proj_queries is not None
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "proj_queries": c,
                            "proj_tokens": proj_tokens,
                            "tokenized": memory_cache["tokenized"],
                        }
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
                    ]
                else:
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                        }
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                    ]
                if outputs_isfinal is not None:
                    assert len(outputs_isfinal[:-1]) == len(out["aux_outputs"])
                    for i in range(len(outputs_isfinal[:-1])):
                        out["aux_outputs"][i]["pred_isfinal"] = outputs_isfinal[i]
            return out


class ContrastiveCriterion(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_text, pooled_image):

        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

        logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(pooled_image.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2.0
        return loss


class QACriterionGQA(nn.Module):
    def __init__(self, split_qa_heads):
        super().__init__()
        self.split_qa_heads = split_qa_heads

    def forward(self, output, answers):
        loss = {}
        if not self.split_qa_heads:
            loss["loss_answer_total"] = F.cross_entropy(output["pred_answer"], answers["answer"], reduction="mean")
            attr_total = (output["pred_answer"].argmax(-1)) == answers["answer"]
            loss["accuracy_answer_total"] = attr_total.float().mean()
            return loss

        device = output["pred_answer_type"].device
        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])

        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_obj = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_rel = answers["answer_type"] == 2
        is_global = answers["answer_type"] == 3
        is_cat = answers["answer_type"] == 4

        ## OBJ type
        obj_norm = is_obj.sum() if is_obj.any() else 1.0
        loss["loss_answer_obj"] = (
            F.cross_entropy(output["pred_answer_obj"], answers["answer_obj"], reduction="none")
            .masked_fill(~is_obj, 0)
            .sum()
            / obj_norm
        )
        obj_acc = (output["pred_answer_obj"].argmax(-1)) == answers["answer_obj"]
        loss["accuracy_answer_obj"] = (
            obj_acc[is_obj].sum() / is_obj.sum() if is_obj.any() else torch.as_tensor(1.0, device=device)
        )

        ## ATTR type
        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(output["pred_answer_attr"], answers["answer_attr"], reduction="none")
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )
        attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
        loss["accuracy_answer_attr"] = (
            attr_acc[is_attr].sum() / is_attr.sum() if is_attr.any() else torch.as_tensor(1.0, device=device)
        )

        ## REL type
        rel_norm = is_rel.sum() if is_rel.any() else 1.0
        loss["loss_answer_rel"] = (
            F.cross_entropy(output["pred_answer_rel"], answers["answer_rel"], reduction="none")
            .masked_fill(~is_rel, 0)
            .sum()
            / rel_norm
        )
        rel_acc = (output["pred_answer_rel"].argmax(-1)) == answers["answer_rel"]
        loss["accuracy_answer_rel"] = (
            rel_acc[is_rel].sum() / is_rel.sum() if is_rel.any() else torch.as_tensor(1.0, device=device)
        )

        ## GLOBAL type
        global_norm = is_global.sum() if is_global.any() else 1.0
        loss["loss_answer_global"] = (
            F.cross_entropy(output["pred_answer_global"], answers["answer_global"], reduction="none")
            .masked_fill(~is_global, 0)
            .sum()
            / global_norm
        )
        global_acc = (output["pred_answer_global"].argmax(-1)) == answers["answer_global"]
        loss["accuracy_answer_global"] = (
            global_acc[is_global].sum() / is_global.sum() if is_global.any() else torch.as_tensor(1.0, device=device)
        )

        ## CAT type
        cat_norm = is_cat.sum() if is_cat.any() else 1.0
        loss["loss_answer_cat"] = (
            F.cross_entropy(output["pred_answer_cat"], answers["answer_cat"], reduction="none")
            .masked_fill(~is_cat, 0)
            .sum()
            / cat_norm
        )
        cat_acc = (output["pred_answer_cat"].argmax(-1)) == answers["answer_cat"]
        loss["accuracy_answer_cat"] = (
            cat_acc[is_cat].sum() / is_cat.sum() if is_cat.any() else torch.as_tensor(1.0, device=device)
        )

        loss["accuracy_answer_total"] = (
            type_acc
            * (is_obj * obj_acc + is_rel * rel_acc + is_attr * attr_acc + is_global * global_acc + is_cat * cat_acc)
        ).sum() / type_acc.numel()

        return loss


class QACriterionClevr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, answers):
        loss = {}
        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])

        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_binary = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_reg = answers["answer_type"] == 2

        binary_norm = is_binary.sum() if is_binary.any() else 1.0
        loss["loss_answer_binary"] = (
            F.binary_cross_entropy_with_logits(output["pred_answer_binary"], answers["answer_binary"], reduction="none")
            .masked_fill(~is_binary, 0)
            .sum()
            / binary_norm
        )
        bin_acc = (output["pred_answer_binary"].sigmoid() > 0.5) == answers["answer_binary"]
        loss["accuracy_answer_binary"] = (
            bin_acc[is_binary].sum() / is_binary.sum() if is_binary.any() else torch.as_tensor(1.0)
        )

        reg_norm = is_reg.sum() if is_reg.any() else 1.0
        loss["loss_answer_reg"] = (
            F.cross_entropy(output["pred_answer_reg"], answers["answer_reg"], reduction="none")
            .masked_fill(~is_reg, 0)
            .sum()
            / reg_norm
        )
        reg_acc = (output["pred_answer_reg"].argmax(-1)) == answers["answer_reg"]
        loss["accuracy_answer_reg"] = reg_acc[is_reg].sum() / is_reg.sum() if is_reg.any() else torch.as_tensor(1.0)

        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(output["pred_answer_attr"], answers["answer_attr"], reduction="none")
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )
        attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
        loss["accuracy_answer_attr"] = (
            attr_acc[is_attr].sum() / is_attr.sum() if is_attr.any() else torch.as_tensor(1.0)
        )

        loss["accuracy_answer_total"] = (
            type_acc * (is_binary * bin_acc + is_reg * reg_acc + is_attr * attr_acc)
        ).sum() / type_acc.numel()

        return loss


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, eos_coef, losses, temperature):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_isfinal(self, outputs, targets, positive_map, indices, num_boxes):
        """This loss is used in some referring expression dataset (specifically Clevr-REF+)
        It trains the model to predict which boxes are being referred to (ie are "final")
        Eg if the caption is "the cube next to the cylinder", MDETR will detect both the cube and the cylinder.
        However, the cylinder is an intermediate reasoning step, only the cube is being referred here.
        """
        idx = self._get_src_permutation_idx(indices)
        src_isfinal = outputs["pred_isfinal"][idx].squeeze(-1)
        target_isfinal = torch.cat([t["isfinal"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_isfinal = F.binary_cross_entropy_with_logits(src_isfinal, target_isfinal, reduction="none")

        losses = {}
        losses["loss_isfinal"] = loss_isfinal.sum() / num_boxes
        acc = (src_isfinal.sigmoid() > 0.5) == (target_isfinal > 0.5)
        if acc.numel() == 0:
            acc = acc.sum()
        else:
            acc = acc.float().mean()
        losses["accuracy_isfinal"] = acc

        return losses

    def loss_labels(self, outputs, targets, positive_map, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        logits = outputs["pred_logits"].log_softmax(-1)  # BS x (num_queries) x (num_tokens)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1)

        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1

        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_contrastive_align(self, outputs, targets, positive_map, indices, num_boxes):
        bs = outputs["proj_queries"].shape[0]
        tokenized = outputs["tokenized"]

        normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        logits = (
            torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.temperature
        )  # BS x (num_queries) x (num_tokens)

        # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
        # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
        positive_map = torch.zeros(logits.shape, dtype=torch.bool)
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
            if "tokens_positive" in tgt:
                cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt]
            else:
                cur_tokens = [tgt["tokens"][j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None
                    positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)

        positive_map = positive_map.to(logits.device)
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return {"loss_contrastive_align": tot_loss / num_boxes}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        ## Count the number of predictions that are NOT "no-object" (which is the last class)
        # normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        # normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        # logits = torch.matmul(
        #    normalized_img_emb, normalized_text_emb.transpose(-1, -2)
        # )  # BS x (num_queries) x (num_tokens)
        # card_pred = (logits[:, :, 0] > 0.5).sum(1)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "isfinal": self.loss_isfinal,
            "contrastive_align": self.loss_contrastive_align,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, positive_map):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, positive_map)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, positive_map, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, positive_map)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, positive_map, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 255
    device = torch.device(args.device)

    assert not args.masks or args.mask_model != "none"

    qa_dataset = None
    if args.do_qa:
        assert not (
            ("clevr" in args.combine_datasets or "clevr_question" in args.combine_datasets)
            and "gqa" in args.combine_datasets
        ), "training GQA and CLEVR simultaneously is not supported"
        assert (
            "clevr_question" in args.combine_datasets
            or "clevr" in args.combine_datasets
            or "gqa" in args.combine_datasets
        ), "Question answering require either gqa or clevr dataset"
        qa_dataset = "gqa" if "gqa" in args.combine_datasets else "clevr"

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = MDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_loss=args.contrastive_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        qa_dataset=qa_dataset,
        split_qa_heads=args.split_qa_heads,
        predict_final=args.predict_final,
    )
    if args.mask_model != "none":
        model = DETRsegm(
            model,
            mask_head=args.mask_model,
            freeze_detr=(args.frozen_weights is not None),
        )
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.ce_loss_coef, "loss_bbox": args.bbox_loss_coef}
    if args.contrastive_loss:
        weight_dict["contrastive_loss"] = args.contrastive_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.predict_final:
        weight_dict["loss_isfinal"] = 1

    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.do_qa:
        if args.split_qa_heads:
            weight_dict["loss_answer_type"] = 1 * args.qa_loss_coef
            if qa_dataset == "gqa":
                weight_dict["loss_answer_cat"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_attr"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_rel"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_obj"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_global"] = 1 * args.qa_loss_coef
            else:
                weight_dict["loss_answer_binary"] = 1
                weight_dict["loss_answer_attr"] = 1
                weight_dict["loss_answer_reg"] = 1

        else:
            weight_dict["loss_answer_total"] = 1 * args.qa_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    if args.predict_final:
        losses += ["isfinal"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]

    criterion = None
    if not args.no_detection:
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            eos_coef=args.eos_coef,
            losses=losses,
            temperature=args.temperature_NCE,
        )
        criterion.to(device)

    if args.contrastive_loss:
        contrastive_criterion = ContrastiveCriterion(temperature=args.temperature_NCE)
        contrastive_criterion.to(device)
    else:
        contrastive_criterion = None

    if args.do_qa:
        if qa_dataset == "gqa":
            qa_criterion = QACriterionGQA(split_qa_heads=args.split_qa_heads)
        elif qa_dataset == "clevr":
            qa_criterion = QACriterionClevr()
        else:
            assert False, f"Invalid qa dataset {qa_dataset}"
        qa_criterion.to(device)
    else:
        qa_criterion = None
    return model, criterion, contrastive_criterion, qa_criterion, weight_dict
