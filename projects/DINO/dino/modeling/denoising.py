from dataclasses import dataclass

import torch
import torch.nn as nn
from overdrive.layers import inverse_sigmoid
from overdrive.structures import BoxMode
from overdrive.structures import Instances

__all__ = [
    "CDNMetadata",
    "CDNQueryGenerator",
]


@dataclass(frozen=True)
class CDNMetadata:
    """
    DN prefix size and the positive/negative slots assigned to each image's GT.
    """

    pad_size: int
    num_groups: int
    positive_indices: list[torch.Tensor]
    negative_indices: list[torch.Tensor]
    target_indices: list[torch.Tensor]


class CDNQueryGenerator(nn.Module):
    """
    Contrastive denoising.

    ``num_dn_queries`` is the dynamic grouping budget, not the final prefix size:
    there are ``max(1, budget // max_gt)`` groups of ``2 * max_gt`` slots.
    A non-positive budget disables denoising and returns four ``None`` values.
    Boxes enter as absolute XYXY Instances and leave as inverse-sigmoid CXCYWH.
    """

    def __init__(self, *, num_queries: int, num_classes: int, embed_dim: int = 256, num_dn_queries: int = 100, label_noise_scale: float = 0.5, box_noise_scale: float = 1.0):
        super().__init__()
        if min(num_queries, num_classes, embed_dim) <= 0:
            raise ValueError("num_queries, num_classes and embed_dim must be positive")
        if not 0 <= label_noise_scale <= 1 or box_noise_scale < 0:
            raise ValueError("invalid contrastive denoising noise scales")
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_dn_queries = num_dn_queries
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, gt_instances: list[Instances]):
        if self.num_dn_queries <= 0:
            return None, None, None, None

        if not gt_instances:
            raise ValueError("ground truth must contain at least one image")

        max_gt = max(len(gt) for gt in gt_instances)
        num_groups = max(1, self.num_dn_queries // max_gt) if max_gt else 1
        pad_size = 2 * max_gt * num_groups
        label_queries = self.label_embedding.weight.new_zeros((len(gt_instances), pad_size, self.embed_dim))
        box_queries = torch.zeros((len(gt_instances), pad_size, 4), device=label_queries.device, dtype=torch.float32)
        positive_indices, negative_indices, target_indices = [], [], []
        for image_idx, gt in enumerate(gt_instances):
            labels = gt.gt_classes
            n = len(gt)
            targets = torch.arange(n, device=labels.device).repeat(num_groups)
            offsets = torch.arange(num_groups, device=labels.device).repeat_interleave(n) * (2 * max_gt)
            positive = targets + offsets
            negative = positive + max_gt
            positive_indices.append(positive)
            negative_indices.append(negative)
            target_indices.append(targets)
            if not n:
                continue

            # Store the positives followed by negatives locally, then scatter into paired groups.
            noisy_labels = labels[targets].repeat(2)
            replace = torch.rand(noisy_labels.shape, device=labels.device) < self.label_noise_scale * 0.5
            noisy_labels = torch.where(replace, torch.randint(self.num_classes, noisy_labels.shape, device=labels.device), noisy_labels)
            h, w = gt.image_size
            boxes = gt.gt_boxes.tensor.float() / box_queries.new_tensor([w, h, w, h])
            noisy_boxes = boxes[targets].repeat(2, 1)
            whwh = (noisy_boxes[:, 2:] - noisy_boxes[:, :2]).repeat(1, 2)
            magnitude = torch.rand_like(noisy_boxes)
            magnitude[len(targets) :] += 1.0
            sign = torch.randint_like(noisy_boxes, 0, 2) * 2 - 1
            noisy_boxes = (noisy_boxes + sign * magnitude * whwh * self.box_noise_scale / 2).clamp(0, 1)
            noisy_boxes = BoxMode.convert(noisy_boxes, BoxMode.XYXY_REL, BoxMode.CXCYWH_REL)
            slots = torch.cat((positive, negative))
            label_queries[image_idx, slots] = self.label_embedding(noisy_labels)
            box_queries[image_idx, slots] = inverse_sigmoid(noisy_boxes, eps=1e-3)

        attention_mask = torch.zeros((pad_size + self.num_queries, pad_size + self.num_queries), dtype=torch.bool, device=label_queries.device)
        attention_mask[pad_size:, :pad_size] = True
        for group in range(num_groups):
            start, end = group * 2 * max_gt, (group + 1) * 2 * max_gt
            attention_mask[start:end, :start] = True
            attention_mask[start:end, end:pad_size] = True
        metadata = CDNMetadata(pad_size, num_groups, positive_indices, negative_indices, target_indices)
        return label_queries, box_queries, attention_mask, metadata
