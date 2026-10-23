from dataclasses import dataclass

import torch
import torch.nn as nn
from overdrive.layers import inverse_sigmoid

__all__ = [
    "DNMetadata",
    "DNQueryGenerator",
]


@dataclass(frozen=True)
class DNMetadata:
    pad_size: int
    num_groups: int
    output_indices: list[torch.Tensor]
    target_indices: list[torch.Tensor]


class DNQueryGenerator(nn.Module):
    """
    Build noisy ground-truth queries and the decoder self-attention mask.
    """

    def __init__(
        self,
        *,
        num_queries: int,
        num_classes: int,
        embed_dim: int,
        denoising_groups: int = 5,
        label_noise_prob: float = 0.2,
        box_noise_scale: float = 0.4,
        with_indicator: bool = True,
    ):
        super().__init__()

        if num_queries <= 0 or num_classes <= 0 or embed_dim <= int(with_indicator):
            raise ValueError("num_queries, num_classes and embed_dim must be positive")
        if denoising_groups <= 0:
            raise ValueError("denoising_groups must be positive")
        if not 0 <= label_noise_prob <= 1 or box_noise_scale < 0:
            raise ValueError("invalid denoising noise parameters")

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.with_indicator = with_indicator
        self.label_encoder = nn.Embedding(num_classes + 1, embed_dim - int(with_indicator))

    def matching_label_queries(self, batch_size: int):
        """
        Use the learned no-object label for ordinary matching queries.
        """
        background = self.label_encoder.weight[self.num_classes]
        if self.with_indicator:
            background = torch.cat((background, background.new_zeros(1)))
        return background.view(1, 1, -1).expand(batch_size, self.num_queries, -1)

    def forward(self, gt_labels_list: list[torch.Tensor], gt_boxes_list: list[torch.Tensor]):
        if not gt_labels_list or len(gt_labels_list) != len(gt_boxes_list):
            raise ValueError("ground-truth labels and boxes must have the same nonzero batch size")

        batch_size = len(gt_labels_list)
        max_gt = max(len(labels) for labels in gt_labels_list)
        pad_size = max_gt * self.denoising_groups
        label_queries = self.label_encoder.weight.new_zeros((batch_size, pad_size, self.embed_dim))
        box_queries = gt_boxes_list[0].new_zeros((batch_size, pad_size, 4))
        output_indices = []
        target_indices = []

        for image_idx, (labels, boxes) in enumerate(zip(gt_labels_list, gt_boxes_list, strict=True)):
            num_gt = len(labels)
            if boxes.shape != (num_gt, 4):
                raise ValueError("ground-truth boxes must have shape (num_gt, 4)")

            target_idx = torch.arange(num_gt, device=labels.device).repeat(self.denoising_groups)
            group_offsets = torch.arange(self.denoising_groups, device=labels.device).repeat_interleave(num_gt) * max_gt
            output_idx = target_idx + group_offsets
            output_indices.append(output_idx)
            target_indices.append(target_idx)
            if num_gt == 0:
                continue

            noised_labels = labels.repeat(self.denoising_groups)
            if self.label_noise_prob > 0:
                random_labels = torch.randint(self.num_classes + 1, noised_labels.shape, device=labels.device)
                replace = torch.rand(noised_labels.shape, device=labels.device) < self.label_noise_prob
                noised_labels = torch.where(replace, random_labels, noised_labels)

            noised_boxes = boxes.repeat(self.denoising_groups, 1)
            if self.box_noise_scale > 0:
                offsets = torch.cat((noised_boxes[:, 2:] * 0.5, noised_boxes[:, 2:]), dim=-1)
                noised_boxes = noised_boxes + (torch.rand_like(noised_boxes) * 2 - 1) * offsets * self.box_noise_scale
                noised_boxes = noised_boxes.clamp(0, 1)

            embeddings = self.label_encoder(noised_labels)
            if self.with_indicator:
                embeddings = torch.cat((embeddings, embeddings.new_ones((len(embeddings), 1))), dim=-1)
            label_queries[image_idx, output_idx] = embeddings
            box_queries[image_idx, output_idx] = inverse_sigmoid(noised_boxes)

        attention_mask = None
        if pad_size:
            attention_mask = torch.zeros((pad_size + self.num_queries, pad_size + self.num_queries), dtype=torch.bool, device=label_queries.device)
            attention_mask[pad_size:, :pad_size] = True
            for group in range(self.denoising_groups):
                start = group * max_gt
                end = start + max_gt
                attention_mask[start:end, :start] = True
                attention_mask[start:end, end:pad_size] = True

        metadata = DNMetadata(pad_size, self.denoising_groups, output_indices, target_indices)
        return label_queries, box_queries, attention_mask, metadata
