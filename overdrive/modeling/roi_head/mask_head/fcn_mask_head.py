import torch
import torch.nn as nn

from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import cat
from overdrive.layers import get_activation
from overdrive.layers import weight_init
from overdrive.structures import Instances
from overdrive.utils.events import get_event_storage

__all__ = ["FCNMaskHead"]


class ConvTranspose2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        activation = kwargs.pop("activation", "")

        self.deconv = nn.ConvTranspose2d(*args, **kwargs)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.deconv(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class FCNMaskHead(nn.Module):
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        conv_dims: list[int],
        norm: str | dict = "",
        activation: str | dict = "ReLU",
        loss_mask: nn.Module,
    ):
        super().__init__()

        if len(conv_dims) == 0:
            raise ValueError("conv_dims have to be non-empty!")

        self.layers = []
        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1], start=1):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                norm=norm,
                activation=activation,
            )
            weight_init.c2_msra_fill(conv)
            self.add_module(f"mask_fcn{k}", conv)
            self.layers.append(f"mask_fcn{k}")
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0, activation=activation)
        weight_init.c2_msra_fill(self.deconv.deconv)

        cur_channels = conv_dims[-1]

        self.predictor = nn.Conv2d(cur_channels, num_classes, kernel_size=1)
        weight_init.normal_fill(self.predictor, std=0.001)

        self.loss_mask = loss_mask

    def forward(self, x):
        for layer in self.layers:
            x = getattr(self, layer)(x)
        x = self.deconv(x)
        x = self.predictor(x)
        return x

    def loss(self, pred_mask_logits: torch.Tensor, instances: list[Instances]):
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        mask_side_len = pred_mask_logits.size(2)
        if pred_mask_logits.size(2) != pred_mask_logits.size(3):
            raise ValueError("Mask prediction must be square!")

        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor,
                mask_side_len,
            ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        if len(gt_masks) == 0:
            return {"loss_mask": pred_mask_logits.sum() * 0}

        gt_masks = cat(gt_masks, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)

        # Log the training accuracy (using gt classes and sigmoid(0.0) == 0.5 threshold)
        mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
        mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        num_positive = gt_masks_bool.sum().item()
        false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(gt_masks_bool.numel() - num_positive, 1.0)
        false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

        storage = get_event_storage()
        storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
        storage.put_scalar("mask_rcnn/false_positive", false_positive)
        storage.put_scalar("mask_rcnn/false_negative", false_negative)

        loss_mask = self.loss_mask(pred_mask_logits, gt_masks)
        return {"loss_mask": loss_mask}

    def inference(self, pred_mask_logits: torch.Tensor, pred_instances: list[Instances]):
        cls_agnostic_mask = pred_mask_logits.size(1) == 1

        if cls_agnostic_mask:
            mask_probs_pred = pred_mask_logits.sigmoid()
        else:
            # Select masks corresponding to the predicted classes
            num_masks = pred_mask_logits.shape[0]
            class_pred = cat([i.pred_classes for i in pred_instances])
            device = class_pred.device
            indices = torch.arange(num_masks, device=device)
            mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

        for prob, instances in zip(mask_probs_pred, pred_instances, strict=True):
            instances.pred_masks = prob  # (1, Hmask, Wmask)

        return pred_instances
