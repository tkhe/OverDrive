# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for OverDrive from MMDetection (Apache-2.0); see LICENSES/mmdetection.txt.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from overdrive.layers import Conv2d

from .solo_head import _BaseSOLOHead, _coordinate_features, _init_convs

__all__ = ["SOLOV2Head", "SOLOV2MaskFeatHead"]


class SOLOV2MaskFeatHead(nn.Module):
    """Fuse FPN levels into a shared mask feature map at the finest stride.

    Each coarser level uses one convolution and 2x bilinear upsampling per
    stride step. Coordinates are added to the coarsest input feature.
    """

    def __init__(self, input_shape, in_features, *, feat_channels=128, out_channels=256, norm="GN"):
        super().__init__()
        if not in_features:
            raise ValueError("in_features must not be empty")
        strides = [input_shape[f].stride for f in in_features]
        if any(s is None or s <= 0 for s in strides) or any(b != a * 2 for a, b in zip(strides, strides[1:])):
            raise ValueError("Mask feature strides must be log2 contiguous")
        self.in_features = list(in_features)
        self.mask_stride = strides[0]
        self.out_channels = out_channels
        self.convs_all_levels = nn.ModuleList()
        for level, feature in enumerate(in_features):
            channels = input_shape[feature].channels + (2 if level == len(in_features) - 1 else 0)
            layers = []
            for i in range(max(1, level)):
                layers.append(Conv2d(channels if i == 0 else feat_channels, feat_channels, 3,
                                     norm=norm, activation="ReLU", bias=not bool(norm)))
                if level:
                    layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.convs_all_levels.append(nn.Sequential(*layers))
        self.conv_pred = Conv2d(feat_channels, out_channels, 1, norm=norm, activation="ReLU", bias=not bool(norm))
        _init_convs(self)

    def forward(self, features):
        combined = None
        for level, (feature, branch) in enumerate(zip(self.in_features, self.convs_all_levels, strict=True)):
            x = features[feature]
            if level == len(self.in_features) - 1:
                x = torch.cat((x, _coordinate_features(x)), dim=1)
            x = branch(x)
            if combined is None:
                combined = x
            else:
                # Support odd feature sizes without changing the standard FPN path.
                if x.shape[-2:] != combined.shape[-2:]:
                    x = F.interpolate(x, size=combined.shape[-2:], mode="bilinear", align_corners=False)
                combined = combined + x
        return self.conv_pred(combined)


class SOLOV2Head(_BaseSOLOHead):
    """SOLOv2 category/kernel towers with shared mask features.

    Forward returns raw ``(kernel_logits, class_logits, mask_features)``.
    Dynamic masks are generated only for positive training cells or inference
    candidates. Loss modules must use sum or none reduction.
    """

    def __init__(self, input_shape, in_features, *, mask_feature_head: SOLOV2MaskFeatHead,
                 feat_channels=512, stacked_convs=4, norm="GN", dynamic_conv_size=1,
                 prior_prob=0.01, **kwargs):
        super().__init__(input_shape, in_features, **kwargs)
        if stacked_convs <= 0:
            raise ValueError("stacked_convs must be positive")
        if dynamic_conv_size <= 0 or dynamic_conv_size % 2 != 1:
            raise ValueError("dynamic_conv_size must be a positive odd integer")
        if not 0 < prior_prob < 1:
            raise ValueError("prior_prob must be in (0, 1)")
        self.mask_feature_head = mask_feature_head
        self.dynamic_conv_size = dynamic_conv_size
        self.kernel_out_channels = mask_feature_head.out_channels * dynamic_conv_size ** 2
        self.kernel_convs = nn.Sequential(*[
            Conv2d(self.in_channels + 2 if i == 0 else feat_channels, feat_channels, 3,
                   norm=norm, activation="ReLU", bias=not bool(norm))
            for i in range(stacked_convs)
        ])
        self.cls_convs = nn.Sequential(*[
            Conv2d(self.in_channels if i == 0 else feat_channels, feat_channels, 3,
                   norm=norm, activation="ReLU", bias=not bool(norm))
            for i in range(stacked_convs)
        ])
        self.conv_cls = nn.Conv2d(feat_channels, self.num_classes, 3, padding=1)
        self.conv_kernel = nn.Conv2d(feat_channels, self.kernel_out_channels, 3, padding=1)
        _init_convs(self)
        nn.init.constant_(self.conv_cls.bias, math.log(prior_prob / (1 - prior_prob)))

    def forward(self, features):
        mask_feats = self.mask_feature_head(features)
        kernel_preds, cls_preds = [], []
        for x, grid in zip(self._resize_features(features), self.num_grids, strict=True):
            x = torch.cat((x, _coordinate_features(x)), dim=1)
            x = F.interpolate(x, size=(grid, grid), mode="bilinear", align_corners=False)
            kernel_preds.append(self.conv_kernel(self.kernel_convs(x)))
            cls_preds.append(self.conv_cls(self.cls_convs(x[:, :-2].contiguous())))
        return kernel_preds, cls_preds, mask_feats

    def _dynamic_masks(self, mask_feats, kernels):
        if not len(kernels):
            return mask_feats.new_empty((0, *mask_feats.shape[-2:]))
        weights = kernels.reshape(-1, self.mask_feature_head.out_channels, self.dynamic_conv_size, self.dynamic_conv_size)
        return F.conv2d(mask_feats.unsqueeze(0), weights, padding=self.dynamic_conv_size // 2)[0]

    def loss(self, kernel_preds, cls_preds, mask_feats, gt_instances, padded_size):
        sizes = [mask_feats.shape[-2:]] * len(self.num_grids)
        targets = [self._get_targets_single(gt, sizes, padded_size, keep_duplicates=True) for gt in gt_instances]
        normalizer = self._normalizer(targets, cls_preds[0])
        # Include kernels even when a batch has no foreground, so DDP sees all parameters.
        loss_mask = mask_feats.reshape(-1)[:0].sum() + sum(x.reshape(-1)[:0].sum() for x in kernel_preds)
        for image, (_, indices, masks) in enumerate(targets):
            for level, index in enumerate(indices):
                if len(index):
                    kernels = kernel_preds[level][image].flatten(1).t()[index]
                    logits = self._dynamic_masks(mask_feats[image], kernels)
                    loss_mask = loss_mask + self.loss_mask(logits, masks[level]).sum()
        return {"loss_mask": loss_mask / normalizer.clamp_min(1), "loss_cls": self._classification_loss(cls_preds, targets, normalizer)}

    @torch.no_grad()
    def inference(self, kernel_preds, cls_preds, mask_feats, image_sizes, padded_size):
        results = []
        for image, image_size in enumerate(image_sizes):
            scores, labels, cells, strides = self._candidate_scores(cls_preds, image)
            kernels = torch.cat([x[image].flatten(1).t() for x in kernel_preds])[cells]
            probs = self._dynamic_masks(mask_feats[image], kernels).sigmoid()
            results.append(self._inference_single(probs, scores, labels, strides, image_size, padded_size))
        return results
