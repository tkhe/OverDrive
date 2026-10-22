import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from overdrive.layers import Conv2d
from overdrive.layers import ShapeSpec
from overdrive.layers import weight_init
from overdrive.modeling.dense_head import DenseHead
from overdrive.structures import Boxes
from overdrive.structures import Instances

__all__ = ["CenterNetHead"]


class CenterNetHead(DenseHead):
    """
    support CenterNetHead proposed in https://arxiv.org/abs/1904.07850.
    """

    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        in_features: list[str],
        *,
        num_classes: int,
        channels: int,
        norm: str | dict = "",
        activation: str | dict = "ReLU",
        loss_heatmap: nn.Module,
        loss_wh: nn.Module,
        loss_offset: nn.Module,
        test_local_maximum_kernel: int = 3,
        test_topk: int = 100,
        test_score_threshold: float = 0.05,
    ):
        super().__init__()

        if len(in_features) != 1:
            raise ValueError("CenterNetHead only uses the last feature map.")

        in_channels = input_shape[in_features[0]].channels
        stride = input_shape[in_features[0]].stride

        self.num_classes = num_classes
        self.in_features = in_features
        self.stride = stride

        self.test_local_maximum_kernel = test_local_maximum_kernel
        self.test_topk = test_topk
        self.test_score_threshold = test_score_threshold

        self.loss_heatmap = torch.compile(loss_heatmap)
        self.loss_wh = loss_wh
        self.loss_offset = loss_offset

        self.heatmap_head = nn.Sequential(
            Conv2d(in_channels, channels, kernel_size=3, stride=1, norm=norm, activation=activation),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )
        self.wh_head = nn.Sequential(
            Conv2d(in_channels, channels, kernel_size=3, stride=1, norm=norm, activation=activation),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        self.offset_head = nn.Sequential(
            Conv2d(in_channels, channels, kernel_size=3, stride=1, norm=norm, activation=activation),
            nn.Conv2d(channels, 2, kernel_size=1),
        )

        bias_value = -(math.log((1 - 0.1) / 0.1))
        weight_init.bias_fill(self.heatmap_head[-1], bias_value)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    weight_init.normal_fill(m, std=0.001)

    def forward(self, features: dict[str, torch.Tensor]):
        features = features[self.in_features[0]]

        heatmap = self.heatmap_head(features).sigmoid()
        wh = self.wh_head(features)
        offset = self.offset_head(features)

        return heatmap, wh, offset

    def loss(self, heatmap, wh, offset, gt_instances):
        heatmap_target, wh_target, offset_target, wh_offset_target_weight, avg_factor = self.get_ground_truth(heatmap, gt_instances)

        loss_heatmap = self.loss_heatmap(heatmap, heatmap_target) / avg_factor
        loss_wh = self.loss_wh(wh * wh_offset_target_weight, wh_target) / (avg_factor * 2)
        loss_offset = self.loss_offset(offset * wh_offset_target_weight, offset_target) / (avg_factor * 2)
        return {
            "loss_heatmap": loss_heatmap,
            "loss_wh": loss_wh,
            "loss_offset": loss_offset,
        }

    @torch.no_grad()
    def get_ground_truth(self, heatmap: torch.Tensor, gt_instances):
        N, _, H, W = heatmap.shape
        device = heatmap.device

        # all gts of the batch are processed at once, on the device, without host syncs.
        gt_boxes = torch.cat([x.gt_boxes.tensor for x in gt_instances]).to(device)
        gt_classes = torch.cat([x.gt_classes for x in gt_instances]).to(device)
        num_gts = torch.tensor([len(x) for x in gt_instances], device=device)
        batch_ids = torch.repeat_interleave(torch.arange(N, device=device), num_gts, output_size=gt_boxes.shape[0])

        centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2 / self.stride
        centers_int = centers.long()
        # (w, h) in float64 so the radius matches the scalar Python formula.
        box_wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).double() / self.stride
        radius = gaussian_radius(box_wh[:, 1], box_wh[:, 0], min_overlap=0.3).long().clamp(min=0)

        # one gaussian per gt over the whole map, cut to its (2r+1)^2 window, merged by max.
        sigma = (2 * radius + 1).double() / 6
        dx = torch.arange(W, device=device, dtype=torch.float32) - centers_int[:, 0:1]
        dy = torch.arange(H, device=device, dtype=torch.float32) - centers_int[:, 1:2]
        gaussian = (dx * dx)[:, None, :] + (dy * dy)[:, :, None]
        gaussian.div_(-(2 * sigma * sigma).float()[:, None, None]).exp_()
        outside = (dx.abs() > radius[:, None])[:, None, :] | (dy.abs() > radius[:, None])[:, :, None]
        gaussian.masked_fill_(outside | (gaussian < torch.finfo(gaussian.dtype).eps), 0)

        heatmap_target = torch.zeros([N * self.num_classes, H * W], device=device)
        index = (batch_ids * self.num_classes + gt_classes)[:, None].expand(-1, H * W)
        heatmap_target.scatter_reduce_(0, index, gaussian.view(-1, H * W), "amax")
        heatmap_target = heatmap_target.view(N, self.num_classes, H, W)

        # gts whose centers share a cell all write the values of the last one, as a sequential loop would.
        cells = (batch_ids * H + centers_int[:, 1]) * W + centers_int[:, 0]
        gt_ids = torch.arange(cells.shape[0], device=device)
        winner = torch.full([N * H * W], -1, device=device).scatter_reduce_(0, cells, gt_ids, "amax")[cells]

        wh_target = torch.zeros([N, 2, H, W], device=device)
        offset_target = torch.zeros([N, 2, H, W], device=device)
        wh_offset_target_weight = torch.zeros([N, 2, H, W], device=device)
        wh_target[batch_ids, :, centers_int[:, 1], centers_int[:, 0]] = box_wh[winner].float()
        offset_target[batch_ids, :, centers_int[:, 1], centers_int[:, 0]] = (centers - centers_int)[winner]
        wh_offset_target_weight[batch_ids, :, centers_int[:, 1], centers_int[:, 0]] = 1

        avg_factor = heatmap_target.eq(1).sum().clamp(min=1)
        return heatmap_target, wh_target, offset_target, wh_offset_target_weight, avg_factor

    def inference(self, heatmap, wh, offset, image_sizes):
        heatmap = pseudo_nms(heatmap, kernel_size=self.test_local_maximum_kernel)

        results = []
        for image_id, image_size in enumerate(image_sizes):
            results_per_image = self.inference_single_image(
                heatmap[image_id],
                wh[image_id],
                offset[image_id],
                image_size,
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, heatmap, wh, offset, image_size):
        scores, classes_idxs, ys, xs = get_topk(heatmap, self.test_topk, self.test_score_threshold)
        wh = wh[:, ys, xs]
        offset = offset[:, ys, xs]
        xs = xs + offset[0]
        ys = ys + offset[1]
        tl_x = (xs - wh[0] / 2) * self.stride
        tl_y = (ys - wh[1] / 2) * self.stride
        br_x = (xs + wh[0] / 2) * self.stride
        br_y = (ys + wh[1] / 2) * self.stride
        boxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=1)
        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = classes_idxs
        return result


def gaussian_radius(height: torch.Tensor, width: torch.Tensor, min_overlap: float):
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1**2 - 4 * a1 * c1).sqrt()
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2**2 - 4 * a2 * c2).sqrt()
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3**2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / (2 * a3)
    return torch.minimum(torch.minimum(r1, r2), r3)


def pseudo_nms(heatmap, kernel_size=3):
    padding = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=padding)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def get_topk(
    heatmap: torch.Tensor,
    topk: int = 100,
    score_threshold: float = 0.05,
):
    _, H, W = heatmap.shape

    scores, inds = torch.topk(heatmap.reshape(-1), k=topk)

    if score_threshold > 0:
        keep = scores > score_threshold
        scores = scores[keep]
        inds = inds[keep]

    classes_idxs = inds // (H * W)
    inds = inds % (H * W)
    ys = inds // W
    xs = inds % W

    return scores, classes_idxs, ys, xs
