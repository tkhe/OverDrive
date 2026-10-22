# AutoAssign

> [AutoAssign: Differentiable Label Assignment for Dense Object Detection](https://arxiv.org/abs/2007.03496)

## Abstract

Determining positive/negative samples for object detection is known as label assignment. Here we present an anchor-free detector named AutoAssign. It requires little human knowledge and achieves appearance-aware through a fully differentiable weighting mechanism. During training, to both satisfy the prior distribution of data and adapt to category characteristics, we present Center Weighting to adjust the category-specific prior distributions. To adapt to object appearances, Confidence Weighting is proposed to adjust the specific assign strategy of each instance. The two weighting modules are then combined to generate positive and negative weights to adjust each location's confidence. Extensive experiments on the MS COCO show that our method steadily surpasses other best sampling strategies by large margins with various backbones. Moreover, our best model achieves 52.1% AP, outperforming all existing one-stage detectors. Besides, experiments on other datasets, e.g., PASCAL VOC, Objects365, and WiderFace, demonstrate the broad applicability of AutoAssign.

## Model Zoo
