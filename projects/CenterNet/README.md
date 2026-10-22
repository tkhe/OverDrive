# CenterNet

> [Objects as Points](https://arxiv.org/abs/1904.07850)

## Abstract

Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point --- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

## What's New

### v0.1.0

Initial release of the CenterNet object detection project built on OverDrive.

- Implement CenterNet with center heatmap, bounding box size, and center offset prediction, using Gaussian focal loss and L1 regression losses.
- Provide a ResNet upsampling neck with standard convolution and DCNv2 variants.
- Add COCO detection configurations for [ResNet-18](configs/COCO-Detection/centernet_resnet18_140e.py) and [ResNet-18 with DCNv2](configs/COCO-Detection/centernet_resnet18_dcnv2_140e.py), both using a 140-epoch training schedule.
- Provide instance-aware random cropping, multi-scale resizing, color augmentation, and random horizontal flipping for training.
- Integrate with OverDrive's LazyConfig, training, evaluation, and detection postprocessing components.

## Model Zoo
