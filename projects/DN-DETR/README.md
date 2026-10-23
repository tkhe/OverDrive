# DN-DETR

> [DN-DETR: Accelerate DETR Training by Introducing Query DeNoising](https://arxiv.org/abs/2203.01305)

## Abstract

We present in this paper a novel denoising training method to speedup DETR (DEtection TRansformer) training and offer a deepened understanding of the slow convergence issue of DETR-like methods. We show that the slow convergence results from the instability of bipartite graph matching which causes inconsistent optimization goals in early training stages. To address this issue, except for the Hungarian loss, our method additionally feeds ground-truth bounding boxes with noises into Transformer decoder and trains the model to reconstruct the original boxes, which effectively reduces the bipartite graph matching difficulty and leads to a faster convergence. Our method is universal and can be easily plugged into any DETR-like methods by adding dozens of lines of code to achieve a remarkable improvement. As a result, our DN-DETR results in a remarkable improvement (+1.9AP) under the same setting and achieves the best result (AP 43.4 and 48.6 with 12 and 50 epochs of training respectively) among DETR-like methods with ResNet-50 backbone. Compared with the baseline under the same setting, DN-DETR achieves comparable performance with 50% training epochs.

## What's New

### v1.0.0

Initial release of the DN-DETR object detection project built on OverDrive.

- Implement denoising query generation, transformer encoding and decoding, matching and denoising losses, and iterative box refinement.
- Provide a ResNet-50 DN-DETR model and a 50-epoch COCO detection configuration.
- Include multi-scale COCO data augmentation, optimizer and training configurations, model zoo APIs, and a distributed training entrypoint.

## Model Zoo
