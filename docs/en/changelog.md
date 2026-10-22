# Changelog

English | [简体中文](../zh/changelog.md)

## v1.0.0

The first release of OverDrive provides a modular PyTorch framework for object detection, instance segmentation, keypoint detection, semantic segmentation, and panoptic segmentation.

### Supported Algorithms

- **Object detection**: Faster R-CNN, Cascade R-CNN, RetinaNet, FCOS, AutoAssign, DETR, Conditional DETR, DAB-DETR, and Deformable DETR.
- **Instance segmentation**: Mask R-CNN and Cascade Mask R-CNN.
- **Keypoint detection**: Keypoint R-CNN.
- **Semantic segmentation**: DeepLab v3, DeepLab v3+, and UPerNet.
- **Panoptic segmentation**: PanopticFPN.

### Models and Configuration

- Modular backbones, necks, prediction heads, loss functions, and postprocessing components.
- Python LazyConfig for experiment configuration, with configuration reuse, object instantiation, and command-line overrides.
- Shared configurations for models, data pipelines, optimizers, and learning rate schedules.
- Experiment configurations with different backbones and training schedules.

### Training and Evaluation

- Epoch-based and iteration-based training.
- Distributed training, mixed precision, and gradient accumulation.
- Checkpoint resumption, checkpoint management, and TensorBoard logging.
- Evaluators for object detection, instance segmentation, keypoint detection, semantic segmentation, and panoptic segmentation.

### Data and Inference

- Dataset interfaces for COCO, LVIS, Cityscapes, ADE20K, PASCAL VOC, and VisDrone.
- Custom dataset registration and configurable data augmentation.
- Image, video, and webcam inference with result visualization.
- A Python inference API.

### Documentation

- English and Chinese READMEs, installation instructions, and user guides.
- Instructions for dataset preparation, training, evaluation, and inference.
- Advanced configuration, custom dataset, and troubleshooting guides.
