# 更新日志

[English](../en/changelog.md) | 简体中文

## v1.0.0

OverDrive 首个版本提供基于 PyTorch 的模块化视觉任务框架，覆盖目标检测、实例分割、关键点检测、语义分割和全景分割。

### 支持的算法

- **目标检测**：Faster R-CNN、Cascade R-CNN、RetinaNet、FCOS、AutoAssign、DETR、Conditional DETR、DAB-DETR、Deformable DETR。
- **实例分割**：Mask R-CNN、Cascade Mask R-CNN。
- **关键点检测**：Keypoint R-CNN。
- **语义分割**：DeepLab v3、DeepLab v3+、UPerNet。
- **全景分割**：PanopticFPN。

### 模型与配置

- 提供模块化的 backbone、neck、预测 head、损失函数及后处理组件。
- 使用 Python LazyConfig 组织实验，支持配置复用、对象实例化和命令行覆盖。
- 提供共享模型、数据处理流程、优化器和学习率调度配置。
- 提供多种骨干网络及训练周期的实验配置。

### 训练与评估

- 支持按 epoch 或 iteration 训练。
- 支持分布式训练、混合精度和梯度累积。
- 支持断点续训、检查点管理及 TensorBoard 日志。
- 提供目标检测、实例分割、关键点检测、语义分割和全景分割评估器。

### 数据与推理

- 提供 COCO、LVIS、Cityscapes、ADE20K、PASCAL VOC 和 VisDrone 数据集接口。
- 支持自定义数据集注册和数据增强配置。
- 提供图片、视频、摄像头推理及结果可视化。
- 提供 Python 推理接口。

### 文档

- 提供中英文 README、安装与使用指南。
- 提供数据准备、训练、评估和推理说明。
- 提供进阶配置、自定义数据集及常见问题文档。
