<div align="center">
  <img src="assets/overdrive-logo-stacked.svg" alt="OverDrive" width="420" />
  <p><strong>基于 PyTorch 的开源视觉任务工具箱。</strong></p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](docs/zh/get_started.md#环境要求)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-orange)](docs/zh/get_started.md#环境要求)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[安装](#安装) |
[快速开始](#快速开始) |
[模型库](#模型库) |
[常见问题](#常见问题) |
[问题反馈](https://github.com/tkhe/OverDrive/issues)

[English](README.md) | 简体中文

</div>

## 简介

OverDrive 是一个基于 PyTorch 的开源视觉任务框架与工具箱，旨在提供灵活、模块化且易于扩展的基础框架。通过可复用的模型组件、灵活的配置系统，以及统一且可定制的数据处理、训练和评估接口，帮助开发者高效地实现算法、开展实验并探索新的视觉任务。


OverDrive 要求 **PyTorch 2.6+**，请使用 **Python 3.10+** 和与 PyTorch 匹配的 TorchVision。

<details open>
<summary>主要特性</summary>

- **模块化设计**

  复用模型组件，并按任务需要定制数据处理、训练循环、Hook 和评估器，为不同视觉任务构建模型与实验流程。

- **灵活的实验配置**

  使用 Python LazyConfig 组织和复用共享模型与训练配置，通过 `LazyCall` 定义组件、`instantiate` 构建对象，并在命令行中覆盖实验参数。

- **训练、评估与可视化**

  支持按 epoch 或 iteration 训练、分布式运行、混合精度、梯度累积、断点续训与 TensorBoard 日志。数据集评估器和图片、视频、摄像头 Demo 覆盖实验流程。

</details>

## 安装

先按照 [PyTorch 官方安装向导](https://pytorch.org/get-started/locally/)安装适合当前平台的 PyTorch 和 TorchVision，再从源码安装 OverDrive：

```bash
git clone https://github.com/tkhe/OverDrive.git
cd OverDrive
python -m pip install -e . --no-build-isolation
```

安装时会编译 C++ 扩展，需要与 PyTorch 兼容的编译工具链。虚拟环境设置、CPU/CUDA 要求、可选依赖和安装检查见[安装指南](docs/zh/get_started.md#安装)。

## 快速开始

按照[使用指南](docs/zh/get_started.md)完成数据准备、模型训练、检查点评估和推理。示例使用 COCO 数据集上的 Faster R-CNN ResNet-50 FPN。

<details open>
<summary>使用指南</summary>

- [准备数据集](docs/zh/get_started.md#数据准备)
- [单卡训练](docs/zh/get_started.md#单卡训练)
- [多卡与多机训练](docs/zh/get_started.md#多卡与多机训练)
- [恢复训练](docs/zh/get_started.md#恢复训练)
- [评估模型](docs/zh/get_started.md#评估模型)
- [图片、视频与摄像头推理](docs/zh/get_started.md#图片视频与摄像头推理)
- [Python 推理接口](docs/zh/get_started.md#python-推理接口)

</details>

<details>
<summary>进阶指南</summary>

- [理解与修改配置](docs/zh/advanced_usage.md#配置与实验管理)
- [注册自定义数据集](docs/zh/advanced_usage.md#自定义数据集)
- [项目结构与可视化工具](docs/zh/advanced_usage.md#项目结构)

</details>

## 模型库

[configs/](configs) 提供实验配置和相应的论文入口。共享模型、数据处理流程、优化器和学习率计划位于 [configs/common](configs/common)。

### 模型架构

<table>
  <thead>
    <tr>
      <th>目标检测</th>
      <th>实例分割</th>
      <th>关键点检测</th>
      <th>语义分割</th>
      <th>全景分割</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td valign="top">
        <ul>
          <li><a href="configs/COCO-Detection/faster_rcnn">Faster R-CNN</a></li>
          <li><a href="configs/COCO-Detection/cascade_rcnn">Cascade R-CNN</a></li>
          <li><a href="configs/COCO-Detection/retinanet">RetinaNet</a></li>
          <li><a href="configs/COCO-Detection/fcos">FCOS</a></li>
          <li><a href="configs/COCO-Detection/autoassign">AutoAssign</a></li>
          <li><a href="configs/COCO-Detection/detr">DETR</a></li>
          <li><a href="configs/COCO-Detection/conditional_detr">Conditional DETR</a></li>
          <li><a href="configs/COCO-Detection/dab_detr">DAB-DETR</a></li>
          <li><a href="configs/COCO-Detection/deformable_detr">Deformable DETR</a></li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><a href="configs/COCO-InstanceSegmentation/mask_rcnn">Mask R-CNN</a></li>
          <li><a href="configs/COCO-InstanceSegmentation/cascade_mask_rcnn">Cascade Mask R-CNN</a></li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><a href="configs/COCO-Keypoint/keypoint_rcnn">Keypoint R-CNN</a></li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><a href="configs/ADE20K-SemanticSegmentation/deeplabv3">DeepLab v3</a></li>
          <li><a href="configs/ADE20K-SemanticSegmentation/deeplabv3plus">DeepLab v3+</a></li>
          <li><a href="configs/ADE20K-SemanticSegmentation/upernet">UPerNet</a></li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><a href="configs/COCO-PanopticSegmentation/panoptic_fpn">PanopticFPN</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

### 数据集与评估

内置数据接口覆盖 **COCO、LVIS、Cityscapes、ADE20K、PASCAL VOC 和 VisDrone**。[评估模块](overdrive/evaluation)提供目标检测、实例分割、关键点、语义分割与全景分割评估器。

标注格式和注册名称见[数据准备指南](docs/zh/get_started.md#数据准备)与[数据集实现](overdrive/data/datasets)。扩展骨干网络和数据集时，请按模型的特征尺寸与任务要求调整配置。

## 常见问题

安装、算子兼容性、数据路径、显存、权重加载和命令行参数等问题见[常见问题文档](docs/zh/faq.md)。

其他问题可以通过 [GitHub Issues](https://github.com/tkhe/OverDrive/issues) 反馈，请附上环境信息、所用配置、完整命令和相关日志。

## 参与贡献

欢迎提交问题、修复、新视觉任务与模型、数据集接口和文档改进。新增任务或模型时，请提供实现、实验配置和可复现的验证说明。独立研究扩展可以放在 [projects/](projects) 下。

涉及模型行为、数据处理或训练引擎的改动，请给出相应的运行验证结果。

## 致谢

感谢以下开源项目提供的工具和研究成果：

- [PyTorch](https://github.com/pytorch/pytorch)
- [TorchVision](https://github.com/pytorch/vision)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [detrex](https://github.com/IDEA-Research/detrex)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [timm](https://github.com/huggingface/pytorch-image-models)

## 许可证

OverDrive 使用 [MIT License](LICENSE)。数据集、第三方代码和预训练权重的使用条件以各自的许可证和发布说明为准。
