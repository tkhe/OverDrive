# 使用指南

[返回 README](../../README_zh.md) | [English](../en/get_started.md) | 简体中文

- [安装](#安装)
- [数据准备](#数据准备)
- [快速开始](#快速开始)

## 安装

### 环境要求

| 依赖 | 要求或说明 |
| --- | --- |
| Python | 使用 **3.10 及以上** |
| PyTorch | **2.6 及以上**，安装与导入时都会检查版本 |
| TorchVision | 与 PyTorch 匹配的版本，提供 NMS、ROIAlign 等算子；需要先单独安装 |
| C++ 编译器 | 安装时编译 `overdrive._C` 扩展，需要与 PyTorch 兼容的编译工具链 |
| CUDA | GPU 训练需要可用的 NVIDIA GPU 和驱动；构建 CUDA 扩展时还需要匹配的 CUDA Toolkit 与 `nvcc` |

GPU 训练以 Linux + NVIDIA CUDA 环境为主要使用场景。默认配置中的 `train.device` 为 `cuda`；CPU 推理或调试需显式改为 `cpu`。示例命令使用 Bash/Zsh，请在仓库根目录执行。

### 从源码安装

```bash
git clone https://github.com/tkhe/OverDrive.git
cd OverDrive

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install ninja
```

先安装 PyTorch 和 TorchVision。根据系统与计算平台，使用 [PyTorch 官方安装向导](https://pytorch.org/get-started/locally/)选择命令，并确保 PyTorch >= 2.6。macOS 可使用下面的基础命令；Linux 的 CPU/CUDA 环境请使用向导给出的对应软件源：

```bash
python -m pip install "torch>=2.6" torchvision
```

再安装 OverDrive：

```bash
python -m pip install -e . --no-build-isolation
```

`setup.py` 在构建时会导入 PyTorch，因此需要先安装它，并通过 `--no-build-isolation` 使用当前环境。安装会编译 C++ 扩展，同时将 `configs/` 链接或复制到 `overdrive/model_zoo/configs/`，供 Model Zoo 配置接口使用。

可选依赖按需安装：

```bash
# LVIS、Cityscapes、全景分割等可选功能的依赖
python -m pip install -e ".[all]" --no-build-isolation
```

`all` 额外依赖包含从 GitHub 安装的 `panopticapi`，需要能访问对应网络资源；普通 COCO 检测训练无需安装全部可选依赖。

### 检查安装

```bash
python -c "import torch, torchvision, overdrive; print('PyTorch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('OverDrive:', overdrive.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "from overdrive import _C; print('Compiler:', _C.get_compiler_version()); print('Extension CUDA:', _C.get_cuda_version())"
python -c "from overdrive.utils.collect_env import collect_env_info; print(collect_env_info())"
```

CPU 构建的扩展会显示 CUDA 不可用；这与 PyTorch 本身是否能使用 GPU 是两项不同的检查。

## 数据准备

内置数据集默认从仓库根目录下的 `datasets/` 读取。也可以在启动 Python **之前**指定外部数据目录：

```bash
export OVERDRIVE_DATASETS=/path/to/datasets
```

数据集路径在导入时注册，运行中的 Python 进程修改此变量后需要重新注册数据集或重启进程。项目不包含数据文件，需自行下载并按以下结构组织。

### COCO 2017

目标检测与实例分割使用相同的图片及 instances 标注：

```text
datasets/
└── coco/
    ├── train2017/
    │   └── *.jpg
    ├── val2017/
    │   └── *.jpg
    └── annotations/
        ├── instances_train2017.json
        ├── instances_val2017.json
        ├── person_keypoints_train2017.json  # 关键点任务需要
        └── person_keypoints_val2017.json    # 关键点任务需要
```

默认注册名为 `coco_2017_train`、`coco_2017_val`；关键点任务使用 `keypoints_coco_2017_train`、`keypoints_coco_2017_val`。图片推理只需配置、完整模型权重和输入图片，无需下载 COCO 训练集或标注。

### ADE20K

```text
datasets/
└── ADEChallengeData2016/
    ├── images/
    │   ├── training/
    │   └── validation/
    └── annotations/
        ├── training/
        └── validation/
```

运行标注转换脚本：

```bash
python datasets/prepare_ade20k_sem_seg.py
```

脚本同样读取 `OVERDRIVE_DATASETS`，生成 `annotations_overdrive/training/` 和 `annotations_overdrive/validation/`，将类别标签 `1–150` 转成 `0–149`，原标签 `0` 转为忽略标签 `255`。内置注册名为 `ade20k_sem_seg_train`、`ade20k_sem_seg_val`，读取的是转换后的标注。

其他数据集的注册名和目录规则见 [builtin.py](../../overdrive/data/datasets/builtin.py) 与[数据集实现目录](../../overdrive/data/datasets)。全景分割需要额外准备对应的 panoptic 标注；提供数据接口和评估器不代表每个数据集都有现成的训练配置。

## 快速开始

以下示例使用 Faster R-CNN ResNet-50 FPN。训练和评估需要完成 COCO 数据准备；推理需要与配置匹配的完整模型权重。

### 单卡训练

```bash
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 1 \
  train.output_dir=./output/faster_rcnn_r50 \
  dataloader.train.total_batch_size=2 \
  optimizer.lr=0.0025
```

原始配置使用全局 batch size `16`、学习率 `0.02`、训练 `12` 个 epoch。上例将 batch size 降到 `2`，并按比例调整学习率，便于单卡起步；实际参数需按显存与实验目标调整。框架不会根据 GPU 数量自动缩放学习率。

该配置默认从 `torchvision://ImageNet/ResNet-50` 加载骨干网络权重，首次运行会按项目权重目录解析远程地址并缓存。离线环境可将 `train.init_checkpoint` 改为本地文件；从随机初始化开始训练可追加 `train.init_checkpoint='""'`。

### 多卡与多机训练

单机 8 卡，使用原始全局 batch size 与学习率：

```bash
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 8 \
  --dist-url auto \
  train.output_dir=./output/faster_rcnn_r50_8gpu
```

`--num-gpus` 表示每台机器的 GPU 数量。全局 batch size 必须能被总训练进程数整除。多机训练时，在每台机器运行相同命令，并分别设置 `--machine-rank`：

```bash
# 节点 0；将 192.168.1.10 替换为节点 0 可被其他节点访问的地址
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 4 --num-machines 2 --machine-rank 0 \
  --dist-url tcp://192.168.1.10:29500 \
  train.output_dir=./output/faster_rcnn_r50_2nodes

# 节点 1
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 4 --num-machines 2 --machine-rank 1 \
  --dist-url tcp://192.168.1.10:29500 \
  train.output_dir=./output/faster_rcnn_r50_2nodes
```

各节点需使用相同代码、配置和可访问的数据路径，初始化地址与端口必须互通。`--dist-url auto` 仅适用于单机。

### 恢复训练

```bash
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 1 \
  --resume \
  train.output_dir=./output/faster_rcnn_r50 \
  dataloader.train.total_batch_size=2 \
  optimizer.lr=0.0025
```

`--resume` 优先读取输出目录中 `last_checkpoint` 指向的检查点，恢复模型和训练状态；找不到检查点时，加载 `train.init_checkpoint`。恢复时请沿用原实验的配置与参数。未指定 `--resume` 时，初始化只加载模型权重。

### 评估模型

完成训练后，使用生成的 `model_final.pth`：

```bash
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 1 \
  --eval-only \
  train.init_checkpoint=./output/faster_rcnn_r50/model_final.pth \
  train.output_dir=./output/faster_rcnn_r50_eval
```

评估器根据测试数据集的 `evaluator_type` 选择，结果写入日志；COCO 预测结果等文件保存在输出目录的 `inference/` 下。也可以同时使用 `--eval-only --resume` 与原输出目录来评估最近保存的检查点。

### 图片、视频与摄像头推理

对单张图片生成可视化结果：

```bash
mkdir -p output/demo
python demo/demo.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --input /path/to/image.jpg \
  --output output/demo \
  --confidence-threshold 0.5 \
  --opts train.init_checkpoint=./output/faster_rcnn_r50/model_final.pth
```

批量图片可使用 `--input '/path/to/images/*.jpg'`，输出目录需要提前创建。视频可将 `--input` 替换为 `--video-input /path/to/video.mp4`；摄像头使用 `--webcam`，并省略 `--output`。省略输出路径时，结果显示在 OpenCV 窗口中，按 `Esc` 退出。

Demo 的配置覆盖项必须放在末尾的 `--opts` 后，格式仍为 `key=value`。CPU 推理可追加 `train.device=cpu`。推理精度由 `--precision fp32|fp16|bf16` 控制，默认为 `fp32`；`fp16` 要求 CUDA，`bf16` 需设备与算子支持。

### Python 推理接口

```python
import cv2

from overdrive.config import LazyConfig
from overdrive.engine import DefaultPredictor

cfg = LazyConfig.load(
    "configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py"
)
cfg.train.init_checkpoint = "./output/faster_rcnn_r50/model_final.pth"
cfg.train.device = "cuda"  # CPU 推理改为 "cpu"

predictor = DefaultPredictor(cfg)
image = cv2.imread("/path/to/image.jpg")  # 输入为 BGR 格式
if image is None:
    raise FileNotFoundError("/path/to/image.jpg")

predictions = predictor(image)
instances = predictions["instances"].to("cpu")
print(instances.pred_boxes.tensor)
print(instances.scores)
print(instances.pred_classes)
```

检测、实例分割和关键点模型返回 `instances`，分割与关键点结果分别包含 `pred_masks`、`pred_keypoints`；语义分割模型返回 `sem_seg`，可沿类别维度取 `argmax` 得到类别标签图。`DefaultPredictor` 会加载权重并完成测试配置中的图像预处理。
