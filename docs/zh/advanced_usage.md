# 进阶指南

[返回 README](../../README_zh.md) | [English](../en/advanced_usage.md) | 简体中文

- [进阶指南](#进阶指南)
  - [配置与实验管理](#配置与实验管理)
    - [配置组织](#配置组织)
    - [常用覆盖项](#常用覆盖项)
    - [输出文件](#输出文件)
  - [自定义数据集](#自定义数据集)
  - [项目结构](#项目结构)

## 配置与实验管理

### 配置组织

[configs/common](../../configs/common)存放共享模型、数据、优化器、训练设置与学习率计划；任务目录中的配置将它们组合成实验。使用 Model Zoo 读取配置时，路径相对于 `configs/`：

```python
from overdrive.config import instantiate
from overdrive.model_zoo import get_config

cfg = get_config("COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py")
cfg.model.roi_head.num_classes = 3
cfg.dataloader.train.total_batch_size = 4
cfg.train.output_dir = "./output/my_experiment"

model = instantiate(cfg.model)  # 仅构建模型，不会自动加载权重
```

`LazyConfig.load()` 返回配置对象，`LazyCall` 描述待调用的类或函数，`instantiate()` 递归构建这些对象。不同模型的类别数、输出通道与阈值所在字段不同，修改前应查看对应的共享模型配置。

### 常用覆盖项

训练入口的覆盖项直接放在命令末尾，Demo 放在 `--opts` 后；两者都使用 `path.key=value`。

| 配置项 | 用途 |
| --- | --- |
| `train.output_dir` | 日志、配置快照、检查点与 TensorBoard 输出目录 |
| `train.init_checkpoint` | 初始化权重路径或已登记的权重 URI |
| `train.device` | 模型设备，默认 `cuda` |
| `train.mode` | `epoch` 或 `iteration` |
| `train.max_epoch` / `train.max_iter` | 对应模式的训练长度，需与学习率计划配套修改 |
| `dataloader.train.total_batch_size` | 所有训练进程合计的 batch size |
| `dataloader.train.num_workers` | 每个进程的数据加载 worker 数量 |
| `optimizer.lr` | 优化器学习率 |
| `train.amp.enable` | 是否启用训练和评估的 CUDA 混合精度 |
| `train.amp.mixed_precision` | `fp16` 或 `bf16`，默认 `fp16` |
| `train.gradient_accumulation_steps` | 每次参数更新累计的 batch 数量 |
| `train.checkpointer.period` / `train.eval_period` | 保存与评估周期，按训练模式使用 epoch 或 iteration 为单位 |
| `train.checkpointer.max_to_keep` | 保留的周期检查点数量，默认 `1` |
| `train.log_period` | 日志与 TensorBoard 记录间隔，按训练 step 计 |
| `train.seed` | 指定随机种子，未指定时自动生成 |

例如，在单卡训练中启用混合精度与梯度累积：

```bash
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 1 \
  train.output_dir=./output/faster_rcnn_r50_amp \
  dataloader.train.total_batch_size=2 \
  train.gradient_accumulation_steps=8 \
  train.amp.enable=true \
  train.amp.mixed_precision=fp16
```

此时每次参数更新累计处理 `2 × 8 = 16` 张图片。有效 batch size 为 `total_batch_size × gradient_accumulation_steps`，应结合学习率与训练计划一起调整。

### 输出文件

```text
output/faster_rcnn_r50/
├── config.yaml               # 本次运行的配置快照
├── log.txt                   # 主进程日志
├── events.out.tfevents.*      # TensorBoard 事件
├── last_checkpoint           # 最近检查点的文件名
├── model_0000012.pth          # 周期检查点，编号依配置而定
├── model_final.pth            # 正常完成训练后的最终检查点
└── inference/                # 评估输出
```

周期检查点默认只保留最近一个，最终检查点单独保存。查看训练曲线：

```bash
tensorboard --logdir output
```

## 自定义数据集

COCO 格式的检测数据可以直接注册。下面可作为自定义 Python 配置文件的主体：

```python
from overdrive.data.datasets import register_coco_instances
from overdrive.model_zoo import get_config

register_coco_instances(
    "my_train", {}, "/path/to/train.json", "/path/to/train_images"
)
register_coco_instances(
    "my_val", {}, "/path/to/val.json", "/path/to/val_images"
)

base = get_config("COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py")
model = base.model
dataloader = base.dataloader
optimizer = base.optimizer
lr_multiplier = base.lr_multiplier
train = base.train

dataloader.train.dataset.names = ["my_train"]
dataloader.test.dataset.names = ["my_val"]
model.roi_head.num_classes = 3  # 与 JSON categories 中的类别数一致
train.output_dir = "./output/my_dataset"
```

将该配置保存为 `configs/my_dataset.py`，即可通过 `--config-file configs/my_dataset.py` 传给训练或 Demo 入口。注册代码需要在每个训练、评估或推理进程加载配置时执行；数据集名称不能重复注册。

`register_coco_instances` 会设置 COCO 评估所需的元数据，并在加载标注时将类别 ID 映射到连续索引。训练集与验证集应使用相同的类别定义；实例分割还需提供 `segmentation` 字段并配置 Mask R-CNN 的类别数与 mapper。

其他格式可通过 `DatasetCatalog.register()` 提供返回样本字典列表的函数，通过 `MetadataCatalog` 设置类别和评估信息，必要时实现自己的 `DatasetMapper` 与 `DatasetEvaluator`。相关实现见[数据模块](../../overdrive/data)和[评估模块](../../overdrive/evaluation)。

## 项目结构

```text
OverDrive/
├── configs/                  # 实验配置与 common 共享配置
├── datasets/                 # 数据目录及 ADE20K 标注准备脚本
├── demo/                     # 图片、视频、摄像头推理与可视化
├── docs/                     # 中英文使用指南、进阶指南与常见问题
├── overdrive/
│   ├── checkpoint/           # 权重加载、保存与远程权重目录
│   ├── config/               # LazyConfig、LazyCall 与实例化
│   ├── csrc/                 # C++/CUDA 扩展与 COCO 加速评估
│   ├── data/                 # 数据注册、加载、采样与增强
│   ├── engine/               # 训练器、后端、循环与 Hook
│   ├── evaluation/           # 各任务与数据集评估器
│   ├── layers/               # 通用网络层与算子封装
│   ├── model_zoo/            # 配置和预训练模型访问接口
│   ├── modeling/             # 模型架构及可组合组件
│   ├── ops/                  # 扩展算子的 Python 接口
│   ├── solver/               # 优化器构建与学习率计划
│   ├── structures/           # Boxes、Instances、Masks 等数据结构
│   └── utils/                # 日志、通信、文件访问与可视化工具
├── projects/                 # 独立项目与扩展的组织入口
├── tools/
│   ├── train_net.py          # 通用训练与评估入口
│   ├── visualize_data.py     # 检查原始标注或数据增强结果
│   └── visualize_json_results.py  # 可视化 COCO/LVIS JSON 预测
├── setup.py                  # Python 包与扩展构建
├── README.md                 # 英文项目介绍
├── README_zh.md              # 中文项目介绍
└── LICENSE
```

查看 COCO 原始标注的可视化结果：

```bash
python tools/visualize_data.py \
  --source annotation \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --output-dir output/data_vis
```

`--source dataloader` 会展示训练预处理后的结果；该工具会遍历数据，请按需要停止运行。检查 COCO 评估生成的预测 JSON：

```bash
python tools/visualize_json_results.py \
  --input output/faster_rcnn_r50_eval/inference/coco_instances_results.json \
  --output output/prediction_vis \
  --dataset coco_2017_val \
  --conf-threshold 0.5
```
