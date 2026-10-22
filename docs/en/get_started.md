# User Guide

[Back to README](../../README.md) | English | [简体中文](../zh/get_started.md)

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Getting Started](#getting-started)

## Installation

### Requirements

| Dependency | Requirement or notes |
| --- | --- |
| Python | Use **3.10 or newer**; |
| PyTorch | **2.6 or newer**, checked during installation and import |
| TorchVision | A version compatible with PyTorch; provides operators such as NMS and ROIAlign and must be installed separately first |
| C++ compiler | A toolchain compatible with PyTorch is required to build the `overdrive._C` extension during installation |
| CUDA | GPU training requires an available NVIDIA GPU and driver; building the CUDA extension also requires a compatible CUDA Toolkit and `nvcc` |

Linux with NVIDIA CUDA is the primary environment for GPU training. Configurations default to `train.device=cuda`; set it to `cpu` for CPU inference or debugging. The examples use Bash/Zsh and should be run from the repository root.

### Install from Source

```bash
git clone https://github.com/tkhe/OverDrive.git
cd OverDrive

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install ninja
```

Install PyTorch and TorchVision first. Use the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to select the command for your operating system and compute platform, ensuring PyTorch >= 2.6. On macOS, you can use the following command; for Linux CPU/CUDA environments, use the package index selected by the guide:

```bash
python -m pip install "torch>=2.6" torchvision
```

Then install OverDrive:

```bash
python -m pip install -e . --no-build-isolation
```

`setup.py` imports PyTorch during the build, so install it first and use `--no-build-isolation` to build with the current environment. Installation compiles the C++ extension and links or copies `configs/` to `overdrive/model_zoo/configs/` for the Model Zoo configuration API.

Install optional dependencies as needed:

```bash
# Dependencies for optional features such as LVIS, Cityscapes, and panoptic segmentation
python -m pip install -e ".[all]" --no-build-isolation
```

The `all` extra includes `panopticapi`, installed from GitHub, and requires access to that resource. Standard COCO detection training does not require all optional dependencies.

### Verify Installation

```bash
python -c "import torch, torchvision, overdrive; print('PyTorch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('OverDrive:', overdrive.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "from overdrive import _C; print('Compiler:', _C.get_compiler_version()); print('Extension CUDA:', _C.get_cuda_version())"
python -c "from overdrive.utils.collect_env import collect_env_info; print(collect_env_info())"
```

An extension built for CPU reports CUDA as unavailable. This is a separate check from whether PyTorch itself can use a GPU.

## Dataset Preparation

Built-in datasets are read from `datasets/` under the repository root by default. To use an external directory, set it **before** starting Python:

```bash
export OVERDRIVE_DATASETS=/path/to/datasets
```

Dataset paths are registered at import time. If you change this variable in a running Python process, register the datasets again or restart the process. Dataset files are not included in the project; download them separately and arrange them as shown below.

### COCO 2017

Object detection and instance segmentation use the same images and instance annotations:

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
        ├── person_keypoints_train2017.json  # Required for keypoint detection
        └── person_keypoints_val2017.json    # Required for keypoint detection
```

The default dataset names are `coco_2017_train` and `coco_2017_val`. Keypoint configurations use `keypoints_coco_2017_train` and `keypoints_coco_2017_val`. Image inference only requires a configuration, a checkpoint containing weights for the complete model, and input images; the COCO training images and annotations are not needed.

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

Convert the annotations:

```bash
python datasets/prepare_ade20k_sem_seg.py
```

The script also reads `OVERDRIVE_DATASETS`. It creates `annotations_overdrive/training/` and `annotations_overdrive/validation/`, converts labels `1–150` to `0–149`, and maps label `0` to the ignore label `255`. The built-in datasets `ade20k_sem_seg_train` and `ade20k_sem_seg_val` use these converted annotations.

See [builtin.py](../../overdrive/data/datasets/builtin.py) and the [dataset implementations](../../overdrive/data/datasets) for other dataset names and directory layouts. Panoptic segmentation requires its corresponding annotations. Dataset interfaces and evaluators are available for more datasets than the supplied training configurations cover.

## Getting Started

The examples below use Faster R-CNN with ResNet-50 FPN. Training and evaluation require the COCO dataset prepared above. Inference requires weights for the complete model that match the configuration.

### Training on One GPU

```bash
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 1 \
  train.output_dir=./output/faster_rcnn_r50 \
  dataloader.train.total_batch_size=2 \
  optimizer.lr=0.0025
```

The original configuration uses a global batch size of `16`, a learning rate of `0.02`, and `12` training epochs. This example reduces the batch size to `2` and scales the learning rate proportionally as a starting point for one GPU. Adjust these settings for your GPU memory and experiment. The framework does not automatically scale the learning rate with the number of GPUs.

The configuration initializes the backbone from `torchvision://ImageNet/ResNet-50`. On the first run, the project resolves this URI through its weight catalog, downloads the checkpoint, and caches it. For offline use, set `train.init_checkpoint` to a local file. To train from random initialization, append `train.init_checkpoint='""'`.

### Training on Multiple GPUs or Machines

Train on eight GPUs on one machine using the original global batch size and learning rate:

```bash
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 8 \
  --dist-url auto \
  train.output_dir=./output/faster_rcnn_r50_8gpu
```

`--num-gpus` is the number of GPUs **per machine**. The global batch size must be divisible by the total number of training processes. For multiple machines, run the same command on each machine with a different `--machine-rank`:

```bash
# Node 0: replace 192.168.1.10 with an address of node 0 reachable from other nodes
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 4 --num-machines 2 --machine-rank 0 \
  --dist-url tcp://192.168.1.10:29500 \
  train.output_dir=./output/faster_rcnn_r50_2nodes

# Node 1
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 4 --num-machines 2 --machine-rank 1 \
  --dist-url tcp://192.168.1.10:29500 \
  train.output_dir=./output/faster_rcnn_r50_2nodes
```

Use the same code and configuration on all nodes, with accessible dataset paths and a reachable initialization address and port. `--dist-url auto` is only supported for one machine.

### Resuming Training

```bash
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 1 \
  --resume \
  train.output_dir=./output/faster_rcnn_r50 \
  dataloader.train.total_batch_size=2 \
  optimizer.lr=0.0025
```

`--resume` first loads the checkpoint referenced by `last_checkpoint` in the output directory and restores the model and training state. If no checkpoint exists there, it loads `train.init_checkpoint`. Keep the configuration and settings consistent with the original experiment when resuming. Without `--resume`, initialization only loads model weights.

### Evaluating a Model

After training, evaluate the generated `model_final.pth`:

```bash
python tools/train_net.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --num-gpus 1 \
  --eval-only \
  train.init_checkpoint=./output/faster_rcnn_r50/model_final.pth \
  train.output_dir=./output/faster_rcnn_r50_eval
```

The evaluator is selected using the test dataset's `evaluator_type`. Metrics are written to the log, while files such as COCO predictions are saved under `inference/` in the output directory. To evaluate the latest saved checkpoint, use `--eval-only --resume` with the original output directory.

### Image, Video, and Webcam Inference

Save a visualization for one image:

```bash
mkdir -p output/demo
python demo/demo.py \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --input /path/to/image.jpg \
  --output output/demo \
  --confidence-threshold 0.5 \
  --opts train.init_checkpoint=./output/faster_rcnn_r50/model_final.pth
```

For multiple images, use `--input '/path/to/images/*.jpg'` and create the output directory beforehand. For a video, replace `--input` with `--video-input /path/to/video.mp4`. For a webcam, use `--webcam` and omit `--output`. When no output path is provided, results are displayed in an OpenCV window; press `Esc` to exit.

Demo configuration overrides use `key=value` and must appear after `--opts` at the end of the command. Append `train.device=cpu` for CPU inference. Use `--precision fp32|fp16|bf16` to select inference precision; the default is `fp32`. `fp16` requires CUDA, and `bf16` requires support from the device and operators.

### Python Inference API

```python
import cv2

from overdrive.config import LazyConfig
from overdrive.engine import DefaultPredictor

cfg = LazyConfig.load(
    "configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py"
)
cfg.train.init_checkpoint = "./output/faster_rcnn_r50/model_final.pth"
cfg.train.device = "cuda"  # Use "cpu" for CPU inference

predictor = DefaultPredictor(cfg)
image = cv2.imread("/path/to/image.jpg")  # Input uses BGR channel order
if image is None:
    raise FileNotFoundError("/path/to/image.jpg")

predictions = predictor(image)
instances = predictions["instances"].to("cpu")
print(instances.pred_boxes.tensor)
print(instances.scores)
print(instances.pred_classes)
```

Detection, instance segmentation, and keypoint models return `instances`. Instance segmentation adds `pred_masks`, and keypoint detection adds `pred_keypoints`. Semantic segmentation models return `sem_seg`; take `argmax` along the class dimension to obtain a label map. `DefaultPredictor` loads the checkpoint and applies the image preprocessing defined by the test configuration.
