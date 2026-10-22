# Advanced Guide

[Back to README](../../README.md) | English | [简体中文](../zh/advanced_usage.md)

- [Advanced Guide](#advanced-guide)
  - [Configuration and Experiment Management](#configuration-and-experiment-management)
    - [Configuration Layout](#configuration-layout)
    - [Common Overrides](#common-overrides)
    - [Output Files](#output-files)
  - [Custom Datasets](#custom-datasets)
  - [Project Structure](#project-structure)

## Configuration and Experiment Management

### Configuration Layout

[configs/common](../../configs/common) contains shared model, data, optimizer, training, and learning rate configurations. Task-specific configurations combine these into experiments. Paths passed to the Model Zoo configuration API are relative to `configs/`:

```python
from overdrive.config import instantiate
from overdrive.model_zoo import get_config

cfg = get_config("COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py")
cfg.model.roi_head.num_classes = 3
cfg.dataloader.train.total_batch_size = 4
cfg.train.output_dir = "./output/my_experiment"

model = instantiate(cfg.model)  # Constructs the model without loading weights
```

`LazyConfig.load()` returns a configuration object, `LazyCall` describes a class or function call, and `instantiate()` recursively constructs the objects. Class counts, output channels, and thresholds live in different fields for different models; check the corresponding shared model configuration before changing them.

### Common Overrides

Place training overrides at the end of the command and Demo overrides after `--opts`. Both use `path.key=value`.

| Configuration key | Purpose |
| --- | --- |
| `train.output_dir` | Output directory for logs, configuration snapshots, checkpoints, and TensorBoard events |
| `train.init_checkpoint` | Initial weights: a local path or a registered weight URI |
| `train.device` | Model device; defaults to `cuda` |
| `train.mode` | `epoch` or `iteration` |
| `train.max_epoch` / `train.max_iter` | Training length for the selected mode; update the learning rate schedule accordingly |
| `dataloader.train.total_batch_size` | Total batch size across all training processes |
| `dataloader.train.num_workers` | Data loading workers per process |
| `optimizer.lr` | Optimizer learning rate |
| `train.amp.enable` | Enable CUDA mixed precision for training and evaluation |
| `train.amp.mixed_precision` | `fp16` or `bf16`; defaults to `fp16` |
| `train.gradient_accumulation_steps` | Number of batches accumulated for each parameter update |
| `train.checkpointer.period` / `train.eval_period` | Checkpoint and evaluation intervals, measured in epochs or iterations according to the training mode |
| `train.checkpointer.max_to_keep` | Number of periodic checkpoints to retain; defaults to `1` |
| `train.log_period` | Logging and TensorBoard interval, measured in training steps |
| `train.seed` | Random seed; generated automatically when not specified |

For example, enable mixed precision and gradient accumulation on one GPU:

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

Each parameter update accumulates `2 × 8 = 16` images. The effective batch size is `total_batch_size × gradient_accumulation_steps`; adjust the learning rate and training schedule accordingly.

### Output Files

```text
output/faster_rcnn_r50/
├── config.yaml               # Configuration snapshot for the run
├── log.txt                   # Main process log
├── events.out.tfevents.*      # TensorBoard events
├── last_checkpoint           # Filename of the latest checkpoint
├── model_0000012.pth          # Periodic checkpoint; numbering depends on the configuration
├── model_final.pth            # Final checkpoint after training completes normally
└── inference/                # Evaluation outputs
```

By default, only the latest periodic checkpoint is retained. The final checkpoint is saved separately. View training curves with:

```bash
tensorboard --logdir output
```

## Custom Datasets

Register detection data in COCO format directly. The following can serve as the body of a custom Python configuration file:

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
model.roi_head.num_classes = 3  # Match the number of classes in the JSON categories
train.output_dir = "./output/my_dataset"
```

Save the configuration as `configs/my_dataset.py` and pass `--config-file configs/my_dataset.py` to the training or Demo entry point. Registration must run when each training, evaluation, or inference process loads the configuration. Dataset names cannot be registered twice.

`register_coco_instances` sets the metadata needed for COCO evaluation and maps category IDs to contiguous indices when loading annotations. Use the same category definitions in the training and validation sets. For instance segmentation, also provide the `segmentation` field and configure the Mask R-CNN class counts and mapper.

For other formats, use `DatasetCatalog.register()` with a function that returns a list of sample dictionaries, set class and evaluation metadata through `MetadataCatalog`, and implement a custom `DatasetMapper` or `DatasetEvaluator` as needed. See the [data module](../../overdrive/data) and [evaluation module](../../overdrive/evaluation).

## Project Structure

```text
OverDrive/
├── configs/                  # Experiment configurations and shared common configurations
├── datasets/                 # Dataset directory and ADE20K annotation preparation script
├── demo/                     # Image, video, and webcam inference visualization
├── docs/                     # User guides, advanced guides, and FAQ in English and Chinese
├── overdrive/
│   ├── checkpoint/           # Checkpoint loading, saving, and remote weight catalogs
│   ├── config/               # LazyConfig, LazyCall, and instantiation
│   ├── csrc/                 # C++/CUDA extension and accelerated COCO evaluation
│   ├── data/                 # Dataset registration, loading, sampling, and augmentation
│   ├── engine/               # Trainer, backend, loops, and hooks
│   ├── evaluation/           # Task and dataset evaluators
│   ├── layers/               # Reusable network layers and operator wrappers
│   ├── model_zoo/            # Configuration and pretrained model access APIs
│   ├── modeling/             # Model architectures and composable components
│   ├── ops/                  # Python interfaces for extension operators
│   ├── solver/               # Optimizer construction and learning rate schedules
│   ├── structures/           # Data structures such as Boxes, Instances, and Masks
│   └── utils/                # Logging, communication, file access, and visualization
├── projects/                 # Location for standalone projects and extensions
├── tools/
│   ├── train_net.py          # Training and evaluation entry point
│   ├── visualize_data.py     # Inspect annotations or augmented training samples
│   └── visualize_json_results.py  # Visualize COCO/LVIS JSON predictions
├── setup.py                  # Python package and extension build
├── README.md                 # Project overview in English
├── README_zh.md              # Project overview in Chinese
└── LICENSE
```

Visualize original COCO annotations:

```bash
python tools/visualize_data.py \
  --source annotation \
  --config-file configs/COCO-Detection/faster_rcnn/faster_rcnn_resnet50_fpn_1x.py \
  --output-dir output/data_vis
```

Use `--source dataloader` to inspect samples after training preprocessing. The tool traverses the dataset; stop it when you have inspected enough samples. To visualize predictions from COCO evaluation:

```bash
python tools/visualize_json_results.py \
  --input output/faster_rcnn_r50_eval/inference/coco_instances_results.json \
  --output output/prediction_vis \
  --dataset coco_2017_val \
  --conf-threshold 0.5
```
