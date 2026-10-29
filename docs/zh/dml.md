# DML 多学生训练

使用独立入口运行三学生 RetinaNet 示例：

```bash
python tools/train_dml.py --num-gpus 1 \
  --config-file configs/COCO-Detection/retinanet/dml/retinanet_res18_res50_res101_1x.py
```

同目录的 `retinanet_res18_res50_res101_sparse_1x.py` 演示：res18 学习 res50 的 logits、res101 的 FPN 特征；res50 只学习 res18；res101 只计算 GT loss。示例的 logits 使用空间 CWD；类别 KL 可通过 `KLDivergence` 和合适的 recorder/connector 配置。

## 模型与互学习关系

`L(DML)(models={学生名: 完整模型配置}, distiller=L(DMLDistiller)(...))` 中至少有两个学生。学生训练前向返回 GT loss 字典，推理前向返回原生预测。学生共享 batch、AMP 配置和梯度累计次数，参数独立。

蒸馏配置的层级是：

```python
recorders[模型名][记录器名]
distill_losses[学习者][知识提供者][损失名]
loss_forward_mappings[学习者][知识提供者][损失名]
connectors[学习者][连接器名]
```

每个模型只配置一次 recorder。损失映射沿用 `from_student`、`recorder`、`record_idx`、`data_idx`、`connector`、`connector_idx`。`from_student=True` 读取学习者，False 读取当前同伴。模型内部的采集位置、空间尺寸和类别语义由配置负责对齐。

每个学生每个 micro-batch 前向一次；所有前向完成后才计算互学习损失。知识提供者的张量在进入 connector 前 detach，因此学习者的损失不会反向更新同伴。connector 和可训练损失模块属于学习者的优化器。不同学生之间不要共享参数化模块。

每个学生的总损失是 GT loss 加配置同伴的平均蒸馏损失。未配置的关系不补齐；没有同伴时只计算 GT。loss 名称沿用仓库约定，包含 `loss`，日志指标使用 `[metric]`，避免进入反向求和。

## 优化、评估与保存

`optimizers` 和 `lr_multipliers` 必须以全部学生名称为键。优化器参数构建沿用 `get_default_optimizer_params`，收到的模型视图包含 `model` 和 `distiller` 两个子模块，因此自定义参数名称规则应包含对应前缀。每个优化器同时管理该学生自己的蒸馏模块。`lr_multipliers` 使用现有 `ParamScheduler` 配置。

`train.gradient_clipping[学生名]` 使用原有 `enable/type/value/norm` 字段，未配置的学生不裁剪。`train.amp`、`train.ddp`、`train.gradient_accumulation_steps` 共享。整个 DML 使用一个 DDP；一次完整累计窗口是一个 Loop step，各 scheduler 每个 step 推进一次，包括 AMP 跳过某个优化器更新的情况。FP16 共享 scaler，各优化器独立处理梯度溢出。

```python
train.eval = dict(period=1, eval_student=["res18", "res50"])
train.best_checkpointer = dict(val_metric="bbox/AP", mode="max")
```

`eval_student` 可为字符串或列表；省略时评估全部学生，空列表不评估。`period` 跟随 Loop 的 epoch/iteration 单位；与现有 EvalHook 一样，训练正常结束时仍执行最终评估。DML 使用 `train.eval.period`，不读取旧的 `train.eval_period`。

指标为 `res18/bbox/AP`；多数据集时为 `res18/<数据集>/bbox/AP`，此时 `val_metric` 配置为 `<数据集>/bbox/AP`。各学生使用独立的 `inference/<学生>/<数据集>` 输出目录。可选 `train.best_checkpointer` 为每个评估学生保存 `model_best_<学生>.pth`；不配置该字段则只定期保存。不要继承普通配置中默认的单模型 BestCheckpointHook。

常规与最佳文件均包含全部模型、蒸馏模块、优化器、scheduler、scaler、Loop 和最佳指标状态。最佳文件不更新 `last_checkpoint`，正常恢复入口仍指向最近常规文件。评估、最佳指标更新、常规保存依次执行。

```python
train.init_checkpoint = dict(
    resume="",  # 完整 DML checkpoint
    models={"res18": "...", "res50": "...", "res101": "..."},
)
```

使用 `--resume` 时，优先加载输出目录的最近 checkpoint，其次加载显式 `resume` 路径并恢复训练状态；两者都没有时按 `models` 初始化学生。不开启 `--resume` 时，显式整体 checkpoint 只加载模型权重，否则按学生初始化。模型初始化映射可省略学生，让它随机初始化。显式路径错误直接报错。恢复以完整累计窗口/epoch checkpoint 为边界，沿用现有 Loop 和数据迭代语义，不保存 micro-batch 中途梯度或数据加载器游标。

`--eval-only` 使用相同加载与学生选择规则。单独部署时使用 `model.models[name]`；DML 容器的推理前向返回所有学生预测的字典，不进行集成。

## 验证

```bash
python -m unittest discover -s tests -p 'test_dml*.py' -v
```

测试覆盖梯度隔离、稀疏关系、独立优化/裁剪、累计、checkpoint 恢复、评估选择及双进程 CPU/Gloo DDP。CUDA AMP 测试仅在 CUDA 可用时执行。
