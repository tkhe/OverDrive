# 常见问题

[返回 README](../../README_zh.md) | [English](../en/faq.md) | 简体中文

| 现象 | 排查方式 |
| --- | --- |
| 安装时提示找不到 `torch` | 先在当前虚拟环境安装 PyTorch，再使用 `--no-build-isolation` 安装 OverDrive |
| `torchvision::nms` 不存在或 ROIAlign 加载失败 | 检查 PyTorch、TorchVision 版本及 CPU/CUDA 构建是否匹配，并在同一环境重新安装 |
| 无法导入 `overdrive._C` 或出现符号解析错误 | 检查编译工具链；更换 PyTorch 或 CUDA 后重新构建扩展。缺少扩展时 COCO 评估会回退到 `pycocotools` 实现 |
| `torch.cuda.is_available()` 为 `False` | 检查驱动、PyTorch 构建与设备可见性；CPU 推理设置 `train.device=cpu`，关闭训练和评估 AMP |
| 找不到图片或标注 | 检查 `OVERDRIVE_DATASETS`、实际目录结构和启动位置；ADE20K 需先生成 `annotations_overdrive` |
| 自定义数据集未注册 | 将注册代码放在配置或入口加载过程中，确保所有进程在构建 dataloader 前执行 |
| 显存不足 | 减小全局 batch size 或输入分辨率，启用 AMP，按需要使用梯度累积，并相应调整学习率 |
| 远程初始化权重下载失败 | 检查网络与权重目录条目，或将 `train.init_checkpoint` 设置为本地文件；缓存目录可通过 `OVERDRIVE_CACHE` 指定，默认为 `~/.cache/overdrive_cache` |
| 权重加载提示参数缺失或形状不匹配 | 骨干网络初始化时任务头缺失可以是预期行为；完整模型评估时需确认架构、类别数和权重匹配 |
| 无图形界面的服务器无法打开 Demo 窗口 | 图片或视频推理指定 `--output` 保存结果，并提前创建输出目录 |
| 命令行覆盖参数报错 | 使用 `key=value`；训练参数放在命令末尾，Demo 参数放在末尾的 `--opts` 后，列表等值可用单引号包住 |

遇到问题时，可以附上环境信息、完整命令、所用配置和错误日志，在[项目 Issues](https://github.com/tkhe/OverDrive/issues)中反馈。
