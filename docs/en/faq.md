# FAQ

[Back to README](../../README.md) | English | [简体中文](../zh/faq.md)

| Symptom | What to check |
| --- | --- |
| Installation cannot find `torch` | Install PyTorch in the current virtual environment first, then install OverDrive with `--no-build-isolation` |
| `torchvision::nms` is missing or ROIAlign cannot load | Check that PyTorch and TorchVision versions and CPU/CUDA builds match, then reinstall them in the same environment |
| `overdrive._C` cannot be imported or reports unresolved symbols | Check the compiler toolchain and rebuild the extension after changing PyTorch or CUDA. COCO evaluation falls back to `pycocotools` when the extension is unavailable |
| `torch.cuda.is_available()` returns `False` | Check the driver, PyTorch build, and device visibility. For CPU inference, set `train.device=cpu`; disable AMP for CPU training and evaluation |
| Images or annotations cannot be found | Check `OVERDRIVE_DATASETS`, the directory layout, and the working directory. ADE20K requires the converted `annotations_overdrive` directory |
| A custom dataset is not registered | Run registration while loading the configuration or entry point, before each process builds its dataloader |
| GPU runs out of memory | Reduce the global batch size or input resolution, enable AMP, or use gradient accumulation; adjust the learning rate accordingly |
| Remote initialization weights cannot be downloaded | Check network access and the weight catalog entry, or set `train.init_checkpoint` to a local file. Set `OVERDRIVE_CACHE` to change the cache directory; the default is `~/.cache/overdrive_cache` |
| Missing parameters or mismatched shapes when loading weights | Missing task heads can be expected during backbone initialization. For evaluation with a complete model checkpoint, ensure the architecture, class counts, and weights match |
| Demo cannot open a window on a server without a display | Use `--output` for image or video inference and create the output directory beforehand |
| Command-line configuration overrides fail | Use `key=value`; place training overrides at the end and Demo overrides after the final `--opts`. Quote values such as lists with single quotes |

When reporting a problem in the [project issue tracker](https://github.com/tkhe/OverDrive/issues), include environment information, the full command, the configuration, and the error log.
