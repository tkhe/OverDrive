import copy
import logging
import os
import weakref
from collections import OrderedDict
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel

from overdrive.checkpoint import Checkpointer
from overdrive.checkpoint.dml_checkpoint import DMLLoopState
from overdrive.config import instantiate
from overdrive.config import try_get_key
from overdrive.evaluation import inference_on_dataset
from overdrive.evaluation import print_csv_format
from overdrive.utils import comm
from overdrive.utils.logger import setup_logger

from . import hook
from .backend import MultiOptimizerNativeBackend
from .trainer import Trainer

__all__ = ["DMLTrainer"]


class DMLTrainer(Trainer):
    """Assemble DML training without changing the single-optimizer Trainer API.

    Config uses ``optimizers`` and ``lr_multipliers`` keyed by student name,
    ``train.eval`` for evaluation, and optional ``train.best_checkpointer``
    (``val_metric``, ``mode`` and ``file_prefix``) for per-student best snapshots.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if not logging.getLogger("overdrive").isEnabledFor(logging.INFO):
            setup_logger()
        model = self.build_model(cfg)
        optimizers = self.build_optimizers(cfg, model)
        self.backend = self.build_backend(cfg, model, optimizers, self.build_train_dataloader(cfg))
        self.loop = self.build_loop(cfg, self.backend)
        self.schedulers = self.build_lr_schedulers(cfg)
        if self.schedulers.keys() != optimizers.keys():
            raise ValueError("lr_multipliers must have exactly the student names")
        self.checkpointer = Checkpointer(
            self.model, cfg.train.output_dir, loop=DMLLoopState(self.loop), backend=weakref.proxy(self.backend)
        )
        self.loop.checkpointer = weakref.proxy(self.checkpointer)
        self.loop.register_hooks(self.build_default_hooks(self.schedulers))
        self.loop.register_hooks(self.build_custom_hooks())

    @property
    def optimizers(self):
        return self.backend.optimizers

    @classmethod
    def build_optimizers(cls, cfg, model):
        cfg = copy.deepcopy(cfg)
        if cfg.optimizers.keys() != model.models.keys():
            raise ValueError("optimizers must have exactly the student names")
        optimizers = {}
        owned = set()
        for name in model.models:
            modules = model.student_modules(name)
            parameters = {id(p) for p in modules.parameters() if p.requires_grad}
            if owned & parameters:
                raise ValueError(f"Student {name} shares trainable parameters with another student")
            owned.update(parameters)
            cfg.optimizers[name].params.model = modules
            optimizers[name] = instantiate(cfg.optimizers[name])
        return optimizers

    @classmethod
    def build_lr_schedulers(cls, cfg):
        return {name: instantiate(config) for name, config in cfg.lr_multipliers.items()}

    @classmethod
    def build_backend(cls, cfg, model, optimizers, dataloader):
        backend = try_get_key(cfg, "train.backend", default="native")
        if backend != "native":
            raise ValueError(f"Unsupported DML backend: {backend}")
        return MultiOptimizerNativeBackend(
            model,
            optimizers,
            dataloader,
            ddp=try_get_key(cfg, "train.ddp", default={}),
            amp=try_get_key(cfg, "train.amp", default={}),
            gradient_clipping=try_get_key(cfg, "train.gradient_clipping", default={}),
            gradient_accumulation_steps=try_get_key(cfg, "train.gradient_accumulation_steps", default=1),
        )

    @staticmethod
    def eval_students(cfg, model):
        students = cfg.train.eval.get("eval_student", list(model.models))
        return [students] if isinstance(students, str) else list(students)

    def build_default_hooks(self, schedulers):
        cfg = self.cfg
        scheduler_hook = hook.MultiLRSchedulerHook(schedulers)
        self.schedulers = scheduler_hook.schedulers
        hooks = [hook.IterationTimeHook(), scheduler_hook]

        def evaluate():
            self._last_eval_results = self.test(cfg, self.model)
            return self._last_eval_results

        hooks.append(hook.EvalHook(cfg.train.eval.period, evaluate))
        best = try_get_key(cfg, "train.best_checkpointer", default=None)
        if best is not None:
            hooks.append(hook.DMLBestCheckpointHook(self.eval_students(cfg, self.checkpointer.model), cfg.train.eval.period, **best))
        if comm.is_main_process():
            # Same priority: evaluation -> all best records -> regular snapshot.
            hooks.append(hook.CheckpointHook(self.checkpointer, **cfg.train.checkpointer))
            period = try_get_key(cfg, "train.log_period", default=10)
            hooks.extend(
                [hook.CommonMetricPrinterHook(period), hook.MultiLRLoggerHook(period), hook.TensorboardXHook(period, cfg.train.output_dir)]
            )
        return hooks

    @staticmethod
    def load_initial_weights(cfg, checkpointer, resume=True):
        """Shared train/eval loading policy; return whether training state was loaded."""
        initial = cfg.train.init_checkpoint
        if resume and checkpointer.has_checkpoint():
            checkpointer.load(checkpointer.get_checkpoint_file())
            return True
        checkpoint = initial.get("resume", "")
        if checkpoint:
            checkpointer.load(checkpoint, checkpointables=None if resume else [])
            return resume
        for name, path in initial.get("models", {}).items():
            Checkpointer(checkpointer.model.models[name]).load(path)
        return False

    def resume_or_load(self, resume=True):
        # Loop.load_state_dict restores its own start position. Do not advance twice.
        self.load_initial_weights(self.cfg, self.checkpointer, resume)

    @staticmethod
    def _build_eval_autocast_context(cfg):
        amp = try_get_key(cfg, "train.amp", default={})
        if not amp.get("enable", False):
            return nullcontext()
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp.get("mixed_precision", "fp16")]
        return torch.autocast(device_type=torch.device(cfg.train.device).type, dtype=dtype)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """Evaluate selected students. Optional evaluators are keyed by student/dataset."""
        if isinstance(model, DistributedDataParallel):
            model = model.module
        dataset_names = cfg.dataloader.test.dataset.names
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        results = OrderedDict()
        for name in cls.eval_students(cfg, model):
            student_results = OrderedDict()
            for dataset_name in dataset_names:
                dataloader = cls.build_test_dataloader(cfg, dataset_name)
                output_folder = os.path.join(cfg.train.output_dir, "inference", name, dataset_name)
                evaluator = cls.build_evaluator(cfg, dataset_name, output_folder) if evaluators is None else evaluators[name][dataset_name]
                with cls._build_eval_autocast_context(cfg):
                    metrics = inference_on_dataset(model.models[name], dataloader, evaluator)
                student_results[dataset_name] = metrics
                if comm.is_main_process():
                    logging.getLogger(__name__).info("Evaluation results for %s/%s:", name, dataset_name)
                    print_csv_format(metrics)
            results[name] = next(iter(student_results.values())) if len(student_results) == 1 else student_results
        return results

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        raise NotImplementedError("Provide a dataset evaluator in a DMLTrainer subclass (see tools/train_dml.py)")
