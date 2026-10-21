import datetime
import logging
import time
from functools import cached_property

import torch

from overdrive.utils.events import get_event_storage

from .base import Hook
from .priority import HookPriority

__all__ = [
    "CommonMetricPrinterHook",
    "TensorboardXHook",
]


class CommonMetricPrinterHook(Hook):
    """
    Periodically log metrics collected in the active event storage.

    The output includes losses, metrics, learning rate, timing, ETA, and GPU
    memory usage when available. Epoch and iteration loops use their respective
    progress format.
    """

    def __init__(self, period: int = 10, window_size: int = 20):
        """
        Initialize the metric printer.

        Args:
            period: Number of steps between log messages.
            window_size: Number of recent samples used for smoothed metrics.
        """
        self.period = period
        self.window_size = window_size
        self.logger = logging.getLogger("overdrive")
        self._last_write = None

    def after_step(self):
        """
        Write metrics at the configured interval or at the final step.
        """
        if self.loop.mode == "epoch":
            if (self.loop.step_in_epoch + 1) % self.period == 0 or (self.loop.step_in_epoch + 1 == self.loop.max_step_in_epoch):
                self.write_by_epoch()
        elif self.loop.mode == "iteration":
            if (self.loop.iter + 1) % self.period == 0 or (self.loop.iter + 1 == self.loop.max_iter):
                self.write_by_iteration()

    def get_eta(self, storage):
        """
        Estimate the remaining training time from recorded step durations.

        Returns:
            A human-readable ETA string, or ``None`` when no estimate exists.
        """
        step_done = self.loop.iter + 1 if self.loop.mode == "iteration" else self.loop.epoch * self.loop.max_step_in_epoch + self.loop.step_in_epoch + 1
        max_step = self.loop.max_iter if self.loop.mode == "iteration" else self.loop.max_epoch * self.loop.max_step_in_epoch

        try:
            eta_seconds = storage.history("time").median(1000) * (max_step - step_done)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            # estimate eta on our own - more noisy
            eta_string = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (step_done - self._last_write[0])
                eta_seconds = estimate_iter_time * (max_step - step_done - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (step_done, time.perf_counter())
            return eta_string

    def write_by_epoch(self):
        """
        Format and log progress for an epoch-based training loop.
        """
        storage = get_event_storage()

        epoch = self.loop.epoch + 1
        max_epoch = self.loop.max_epoch
        step_in_epoch = self.loop.step_in_epoch + 1
        max_step_in_epoch = self.loop.max_step_in_epoch

        try:
            avg_data_time = storage.history("data_time").avg(storage.count_samples("data_time", self.window_size))
            last_data_time = storage.history("data_time").latest()
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            avg_data_time = None
            last_data_time = None

        try:
            avg_iter_time = storage.history("time").global_avg()
            last_iter_time = storage.history("time").latest()
        except KeyError:
            avg_iter_time = None
            last_iter_time = None

        try:
            lr = f"{storage.history('lr').latest():.5g}"
        except KeyError:
            lr = "N/A"

        eta_string = self.get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        progress_bar = f"Epoch: [{epoch}/{max_epoch}] Step: [{step_in_epoch}/{max_step_in_epoch}]"
        losses = "  ".join([f"{k}: {v.median(storage.count_samples(k, self.window_size)):.4g}" for k, v in storage.histories().items() if "loss" in k])
        metrics = "  ".join([f"{k}: {v.median(storage.count_samples(k, self.window_size)):.4g}" for k, v in storage.histories().items() if "[metric]" in k])
        avg_time = f"time: {avg_iter_time:.4f}" if avg_iter_time is not None else ""
        last_time = f"last_time: {last_iter_time:.4f}" if last_iter_time is not None else ""
        avg_data_time = f"data_time: {avg_data_time:.4f}" if avg_data_time is not None else ""
        last_data_time = f"last_data_time: {last_data_time:.4f}" if last_data_time is not None else ""
        lr = f"lr: {lr}"
        eta = f"ETA: {eta_string}" if eta_string else ""
        memory = f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else ""

        self.logger.info("  ".join([progress_bar, losses, metrics, avg_time, last_time, avg_data_time, last_data_time, lr, eta, memory]))

    def write_by_iteration(self):
        """
        Format and log progress for an iteration-based training loop.
        """
        storage = get_event_storage()

        iter = self.loop.iter + 1
        max_iter = self.loop.max_iter

        try:
            avg_data_time = storage.history("data_time").avg(storage.count_samples("data_time", self.window_size))
            last_data_time = storage.history("data_time").latest()
        except KeyError:
            avg_data_time = None
            last_data_time = None

        try:
            avg_iter_time = storage.history("time").global_avg()
            last_iter_time = storage.history("time").latest()
        except KeyError:
            avg_iter_time = None
            last_iter_time = None

        try:
            lr = f"{storage.history('lr').latest():.5g}"
        except KeyError:
            lr = "N/A"

        eta_string = self.get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        progress_bar = f"Iter: [{iter}/{max_iter}]"
        losses = "  ".join([f"{k}: {v.median(storage.count_samples(k, self.window_size)):.4g}" for k, v in storage.histories().items() if "loss" in k])
        metrics = "  ".join([f"{k}: {v.median(storage.count_samples(k, self.window_size)):.4g}" for k, v in storage.histories().items() if "[metric]" in k])
        avg_time = f"time: {avg_iter_time:.4f}" if avg_iter_time is not None else ""
        last_time = f"last_time: {last_iter_time:.4f}" if last_iter_time is not None else ""
        avg_data_time = f"data_time: {avg_data_time:.4f}" if avg_data_time is not None else ""
        last_data_time = f"last_data_time: {last_data_time:.4f}" if last_data_time is not None else ""
        lr = f"lr: {lr}"
        eta = f"ETA: {eta_string}" if eta_string else ""
        memory = f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else ""

        self.logger.info("  ".join([progress_bar, losses, metrics, avg_time, last_time, avg_data_time, last_data_time, lr, eta, memory]))

    @property
    def priority(self):
        """
        Run with normal hook priority.
        """
        return HookPriority.NORMAL


class TensorboardXHook(Hook):
    """
    Write event storage scalars, images, and histograms to TensorBoard.
    """

    def __init__(self, period: int, log_dir: str, window_size: int = 20):
        """
        Initialize the TensorBoard writer schedule.

        Args:
            period: Number of steps between TensorBoard writes.
            log_dir: Directory where TensorBoard event files are stored.
            window_size: Smoothing window used for scalar values.
        """
        super().__init__()

        self.log_dir = log_dir
        self.period = period
        self.window_size = window_size
        self.last_write = -1

    @cached_property
    def writer(self):
        """
        Lazily construct and cache the TensorBoard summary writer.
        """
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=self.log_dir)

    def after_step(self):
        """
        Flush event storage at the configured interval or final step.
        """
        if self.loop.mode == "epoch":
            if (self.loop.step_in_epoch + 1) % self.period == 0 or (self.loop.step_in_epoch + 1 == self.loop.max_step_in_epoch):
                self.write()
        elif self.loop.mode == "iteration":
            if (self.loop.iter + 1) % self.period == 0 or (self.loop.iter + 1 == self.loop.max_iter):
                self.write()

    def after_train(self):
        """
        Write remaining events and close the TensorBoard writer.
        """
        self.write()
        self.close()

    def write(self):
        """
        Write new scalars, images, and histograms from event storage.
        """
        storage = get_event_storage()

        new_last_write = self.last_write
        for k, (v, step) in storage.latest_with_smoothing_hint(self.window_size).items():
            if step > self.last_write:
                self.writer.add_scalar(k, v, step)
                new_last_write = max(new_last_write, step)
        self.last_write = new_last_write

        if len(storage._vis_data) >= 1:
            for image_name, image, step_num in storage._vis_data:
                self.writer.add_image(image_name, image, step_num)
            storage.clear_images()

        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self.writer.add_histogram_raw(**params)
            storage.clear_histograms()

    def close(self):
        """
        Close the writer if it has been initialized.
        """
        if "writer" in self.__dict__:
            self.writer.close()

    @property
    def priority(self):
        """
        Use a lower priority than EvalHook so evaluation metrics are available
        in event storage before they are written to TensorBoard.
        """
        return HookPriority.VERY_LOW
