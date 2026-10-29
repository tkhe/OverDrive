import logging
import os

import torch

from overdrive.utils import comm
from overdrive.utils.file_io import PathManager

from .evaluator import DatasetEvaluator

__all__ = ["ClassificationEvaluator"]


class ClassificationEvaluator(DatasetEvaluator):
    """Evaluate single-label classification with Top-k accuracy percentages.

    Args:
        topk: Non-empty sequence of positive integers, each no greater than the
            number of classes. Use e.g. ``(1,)`` for datasets with fewer than five
            classes.
        distributed: Collect counts from all ranks and report on the main process.
            If False, evaluate only the current process.
        output_dir: Optional directory for ``classification_evaluation.pth``.
    """

    def __init__(self, topk=(1, 5), distributed=True, output_dir=None):
        self._topk = tuple(topk)
        self._distributed = distributed
        self._output_dir = output_dir
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._num_samples = 0
        self._num_correct = {k: 0 for k in self._topk}

    def process(self, inputs, outputs):
        """Accumulate counts from input ``label`` and output ``pred_scores``.

        Each label is a Python integer or integer scalar tensor; each score
        vector has one entry per class. Evaluation inputs must retain labels.
        """
        for input_per_image, output_per_image in zip(inputs, outputs, strict=True):
            scores = output_per_image["pred_scores"]
            indices = scores.topk(max(self._topk)).indices
            label = torch.as_tensor(input_per_image["label"], device=indices.device)
            correct = indices == label
            for k in self._num_correct:
                self._num_correct[k] += correct[:k].any().item()
            self._num_samples += 1

    def evaluate(self):
        num_samples = self._num_samples
        num_correct = self._num_correct
        if self._distributed:
            comm.synchronize()
            counts = comm.gather((num_samples, num_correct), dst=0)
            if not comm.is_main_process():
                return None
            num_samples = sum(count for count, _ in counts)
            num_correct = {k: sum(correct[k] for _, correct in counts) for k in self._topk}

        if num_samples == 0:
            self._logger.warning("[ClassificationEvaluator] Did not receive any predictions.")
            return {}

        metrics = {f"top{k}": 100.0 * num_correct[k] / num_samples for k in self._topk}
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "classification_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(metrics, f)

        results = {"classification": metrics}
        self._logger.info(results)
        return results
