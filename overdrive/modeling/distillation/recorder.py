from abc import ABC
from abc import abstractmethod

import torch

__all__ = [
    "Recorder",
    "ModuleInputsRecorder",
    "ModuleOutputsRecorder",
    "RecorderManager",
]


def _snapshot(data):
    # clone preserves autograd while protecting records from later inplace ops.
    if isinstance(data, torch.Tensor):
        return data.clone()
    if isinstance(data, dict):
        return {key: _snapshot(value) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_snapshot(value) for value in data)
    if isinstance(data, list):
        return [_snapshot(value) for value in data]
    return data


class Recorder(ABC):
    """
    Base class for recording module data during a context.

    ``source`` is a path accepted by ``model.get_submodule`` (empty for root).
    Successful context exit stops recording but keeps records for loss mapping.
    Call ``clear`` after consuming records, or ``close`` to also remove the hook.
    """

    def __init__(self, source: str):
        self.source = source
        self.data_buffer = []
        self._recording = False
        self._handle = None

    def initialize(self, model):
        self.close()
        self._handle = self._register_hook(model.get_submodule(self.source))

    @abstractmethod
    def _register_hook(self, module):
        raise NotImplementedError

    def get_record_data(self, record_idx=0, data_idx=None):
        data = self.data_buffer[record_idx]
        return data if data_idx is None else data[data_idx]

    def clear(self):
        self.data_buffer.clear()

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._recording = False
        self.clear()

    def __enter__(self):
        self.clear()
        self._recording = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._recording = False
        if exc_type is not None:
            self.clear()


class ModuleOutputsRecorder(Recorder):
    """
    Record module outputs during a context, including repeated calls.
    """

    def _register_hook(self, module):
        return module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self._recording:
            self.data_buffer.append(_snapshot(output))


class ModuleInputsRecorder(Recorder):
    """
    Record positional inputs before the module can modify them inplace.
    """

    def _register_hook(self, module):
        return module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        if self._recording:
            self.data_buffer.append(_snapshot(inputs))


class RecorderManager:
    """
    Named recorder instances bound to one model.
    """

    def __init__(self, recorders=None):
        self.recorders = dict(recorders or {})

    def initialize(self, model):
        for recorder in self.recorders.values():
            recorder.initialize(model)

    def get_recorder(self, name):
        return self.recorders[name]

    def clear(self):
        for recorder in self.recorders.values():
            recorder.clear()

    def close(self):
        for recorder in self.recorders.values():
            recorder.close()

    def __enter__(self):
        for recorder in self.recorders.values():
            recorder.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for recorder in self.recorders.values():
            recorder.__exit__(exc_type, exc_value, traceback)
