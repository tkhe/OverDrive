from collections import defaultdict
from contextlib import contextmanager

import torch

from .history_buffer import HistoryBuffer

__all__ = [
    "EventStorage",
    "get_event_storage",
    "has_event_storage",
]

_CURRENT_STORAGE_STACK = []


def get_event_storage() -> "EventStorage":
    """
    Returns:
        The `EventStorage` object that's currently being used.
        Throws an error if no `EventStorage` is currently enabled.
    """
    if len(_CURRENT_STORAGE_STACK) == 0:
        raise RuntimeError("get_event_storage() has to be called inside a 'with EventStorage(...)' context!")

    return _CURRENT_STORAGE_STACK[-1]


def has_event_storage() -> bool:
    """
    Returns:
        Check if there are EventStorage() context existed.
    """
    return len(_CURRENT_STORAGE_STACK) > 0


class EventStorage:
    """
    Store events produced during training, such as scalar values, images, and
    histograms. The storage keeps track of the current iteration and provides
    access to scalar histories and optionally smoothed latest values.

    EventStorage is designed to be used as a context manager. While a storage
    is active, it can be obtained through :func:`get_event_storage`.

    Examples:
        >>> with EventStorage(start_iter=0) as storage:
        ...     storage.put_scalar("loss", 1.0)
        ...     storage.step()
    """

    def __init__(self, start_iter: int):
        """
        Args:
            start_iter (int): the iteration number at which event recording
                starts.
        """
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ""
        self._vis_data = []
        self._histograms = []

    def put_image(self, image_name, image_tensor):
        """
        Store an image together with the current iteration.

        Args:
            image_name (str): name used to identify the image.
            image_tensor (torch.Tensor): an image in ``CHW`` format. The values
                are expected to lie in the range ``[0, 1]``.
        """
        self._vis_data.append((image_name, image_tensor, self._iter))

    def put_scalar(self, name, value, smoothing_hint=True, cur_iter=None):
        """
        Add a scalar value to the storage.

        Args:
            name (str): name used to identify the scalar.
            value (float or torch.Tensor): scalar value to record. Values that
                can be converted by :class:`float` are accepted.
            smoothing_hint (bool): whether consumers should smooth this scalar
                when displaying its latest value. The hint must remain the same
                for every value recorded under a given name.
            cur_iter (int, optional): iteration associated with the value. If
                omitted, the storage's current iteration is used.
        """
        name = self._current_prefix + name
        cur_iter = self._iter if cur_iter is None else cur_iter
        history = self._history[name]
        value = float(value)
        history.update(value, cur_iter)
        self._latest_scalars[name] = (value, cur_iter)

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None and existing_hint != smoothing_hint:
            raise ValueError(f"Scalar {name} was put with a different smoothing_hint!")

        self._smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, cur_iter=None, **kwargs):
        """
        Put multiple scalars from keyword arguments.

        Examples:
            >>> storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint, cur_iter=cur_iter)

    def put_histogram(self, hist_name, hist_tensor, bins=1000):
        """
        Create a histogram from a tensor.

        Args:
            hist_name (str): The name of the histogram to put into tensorboard.
            hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
                into a histogram.
            bins (int): Number of histogram bins.
        """
        ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()

        # Create a histogram with PyTorch
        hist_counts = torch.histc(hist_tensor, bins=bins)
        hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)

        # Parameter for the add_histogram_raw function of SummaryWriter
        hist_params = dict(
            tag=hist_name,
            min=ht_min,
            max=ht_max,
            num=hist_tensor.numel(),
            sum=float(hist_tensor.sum()),
            sum_squares=float(torch.sum(hist_tensor**2)),
            bucket_limits=hist_edges[1:].tolist(),
            bucket_counts=hist_counts.tolist(),
            global_step=self._iter,
        )
        self._histograms.append(hist_params)

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError(f"No history metric available for {name}!")
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.

        This provides a default behavior that other writers can use.

        Note: All scalars saved in the past `window_size` iterations are used for smoothing.
        This is different from the `window_size` definition in HistoryBuffer.
        Use :meth:`get_history_window_size` to get the `window_size` used in HistoryBuffer.
        """
        result = {}
        for k, (v, itr) in self._latest_scalars.items():
            result[k] = (
                (self._history[k].median(self.count_samples(k, window_size)) if self._smoothing_hints[k] else v),
                itr,
            )
        return result

    def count_samples(self, name, window_size=20):
        """
        Return the number of samples logged in the past `window_size` iterations.
        """
        samples = 0
        data = self._history[name].values()
        for _, iter_ in reversed(data):
            if iter_ > data[-1][1] - window_size:
                samples += 1
            else:
                break
        return samples

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should either: (1) Call this function to increment storage.iter when needed. Or
        (2) Set `storage.iter` to the correct iteration number before each iteration.

        The storage will then be able to associate the new data with an iteration number.
        """
        self._iter += 1

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)

    def __enter__(self):
        """
        Activate this storage and return it as the current event storage.
        """
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Remove this storage from the active storage stack.

        Storages must leave their contexts in the reverse order in which they
        were entered.
        """
        if _CURRENT_STORAGE_STACK[-1] is not self:
            raise RuntimeError("Expected _CURRENT_STORAGE_STACK[-1] == self")

        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        try:
            yield
        finally:
            self._current_prefix = old_prefix

    def clear_images(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._vis_data = []

    def clear_histograms(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []
