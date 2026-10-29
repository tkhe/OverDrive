import torch
from torch import nn

from .recorder import RecorderManager

__all__ = ["DMLDistiller"]


def _detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    if isinstance(data, dict):
        return {key: _detach(value) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_detach(value) for value in data)
    if isinstance(data, list):
        return [_detach(value) for value in data]
    return data


class DMLDistiller(nn.Module):
    """Record each model once and compute directed, learner-owned losses.

    ``recorders[model][name]`` uses the existing recorder classes. Losses and
    argument mappings are indexed by ``[learner][peer][loss_name]``; connectors
    by ``[learner][connector_name]``. Within a mapping, ``from_student`` selects
    the learner (True) or peer (False). Peer data is detached before connectors
    run, so even a connector applied to peer data belongs to the learner.
    """

    def __init__(self, recorders, distill_losses, loss_forward_mappings, connectors=None):
        super().__init__()
        self.recorders = {name: RecorderManager(value) for name, value in recorders.items()}
        self.distill_losses = nn.ModuleDict(
            {
                learner: nn.ModuleDict({peer: nn.ModuleDict(losses) for peer, losses in peers.items()})
                for learner, peers in distill_losses.items()
            }
        )
        self.connectors = nn.ModuleDict({name: nn.ModuleDict(value) for name, value in (connectors or {}).items()})
        self.loss_forward_mappings = loss_forward_mappings

        if self.distill_losses.keys() != loss_forward_mappings.keys():
            raise ValueError("distill_losses and loss_forward_mappings must have the same learners")
        for learner, peers in self.distill_losses.items():
            if learner in peers:
                raise ValueError(f"DML does not allow self-learning edges: {learner}")
            mappings = loss_forward_mappings[learner]
            if peers.keys() != mappings.keys() or any(losses.keys() != mappings[peer].keys() for peer, losses in peers.items()):
                raise ValueError(f"Losses and mappings must have the same peers and loss names for {learner}")

    def prepare(self, models):
        referenced = set(self.distill_losses) | set(self.connectors) | set(self.recorders)
        referenced.update(peer for peers in self.distill_losses.values() for peer in peers)
        unknown = referenced - models.keys()
        if unknown:
            raise ValueError(f"Distillation references unknown students: {sorted(unknown)}")
        for name, manager in self.recorders.items():
            manager.initialize(models[name])
        for name in models:
            self.recorders.setdefault(name, RecorderManager())

    def learner_modules(self, name):
        """A temporary parameter view, not another registered copy of the modules."""
        modules = nn.ModuleDict()
        if name in self.distill_losses:
            modules["losses"] = self.distill_losses[name]
        if name in self.connectors:
            modules["connectors"] = self.connectors[name]
        return modules

    def get_record(self, learner, peer, recorder, from_student, record_idx=0, data_idx=None, connector=None, connector_idx=None):
        name = learner if from_student else peer
        data = self.recorders[name].get_recorder(recorder).get_record_data(record_idx, data_idx)
        if not from_student:
            data = _detach(data)
        if connector is not None:
            data = self.connectors[learner][connector](data)
        return data if connector_idx is None else data[connector_idx]

    def compute_distill_losses(self):
        result = {}
        for learner, peers in self.distill_losses.items():
            for peer, losses in peers.items():
                for name, loss in losses.items():
                    mapping = self.loss_forward_mappings[learner][peer][name]
                    arguments = {arg: self.get_record(learner, peer, **record) for arg, record in mapping.items()}
                    result[f"{learner}/from_{peer}/{name}"] = loss(**arguments) / len(peers)
        return result

    def clear(self):
        for manager in self.recorders.values():
            manager.clear()

    def close(self):
        for manager in self.recorders.values():
            manager.close()

    def __enter__(self):
        self.clear()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear()
