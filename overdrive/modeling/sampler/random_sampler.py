import torch

from .base import Sampler

__all__ = ["RandomSampler"]


class RandomSampler(Sampler):
    def __init__(
        self,
        num_samples: int,
        positive_fraction: float,
    ):
        super().__init__()

        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def sample(self, match_labels):
        positive = ((match_labels != -1) & (match_labels != 0)).nonzero(as_tuple=True)[0]
        negative = (match_labels == 0).nonzero(as_tuple=True)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        # randomly select positive and negative examples
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[perm1]
        neg_idx = negative[perm2]

        # Fill with the ignore label (-1), then set positive and negative labels
        match_labels.fill_(-1)
        match_labels.scatter_(0, pos_idx, 1)
        match_labels.scatter_(0, neg_idx, 0)

        return match_labels
