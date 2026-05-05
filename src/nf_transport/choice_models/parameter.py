import torch
from torch import Tensor
from torch.distributions import Normal


class ChoiceParameter:
    def __init__(self, dim: int, name: str, requires_grad: bool = True):
        self.prior = Normal(torch.tensor([0.0]), torch.tensor([3.0]))
        self.dim = dim
        self.name = name

    def set_prior(self, dist):
        self.prior = dist

    def log_prior(self, x: Tensor) -> Tensor:
        return self.prior.log_prob(x).sum()
