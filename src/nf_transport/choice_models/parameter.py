import torch
from torch import Tensor
from torch.distributions import Normal


class ChoiceParameter:
    def __init__(self, dim: int, name: str):
        self.prior = Normal(torch.zeros(dim), 3 * torch.ones(dim))
        self.dim = dim
        self.name = name

    def set_prior(self, dist) -> None:
        self.prior = dist

    def log_prior(self, x: Tensor) -> Tensor:
        return self.prior.log_prob(x).sum(dim=-1)
