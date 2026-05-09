import torch
from torch import Tensor
from torch.distributions import Normal


class ChoiceParameter:
    def __init__(self, dim: int, name: str):
        self.prior = Normal(torch.zeros(dim), 3*torch.ones(dim))
        self.dim = dim
        self.name = name

    def set_prior(self, dist) -> None:
        self.prior = dist

    def log_prior(self, x: Tensor) -> Tensor:
        return self.prior.log_prob(x).sum(dim=-1)

class RandomParameter():
    def __init__(self, n_groups: int, name: str):
        self.n_groups = n_groups
        self.name = name
        self.prior = None
    
    def set_prior(self, dist, mean_hyperparam: ChoiceParameter, scale_hyperparam: ChoiceParameter) -> None:
        params = []
        for i in range(self.n_groups):
            p = ChoiceParameter(dim=1, name=f"{self.name}_{i}")
            p.set_prior(Normal(mean_hyperparam, scale_hyperparam))
            params.append(p)
        return

