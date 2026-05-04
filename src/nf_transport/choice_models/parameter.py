import torch
from torch import Tensor, nn


class ChoiceParameter(nn.Parameter):
    # Main addition from nn.Parameter: priors
    def __new__(cls, data, requires_grad=True):
        return super().__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad: bool = True):
        self.prior = torch.distributions.Uniform(
            torch.tensor([-2.0]), torch.tensor([2.0])
        )

    def set_prior(self, dist):
        self.prior = dist

    def eval_prior(self, x):
        return self.prior.log_prob(x)
