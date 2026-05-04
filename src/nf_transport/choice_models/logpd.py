import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from nf_transport.choice_models.parameter import ChoiceParameter


class LogJointDensity:
    logpd: float

    def __init__(self):
        self.logpd = 0.0

    def add(self, pdf_eval: float) -> None:
        self.logpd += pdf_eval

    def value(self) -> float:
        return self.logpd


class ChoiceModelLogJoint:
    def __init__(self, data, params: list[ChoiceParameter], likelihood_fn) -> None:
        self.data = data
        self.parameters = params
        self.likelihood_fn = likelihood_fn

    def __call__(self, z: Tensor) -> float:
        lj = LogJointDensity()
        offset = 0
        slices = {}
        for param in self.parameters:
            chunk = z[..., offset : offset + param.dim]
            lj.add(param.log_prior(chunk))
            slices[param.name] = chunk
            offset += param.dim
        lj.add(self.likelihood_fn(slices, self.data))
        return lj.value()
