import torch
from torch import Tensor
from nf_transport.choice_models.parameter import ChoiceParameter


class LogJointDensity:
    logpd: Tensor

    def __init__(self, base_samples: int = 1):
        self.logpd = torch.zeros(base_samples)

    def add(self, pdf_eval: Tensor) -> None:
        self.logpd += pdf_eval

    def value(self) -> Tensor:
        return self.logpd


class ChoiceModelLogJoint:
    def __init__(self, data, params: list[ChoiceParameter], likelihood_fn, base_samples: int = 1) -> None:
        self.data = data
        self.parameters = params
        self.likelihood_fn = likelihood_fn
        self.base_samples = base_samples

    def __call__(self, z: Tensor) -> Tensor:
        lj = LogJointDensity(base_samples=self.base_samples)
        offset = 0
        slices = {}
        for param in self.parameters:
            chunk = z[..., offset : offset + param.dim]
            lj.add(param.log_prior(chunk))
            slices[param.name] = chunk
            offset += param.dim
        lj.add(self.likelihood_fn(slices, self.data))
        return lj.value()
