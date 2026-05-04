import torch
from numpy import float32
from torch import nn
from torch.distributions import Distribution


class PosteriorDensity:
    logpd: float

    def __init__(self):
        self.logpd = 0.0

    def add(self, pdf_eval: float) -> None:
        self.logpd += pdf_eval

    def value(self) -> float:
        return self.logpd
