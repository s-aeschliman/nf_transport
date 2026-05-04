import torch
from torch import Tensor, tensor


class ChoiceUtility:
    v: Tensor

    def __init__(self, predictors: Tensor, params: Tensor):
        """
        Predictors: design matrix augmented with 1s for the ASC
        Params: vector of ChoiceParameters, including the ASC
        """
        self.v = torch.matmul(predictors, params)

    def value(self):
        return self.v
    
