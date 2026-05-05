import torch
from torch import Tensor, tensor


class ChoiceUtility:
    v: Tensor

    def __init__(self, predictors: Tensor, params: Tensor, has_asc: bool = True):
        """
        Predictors: design matrix augmented with 1s for the ASC
        Params: vector of ChoiceParameters, including the ASC
        """
        if has_asc:
            predictors = torch.cat(
                (torch.ones(predictors.size(0), 1), predictors), dim=1
            )

        self.v = torch.matmul(params, predictors.T)

    def value(self):
        return self.v
