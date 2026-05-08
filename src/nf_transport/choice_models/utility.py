import torch
from torch import Tensor


class ChoiceUtility:
    v: Tensor

    def __init__(
            self, 
            features: Tensor, 
            feature_names: list[str],
            param_slices: dict[str, Tensor], 
            feature_mapping: dict[str, str],
            asc_name: str
    ):

        batch_size = next(iter(param_slices.values())).size(0)
        v = torch.zeros(batch_size, features.size(0))

        
        # add ASC
        if asc_name and asc_name in param_slices:
            v += param_slices[asc_name]
        
        # add predictors
        for i, feat in enumerate(feature_names):
            if feat in feature_mapping:
                param_name = feature_mapping[feat]
                beta = param_slices[param_name]
                x = features[:, i].unsqueeze(0)
                v += beta * x

        self.v = v

    def value(self):
        return self.v
