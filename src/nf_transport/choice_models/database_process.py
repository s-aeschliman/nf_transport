import pandas as pd
import torch
from torch.utils.data import Dataset


class ChoiceDataset(Dataset):
    """
    A class that prepares a torch Dataset for any generic choice model data set.
    __getitem__() returns three objects. in order, they are:
        1. the data set features (that enter into the choice utilities)
        2. the alternative availability matrix
        3. the choices
    """

    def __init__(self, feature_cols, choice_col, avail_cols, filepath: str) -> None:
        super().__init__()
        if ".dat" in filepath:
            sepstring = "\t"
        else:
            sepstring = ","

        df = pd.read_csv(filepath, sep=sepstring)
        df = df[df[choice_col] > 0]
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.choices = torch.tensor(df[choice_col].values, dtype=torch.long) - 1
        self.availability = torch.tensor(df[avail_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.choices)

    def __getitem__(self, idx):
        return self.features, self.availability, self.choices
