import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torch.distributions import (
    Bernoulli,
    Categorical,
    Exponential,
    LogNormal,
    Normal,
    Uniform,
)


from nf_transport.choice_models.database_process import (
    ChoiceDataset,
    MultinomialChoiceData,
)
from nf_transport.choice_models.logpd import ChoiceModelLogJoint
from nf_transport.choice_models.nfvi import NFVI, PlanarFlow, RealNVPFlow
from nf_transport.choice_models.parameter import ChoiceParameter
from nf_transport.choice_models.utility import ChoiceUtility

def likelihood(param_slices, data: MultinomialChoiceData):
    feature_mapping = data.feature_mapping
    utilities = []
    availabilities = []

    asc_names = ["asc_train", None, "asc_car"]

    for i, dataset in enumerate(data.alternatives):
        V = ChoiceUtility(
            features=dataset.features, 
            feature_names=dataset.feature_names, 
            param_slices=param_slices,
            feature_mapping=feature_mapping,
            asc_name=asc_names[i]
        )

        utilities.append(V.value())
        availabilities.append(dataset.availability.all(dim=-1))


    logits = torch.stack(utilities, dim=-1)  # [Base_samples, N, J]
    avail = torch.stack(availabilities, dim=-1)  # [N, J]
    logits = logits.masked_fill(~avail, float("-inf"))
    ll = Categorical(logits=logits).log_prob(data.choices).sum(dim=1)
    return ll

def swissmetro_likelihood():

    sm_train = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["TRAIN_TT", "TRAIN_CO"],
        avail_cols=["TRAIN_AV"],
        id_col="ID",
        has_asc=True,
        scale=True,
    )

    sm_car = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["CAR_TT", "CAR_CO"],
        avail_cols=["CAR_AV"],
        id_col="ID",
        has_asc=True,
        scale=True,
    )

    sm_sm = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["SM_TT", "SM_CO"],
        avail_cols=["SM_AV"],
        id_col="ID",
        has_asc=False,
        scale=True,
    )

    num_inds = sm_train.num_ind

    # ASCs
    asc_train = ChoiceParameter(dim=1, name="asc_train")
    asc_car = ChoiceParameter(dim=1, name="asc_car")

    # random parameter for cost
    b_cost = ChoiceParameter(dim=sm_train.num_ind, name="b_cost_random")
    mu_cost = ChoiceParameter(dim=1, name="mu_cost")
    s_cost = ChoiceParameter(dim=1, name="s_cost")
    
    