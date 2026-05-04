import numpy as np
import torch
from torch.distributions import (
    Bernoulli,
    Categorical,
    Exponential,
    LogNormal,
    Normal,
    Uniform,
)
from torch.utils.data import Dataset

from nf_transport.choice_models.database_process import (
    ChoiceDataset,
    MultinomialChoiceData,
)
from nf_transport.choice_models.logpd import ChoiceModelLogJoint, LogJointDensity
from nf_transport.choice_models.nfvi import NFVI, PlanarFlow, PlanarTransform
from nf_transport.choice_models.parameter import ChoiceParameter
from nf_transport.choice_models.utility import ChoiceUtility


def likelihood(param_slices, data: ChoiceDataset):
    utilities = []
    availabilities = []
    for dataset, (name, beta) in zip(data.alternatives, param_slices.items()):
        V = ChoiceUtility(dataset.features, beta, dataset.has_asc)
        utilities.append(V.value())
        availabilities.append(dataset.availability.all(dim=-1))  # [N]
    logits = torch.stack(utilities, dim=1)  # [N, J]
    avail = torch.stack(availabilities, dim=1)  # [N, J]
    logits = logits.masked_fill(~avail, float("-inf"))
    ll = Categorical(logits=logits).log_prob(data.choices).sum()
    return ll


def swissmetro_likelihood():

    sm_train = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["TRAIN_TT", "TRAIN_CO"],
        avail_cols=["TRAIN_AV"],
        has_asc=True,
        scale=True,
    )

    sm_car = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["CAR_TT", "CAR_CO"],
        avail_cols=["CAR_AV"],
        has_asc=True,
        scale=True,
    )

    sm_sm = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["SM_TT", "SM_CO"],
        avail_cols=["SM_AV"],
        has_asc=False,
        scale=True,
    )

    beta_train = ChoiceParameter(dim=sm_train.features.size(1) + 1, name="beta_train")
    beta_train.set_prior(
        Normal(torch.zeros(beta_train.dim), torch.ones(beta_train.dim))
    )
    beta_car = ChoiceParameter(dim=sm_car.features.size(1) + 1, name="beta_car")
    beta_car.set_prior(Normal(torch.zeros(beta_car.dim), torch.ones(beta_car.dim)))
    beta_sm = ChoiceParameter(dim=sm_sm.features.size(1), name="beta_sm")
    beta_sm.set_prior(Normal(torch.zeros(beta_sm.dim), torch.ones(beta_sm.dim)))

    log_joint = ChoiceModelLogJoint(
        data=MultinomialChoiceData(
            alternatives=[sm_train, sm_sm, sm_car], choices=sm_train.choices
        ),
        params=[beta_train, beta_sm, beta_car],
        likelihood_fn=likelihood,
    )

    param_list = [beta_train, beta_sm, beta_car]

    return log_joint, param_list


def swissmetro():
    import matplotlib.pyplot as plt
    import numpy as np

    log_joint, param_list = swissmetro_likelihood()
    param_dim = sum(p.dim for p in param_list)
    nfvi = NFVI(
        flow=PlanarFlow(dim=param_dim, K=12), log_joint=log_joint, param_dim=param_dim
    )

    # train
    lr = 1e-3
    n_steps = 2000
    optimizer = torch.optim.Adam(params=nfvi.parameters(), lr=lr)
    elbo_list = []
    for i in range(n_steps):
        loss = -nfvi.elbo(n_samples=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elbo_list.append(-loss.item())
        if i % 10 == 0:
            print(f"iter {i:4d}  ELBO = {-loss.item():.2f}")

    with torch.no_grad():
        z0 = nfvi.base.rsample((2000,))
        z_K, _ = nfvi.flow(z0)
        posterior_mean = z_K.mean(dim=0)

    posterior = {}
    offset = 0
    for p in param_list:
        print(f"{p.name}: {posterior_mean[offset : offset + p.dim]}")
        for d in range(p.dim):
            posterior[f"{p.name}[{d}]"] = z_K[:, offset + d]
        offset += p.dim

    np.savez(
        "estimates/sm_posterior.npz",
        **{k: v.numpy()[np.newaxis, :] for k, v in posterior.items()},
    )


swissmetro()
