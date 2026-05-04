import torch
from database_process import ChoiceDataset
from logpd import LogJointDensity
from parameter import ChoiceParameter
from torch.distributions import (
    Bernoulli,
    Categorical,
    Exponential,
    LogNormal,
    Normal,
    Uniform,
)
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset
from utility import ChoiceUtility

from nf_transport.choice_models.database_process import MultinomialChoiceData
from nf_transport.choice_models.logpd import ChoiceModelLogJoint
from nf_transport.choice_models.nfvi import NFVI, PlanarFlow, PlanarTransform


def test_adding_priors():
    post = LogJointDensity()
    print(post.value())
    alpha = ChoiceParameter(torch.randn(1).uniform_())
    beta = ChoiceParameter(torch.randn(1).uniform_())
    alpha.set_prior(Normal(torch.tensor([0.0]), torch.tensor([1.0])))
    post.add(alpha.eval_prior(alpha))
    print(post.value())
    beta.set_prior(Normal(torch.tensor([0.0]), torch.tensor([3.0])))
    post.add(alpha.eval_prior(alpha))
    print(post.value())
    return


def likelihood(param_slices, data: ChoiceDataset):
    utilities = []
    for dataset, (name, beta) in zip(data.alternatives, param_slices.items()):
        V = ChoiceUtility(dataset.features, beta, dataset.has_asc)
        utilities.append(V.value())
    logits = torch.stack(utilities, dim=1)
    ll = Categorical(logits=logits).log_prob(data.choices).sum()
    return ll


def test_adding_likelihood():

    sm_train = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["AGE", "INCOME", "TRAIN_TT", "TRAIN_CO", "TRAIN_HE"],
        avail_cols=["TRAIN_AV"],
        has_asc=True,
    )

    sm_car = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["CAR_TT", "CAR_CO"],
        avail_cols=["CAR_AV"],
        has_asc=False,
    )

    sm_sm = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["AGE", "INCOME", "SM_TT", "SM_CO", "SM_HE"],
        avail_cols=["SM_AV"],
        has_asc=True,
    )

    beta_train = ChoiceParameter(dim=sm_train.features.size(1) + 1, name="beta_train")
    beta_train.set_prior(
        Normal(torch.zeros(beta_train.dim), torch.ones(beta_train.dim))
    )
    beta_car = ChoiceParameter(dim=sm_car.features.size(1), name="beta_car")
    beta_car.set_prior(Normal(torch.zeros(beta_car.dim), torch.ones(beta_car.dim)))
    beta_sm = ChoiceParameter(dim=sm_sm.features.size(1) + 1, name="beta_sm")
    beta_sm.set_prior(Normal(torch.zeros(beta_sm.dim), torch.ones(beta_sm.dim)))

    log_joint = ChoiceModelLogJoint(
        data=MultinomialChoiceData(
            alternatives=[sm_train, sm_car, sm_sm], choices=sm_train.choices
        ),
        params=[beta_train, beta_car, beta_sm],
        likelihood_fn=likelihood,
    )

    param_list = [beta_train, beta_car, beta_sm]

    return log_joint, param_list


def swissmetro():
    import matplotlib.pyplot as plt
    import numpy as np

    log_joint, param_list = test_adding_likelihood()
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

    offset = 0
    for p in param_list:
        print(f"{p.name}: {posterior_mean[offset : offset + p.dim]}")
        offset += p.dim

    plt.plot(np.arange(n_steps), elbo_list)
    plt.savefig("nfvi_elbo.png")


swissmetro()
