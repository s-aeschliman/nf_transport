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


def swissmetro_likelihood(base_samples: int = 1):

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

    # define params
    asc_train = ChoiceParameter(dim=1, name="asc_train")
    asc_car = ChoiceParameter(dim=1, name="asc_car")
    b_time = ChoiceParameter(dim=1, name="b_time")
    b_cost = ChoiceParameter(dim=1, name="b_cost")

    for b in [b_time, b_cost]:
        b.set_prior(Normal(torch.tensor([-1.5]), torch.tensor([1.0])))

    param_list = [asc_train, asc_car, b_time, b_cost]

    # map params to predictors
    feature_mapping = {
        "TRAIN_TT": "b_time",
        "CAR_TT": "b_time",
        "SM_TT": "b_time",
        "TRAIN_CO": "b_cost",
        "SM_CO": "b_cost",
        "CAR_CO": "b_cost",
    }

    log_joint = ChoiceModelLogJoint(
        data=MultinomialChoiceData(
            alternatives=[sm_train, sm_sm, sm_car], 
            choices=sm_train.choices,
            feature_mapping=feature_mapping
        ),
        params=param_list,
        likelihood_fn=likelihood,
        base_samples=base_samples
    )


    return log_joint, param_list


def swissmetro():

    base_samples = 1

    log_joint, param_list = swissmetro_likelihood(base_samples=base_samples)
    param_dim = sum(p.dim for p in param_list)

    dim_half = int(math.floor(param_dim/2))

    # PLANAR FLOW
    nfvi_planar = NFVI(
        flow=PlanarFlow(dim=param_dim, K=12), 
        log_joint=log_joint, 
        param_dim=param_dim
    )

    # REALNVP (AFFINE COUPLING) FLOW
    mask = torch.zeros(param_dim)
    mask[dim_half:] = 1

    nfvi_realNVP = NFVI(
        flow=RealNVPFlow(dim=param_dim, hdim=dim_half, K=12, mask=mask),
        log_joint=log_joint,
        param_dim=param_dim
    )

    # train
    lr = 1e-3
    n_steps = 2000
    elbos = {}
    for flow in [nfvi_planar, nfvi_realNVP]:
        nme = flow.__str__()
        optimizer = torch.optim.Adam(params=flow.parameters(), lr=lr)
        elbo_list = []
        for i in range(n_steps):
            loss = -flow.elbo(n_samples=base_samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elbo_list.append(-loss.item())
            if i % 100 == 0:
                print(f"iter {i:4d}  ELBO = {-loss.item():.2f}")

        elbos[nme] = elbo_list

        with torch.no_grad():
            z0 = flow.base.rsample((2000,))
            z_K, _ = flow.flow(z0)
            posterior_mean = z_K.mean(dim=0)

        posterior = {}
        offset = 0
        for p in param_list:
            print(f"{p.name}: {posterior_mean[offset : offset + p.dim]}")
            for d in range(p.dim):
                posterior[f"{p.name}[{d}]"] = z_K[:, offset + d]
            offset += p.dim

        np.savez(
            f"estimates/sm_posterior_{nme}.npz",
            **{k: v.numpy()[np.newaxis, :] for k, v in posterior.items()},
        )

    plt.plot(np.arange(n_steps), elbos["realNVP"], "b:", np.arange(n_steps), elbos["planar"], "r-.", lw=0.5)
    plt.ylim((-30000, max(elbos["realNVP"]) + 300))
    plt.ylabel("ELBO")
    plt.xlabel("Training step")
    plt.legend(("Affine Coupling", "Planar Flow"), loc="lower center")
    plt.tight_layout()
    plt.savefig("figures/elbo_comp.png", dpi=300)
    

swissmetro()
