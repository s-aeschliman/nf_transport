import torch
from database_process import ChoiceDataset
from logpd import PosteriorDensity
from parameter import ChoiceParameter
from torch.distributions import (
    Bernoulli,
    Categorical,
    Exponential,
    LogNormal,
    Normal,
    Uniform,
)
from torch.utils.data import DataLoader, Dataset
from utility import ChoiceUtility


def test_adding_priors():
    post = PosteriorDensity()
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


def likelihood(Vlist, y):

    return


def test_adding_likelihood():

    post = PosteriorDensity()
    print(post.value())
    sm_train = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["AGE", "INCOME", "TRAIN_TT", "TRAIN_CO", "TRAIN_HE"],
        avail_cols=["TRAIN_AV"],
    )

    sm_car = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["CAR_TT", "CAR_CO"],
        avail_cols=["CAR_AV"],
    )

    sm_sm = ChoiceDataset(
        filepath="data/swissmetro.dat",
        choice_col="CHOICE",
        feature_cols=["AGE", "INCOME", "SM_TT", "SM_CO", "SM_HE"],
        avail_cols=["SM_AV"],
    )

    Xtrain = sm_train.features
    Atrain = sm_train.availability
    Xcar = sm_car.features
    Acar = sm_car.availability
    Xsm = sm_sm.features
    Asm = sm_sm.availability
    y = sm_train.choices

    # parameters
    beta_train = ChoiceParameter(
        torch.randn(Xtrain.shape[1] + 1).uniform_()  # +1 for ASC
    )
    beta_car = ChoiceParameter(torch.randn(Xcar.shape[1]).uniform_())
    beta_sm = ChoiceParameter(
        torch.randn(Xtrain.shape[1] + 1).uniform_()  # +1 for ASC
    )

    print("Adding train priors")
    beta_train.set_prior(
        Normal(torch.zeros(Xtrain.size(1) + 1), torch.ones(Xtrain.size(1) + 1))
    )
    post.add(beta_train.eval_prior(beta_train).sum())
    print(post.value())

    print("adding car priors")
    beta_car.set_prior(Normal(torch.zeros(Xcar.size(1)), torch.ones(Xcar.size(1))))
    post.add(beta_car.eval_prior(beta_car).sum())
    print(post.value())

    print("adding sm priors")
    beta_sm.set_prior(Normal(torch.zeros(Xsm.size(1) + 1), torch.ones(Xsm.size(1) + 1)))
    post.add(beta_sm.eval_prior(beta_sm).sum())
    print(post.value())

    Xtrain_aug = torch.cat((torch.ones(Xtrain.size(0), 1), Xtrain), 1)
    Xsm_aug = torch.cat((torch.ones(Xsm.size(0), 1), Xsm), 1)

    V_train = ChoiceUtility(Xtrain_aug, beta_train)
    V_sm = ChoiceUtility(Xsm_aug, beta_sm)
    V_car = ChoiceUtility(Xcar, beta_car)

    logits = torch.stack([V_train.value(), V_sm.value(), V_car.value()], dim=1)

    print("adding likelihood")
    likelihood = Categorical(logits=logits).log_prob(y).sum()

    post.add(likelihood)

    print(post.value())

    return


test_adding_likelihood()
