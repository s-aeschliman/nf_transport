import arviz as az
import arviz_plots as azp
import numpy as np

# from arviz_plots import azp


def plot_params():
    post = np.load("estimates/sm_posterior.npz")
    idata = az.from_dict(posterior=dict(post))
    forest = azp.plot_forest(idata)
    forest.savefig("figures/sm_forest.png")


plot_params()
