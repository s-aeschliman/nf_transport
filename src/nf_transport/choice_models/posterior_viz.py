import arviz as az
import arviz_plots as azp
import numpy as np
import matplotlib.pyplot as plt


def plot_params():
    post = np.load("estimates/sm_posterior.npz")
    idata = az.from_dict(posterior=dict(post))
    forest = azp.plot_forest(idata)
    forest.savefig("figures/sm_forest.png")

def compare_params():
    post_planar = np.load("estimates/sm_posterior_planar.npz")
    post_realNVP = np.load("estimates/sm_posterior_realNVP.npz")

    idata_planar = az.from_dict(posterior=dict(post_planar))
    idata_realNVP = az.from_dict(posterior=dict(post_realNVP))

    forest = azp.plot_forest(

        {"Planar": idata_planar, "realNVP": idata_realNVP},
    )

    forest = azp.add_lines(
        forest,
        values=0
    )
    plt.legend(["Planar", "realNVP"])
    plt.savefig("figures/sm_forest_comp.png", dpi=300)

compare_params()
