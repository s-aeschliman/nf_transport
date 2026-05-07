import torch
from nf_transport.choice_models.logpd import LogJointDensity
from nf_transport.choice_models.parameter import ChoiceParameter
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.utils.data.dataset import TensorDataset


class NFVI(nn.Module):
    def __init__(self, flow: nn.Module, log_joint, param_dim: int) -> None:
        super().__init__()
        self.flow = flow
        self.log_joint = log_joint
        self.base = MultivariateNormal(torch.zeros(param_dim), torch.eye(param_dim))

    def elbo(self, n_samples: int) -> Tensor:
        z0 = self.base.rsample((n_samples,))
        z, sum_log_det_J = self.flow(z0)
        return (self.log_joint(z) + sum_log_det_J - self.base.log_prob(z0)).mean()

    def __str__(self):
        return self.flow.__str__()

class PlanarTransform(nn.Module):
    # borrowed, for now, from https://github.com/e-hulten/planar-flows/blob/master/planar_transform.py
    """Implementation of the invertible transformation used in planar flow:
        f(z) = z + u * h(dot(w.T, z) + b)
    See Section 4.1 in https://arxiv.org/pdf/1505.05770.pdf.
    """

    def __init__(self, dim: int = 2):
        """Initialise weights and bias.

        Args:
            dim: Dimensionality of the distribution to be estimated.
        """
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))

    def forward(self, z: Tensor) -> Tensor:
        if torch.mm(self.u, self.w.T) < -1:
            self.get_u_hat()

        return z + self.u * nn.Tanh()(torch.mm(z, self.w.T) + self.b)

    def log_det_J(self, z: Tensor) -> Tensor:
        if torch.mm(self.u, self.w.T) < -1:
            self.get_u_hat()
        a = torch.mm(z, self.w.T) + self.b
        psi = (1 - nn.Tanh()(a) ** 2) * self.w
        abs_det = (1 + torch.mm(self.u, psi.T)).abs()
        log_det = torch.log(1e-4 + abs_det)

        return log_det

    def get_u_hat(self) -> None:
        """Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition
        for invertibility of the transformation f(z). See Appendix A.1.
        """
        wtu = torch.mm(self.u, self.w.T)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        self.u.data = (
            self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1) ** 2
        )


class PlanarFlow(nn.Module):
    def __init__(self, dim: int = 2, K: int = 6):
        """Make a planar flow by stacking planar transformations in sequence.

        Args:
            dim: Dimensionality of the distribution to be estimated.
            K: Number of transformations in the flow.
        """
        super().__init__()
        self.layers = nn.ModuleList([PlanarTransform(dim) for _ in range(K)])
        self.model = nn.Sequential(*self.layers)

    def forward(self, z: Tensor) -> tuple[Tensor, float]:
        log_det_J = 0

        for layer in self.layers:
            log_det_J += layer.log_det_J(z) # type: ignore
            z = layer(z) # type: ignore

        return z, log_det_J

    def __str__(self):
        return "planar"

class GenericFlow(nn.Module):
    def __init__(self, transform: type[nn.Module], K: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleList([transform(dim) for _ in range(K)])
        self.model = nn.Sequential(*self.layers)

    def forward(self, z: Tensor) -> tuple[Tensor, float]:
        log_det_J = 0
        for layer in self.layers:
            log_det_J += layer.log_det_J(z) # type: ignore
            z = layer(z) # type: ignore
        return z, log_det_J
    
class AffineCoupling(nn.Module):
    mask: Tensor

    def __init__(self, dim: int, hdim: int, mask):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)
        self.network = nn.Sequential(
            nn.Linear(dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, 2*dim) # 2 * dim to essentially produce 2 heads, one for s and one for t
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        z_m = z * self.mask
        h = self.network(z_m)
        s, t = h.chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        y = z_m + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det_J = s.sum(dim=-1)
        return y, log_det_J
    
class RealNVPFlow(nn.Module):
    def __init__(self, dim: int, hdim: int, mask, K: int):
        super().__init__()
        temp_mask = mask
        coupling_layers = []
        for _ in range(K):
            layer = AffineCoupling(dim=dim, hdim=hdim, mask=temp_mask)
            temp_mask = 1 - temp_mask
            coupling_layers.append(layer)
                            
        self.layers = nn.ModuleList(coupling_layers)
    
    def forward(self, z: Tensor) -> tuple[Tensor, float]:
        log_det_J = 0
        for layer in self.layers:
            z, ldj = layer(z)
            log_det_J += ldj
        
        return z, log_det_J

    def __str__(self):
        return "realNVP"