import torch
from torch import nn
import numpy as np


def tmod(x):
    """R^3 -> [-np.pi, np.pi)"""
    return torch.fmod(x + np.pi, 2 * np.pi) - np.pi


def sample_wn(n, mu, sigma, device="cpu"):
    """sample from wrapped normal distribution"""
    samples = torch.randn(n, device=device) * sigma + mu # (n,)
    samples = tmod(samples)
    return samples.unsqueeze(1)     # (n, 1)


class TorusConditionalFlowMatcher:
    """
    Class to compute the FoldFlow-base method.
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def sample_xt(self, x0, x1, t):
        xt = t.reshape(-1, 1) * (x1 - x0) + x0 + self.sigma * \
             torch.sqrt(t * (1 - t)).reshape(-1, 1) * torch.randn_like(x0)
        return tmod(xt)

    def sample_location_and_conditional_flow(self, x0, x1, t):
        """
        Compute the sample xt along the geodesic from x0 to x1
        and the conditional vector field ut(xt|z).

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t  : Tensor, shape (bs,)
            represents the noised time

        Returns
        -------
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn along the geodesic
        ut : conditional vector field ut(xt|z)
        """
        xt = self.sample_xt(x0, x1, t)
        ut = (x1 - xt) / (1 - t)
        return xt, tmod(ut)


class TorMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=64, time_varying=True, act_fn=nn.SiLU(), dropout=0.1):
        super(TorMLP, self).__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = 1

        in_dim_with_t = in_dim + (1 if time_varying else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim_with_t, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, input):
        v = self.net(input)
        return tmod(v)
