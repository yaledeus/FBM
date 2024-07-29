import torch
from torch import nn
import numpy as np


class EuConditionalFlowMatcher:
    """
    Class to compute the FoldFlow-base method.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def sample_xt(self, x0, x1, t):
        xt = t.reshape(-1, 1) * (x1 - x0) + x0 + self.sigma * \
             torch.sqrt(t * (1 - t)).reshape(-1, 1) * torch.randn_like(x0)
        return xt

    def likelihood(self, xt, x0, x1, t):
        """
        Compute the likelihood q_t(x_t|x_0, x_1)
        """
        ndim = x0.numel()
        mu = t * x1 + (1 - t) * x0
        t_mul_1_minus_t = np.max((t * (1 - t), 1e-2))
        nll = torch.norm(xt - mu) ** 2 / (2 * t_mul_1_minus_t * self.sigma ** 2) \
              + 0.5 * ndim * (np.log(2 * np.pi) + np.log(t_mul_1_minus_t)) \
              + ndim * np.log(self.sigma)
        nll /= ndim     # scaling for numerical stability
        return torch.exp(-nll)

    def conditional_score(self, xt, x0, x1, t):
        """
        Compute the likelihood \nabla log q_t(x_t|x_0, x_1)
        """
        mu = t * x1 + (1 - t) * x0
        t_mul_1_minus_t = np.max((t * (1 - t), 1e-2))
        cscore = -(xt - mu) / (t_mul_1_minus_t * self.sigma ** 2)
        return cscore

    def vector_field_to_score(self, v_fwd, v_rev, t):
        score = (v_fwd - v_rev) / (self.sigma ** 2)
        return score

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
        u_fwd : conditional vector field ut(xt|z) of dx_t=(x_t-x_0)/t dt+\sigma dB_t
        u_rev : conditional vector field ut(xt|z) of dx_t=(x_1-x_t)/(1-t) dt+\sigma dB_t
        """
        xt = self.sample_xt(x0, x1, t)
        u_fwd = (x1 - xt) / (1 - t)
        u_rev = (xt - x0) / t
        return xt, u_fwd, u_rev


class EuMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=64, time_varying=True, act_fn=nn.SiLU(), dropout=0.1):
        super(EuMLP, self).__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = 3

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
        v = v.clip(min=-5., max=5.)
        return v
