import torch
from torch import nn
from einops import rearrange
import sys
sys.path.append("..")
from utils import *


def smod(x):
    xn = torch.norm(x, dim=-1, keepdim=True)
    angle = torch.fmod(xn, 2 * np.pi)
    x = x * angle / xn
    return x


def _tmod(x):
    """R^3 -> [-np.pi, np.pi)"""
    return torch.fmod(x + np.pi, 2 * np.pi) - np.pi


class SO3ConditionalFlowMatcher:
    """
    Class to compute the FoldFlow-base method.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def fuse_x0(self, x0, matrix_format=False):
        rot_x0 = x0 if matrix_format else Exp(x0)
        igso3 = sample_igso3(torch.tensor(self.sigma, device=x0.device), rot_x0.shape[0])
        rot_x0 = Exp(igso3).type_as(x0) @ rot_x0
        x0 = rot_x0 if matrix_format else Log(rot_x0)
        return x0

    def sample_kt(self, k0, k1, t):
        """
        Function which compute the sample xt along the geodesic from x0 to x1 on sphere.
        """
        log_k0_k1 = sp_logmap(k0, k1)
        kt = sp_expmap(k0, t.reshape(-1, 1) * log_k0_k1)
        return kt

    def sample_tt(self, theta0, theta1, t):
        """
        Function which compute the sample xt along the geodesic from x0 to x1 on angle.
        """
        thetat = t.reshape(-1, 1) * (theta1 - theta0) + theta0
        return thetat

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
        k0, theta0 = to_axis_angle(x0)
        k1, theta1 = to_axis_angle(x1)
        kt = self.sample_kt(k0, k1, t)
        tt = self.sample_tt(theta0, theta1, t)
        # log_kt_k1 / (1 - t)
        ukt = sp_logmap(kt, k1) / (1 - t.reshape(-1, 1))
        utt = _tmod(theta1 - theta0)
        return kt, tt, ukt, utt


class SO3MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=64, time_varying=True, act_fn=nn.SiLU(), dropout=0.1):
        super(SO3MLP, self).__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = 4     # 3 for rotation axis k (unit vector) and 1 for rotation angle theta
        self.net = nn.Sequential(
            nn.Linear(in_dim + (1 if time_varying else 0), hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, input):
        v = self.net(input)
        k = sp_proj(input[:, -4:-1], v[:, :3])  # (B, 3)
        theta = _tmod(v[:, -1]).unsqueeze(-1)   # (B, 1)
        # x = rearrange(input[:, -9:], 'b (c d) -> b c d', c=3, d=3)
        # v = rearrange(v, 'b (c d) -> b c d', c=3, d=3)
        # Pv = tangent_space_proj(x, v) # Pv is on the tangent space of x
        # Pv = rearrange(Pv, 'b c d -> b (c d)', c=3, d=3)
        return k, theta
