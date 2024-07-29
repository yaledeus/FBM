import torch
from torch import nn
import numpy as np
from .graph import construct_edges
from .torchmd_et import TorchMD_ET
from .Euclidean import EuConditionalFlowMatcher

import sys

sys.path.append('..')
from utils import *


def _init_linear_(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class GeomSFM(nn.Module):
    def __init__(self, hidden_dim, rbf_dim, heads, layers, cutoff=5.0, s_eu=0.1):
        super(GeomSFM, self).__init__()

        self.hidden_dim = hidden_dim
        self.rbf_dim = rbf_dim
        self.heads = heads
        self.layers = layers
        self.cutoff = cutoff

        # matcher
        self.eu_matcher = EuConditionalFlowMatcher(sigma=s_eu)

        # TorchMD-net
        self.net = TorchMD_ET(
            hidden_channels=hidden_dim, num_layers=layers, num_rbf=rbf_dim, num_heads=heads,
            cutoff_lower=0.0, cutoff_upper=cutoff, max_z=NUM_ATOM_TYPE
        )

        self.eu_fwd_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.eu_rev_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        _init_linear_(self.eu_fwd_weight)
        _init_linear_(self.eu_rev_weight)

        self.eu_scale = 0.5

    def _train(self, batch):
        x0, x1, atype, abid = batch["x0"], batch["x1"], batch["atype"],  batch["abid"]
        N = x0.shape[0]

        # uniformly sample time step
        t = np.random.rand()  # use the same time step for all mini-batches
        t_atom = torch.ones(N, 1, dtype=x0.dtype, device=x0.device) * t

        # sample noised data
        xt, eu_ut_fwd, eu_ut_rev = self.eu_matcher.sample_location_and_conditional_flow(x0, x1, t_atom)

        # conv
        h, vec, _, _, _ = self.net(z=atype, t=t_atom, pos=xt, batch=abid)

        eu_out_fwd = torch.sum(self.eu_fwd_weight(h).unsqueeze(2) * vec.transpose(1, 2), dim=1) # (N, 3)
        eu_out_rev = torch.sum(self.eu_rev_weight(h).unsqueeze(2) * vec.transpose(1, 2), dim=1)

        loss_eu = F.mse_loss(eu_out_fwd, self.eu_scale * (1 - t) * eu_ut_fwd, reduction='mean') + \
                  F.mse_loss(eu_out_rev, self.eu_scale * (-t) * eu_ut_rev, reduction='mean')

        loss_aux = 0
        # bb_index = batch["bb_index"]
        same_abid = abid.unsqueeze(-1).repeat(1, abid.shape[0])
        same_abid = same_abid == same_abid.transpose(0, 1)  # (N, N)
        # full atom
        pred_x0 = xt + eu_out_rev / self.eu_scale
        # pairwise backbone
        D0 = torch.cdist(x0, x0)  # (N, N)
        pred_D0 = torch.cdist(pred_x0, pred_x0)
        valid_mask = (D0 < 6.0) & same_abid
        loss_aux += (1 - t) * F.mse_loss(pred_D0[valid_mask], D0[valid_mask], reduction='sum') \
                    / (valid_mask.sum() - x0.shape[0])
        # full atom
        pred_x1 = xt + eu_out_fwd / self.eu_scale
        # pairwise backbone
        D1 = torch.cdist(x1, x1)
        pred_D1 = torch.cdist(pred_x1, pred_x1)
        valid_mask = (D1 < 6.0) & same_abid
        loss_aux += t * F.mse_loss(pred_D1[valid_mask], D1[valid_mask], reduction='sum') \
                    / (valid_mask.sum() - x1.shape[0])

        loss = loss_eu + 0.25 * loss_aux

        return loss, (loss_eu, loss_aux)

    def fwd(self, atype, t_atom, xt, abid):
        h, vec, _, _, _ = self.net(z=atype, t=t_atom, pos=xt, batch=abid)
        vec = vec.transpose(1, 2)   # (N, H, 3)
        eu_out_fwd = torch.sum(self.eu_fwd_weight(h).unsqueeze(2) * vec, dim=1) / (1 - t_atom + 1e-6) / self.eu_scale
        eu_out_rev = torch.sum(self.eu_rev_weight(h).unsqueeze(2) * vec, dim=1) / (-t_atom + 1e-6) / self.eu_scale
        return eu_out_fwd, eu_out_rev

    def ode(self, batch, ode_step=50, guidance=None):
        x0, atype, abid = batch["x0"], batch["atype"], batch["abid"]
        N = x0.shape[0]

        xt = x0.clone()

        ode_step = np.linspace(0, 1. - 1. / ode_step, ode_step)
        dt = ode_step[1] - ode_step[0]

        for t in ode_step:
            t_atom = torch.ones(N, 1, dtype=x0.dtype, device=x0.device) * t

            # conv
            h, vec, _, _, _ = self.net(z=atype, t=t_atom, pos=xt, batch=abid)
            vt = torch.sum(self.eu_fwd_weight(h).unsqueeze(2) * vec.transpose(1, 2), dim=1) / (1 - t) / self.eu_scale
            st = 1 # 1 + 10 * t
            xt = xt + st * vt * dt + self.eu_matcher.sigma * np.sqrt(dt) * torch.randn_like(xt) \
                if t != ode_step[-1] else xt + st * vt * dt

        return xt


class GeomFSFM(nn.Module):
    def __init__(self, baseline, hidden_dim, rbf_dim, heads, layers, cutoff=5.0):
        super(GeomFSFM, self).__init__()

        self.base: GeomSFM = baseline
        # froze parameters of baseline model
        for param in self.base.parameters():
            param.requires_grad = False

        self.hidden_dim = hidden_dim
        self.rbf_dim = rbf_dim
        self.heads = heads
        self.layers = layers
        self.cutoff = cutoff

        # TorchMD-net
        self.net = TorchMD_ET(
            hidden_channels=hidden_dim, num_layers=layers, num_rbf=rbf_dim, num_heads=heads,
            cutoff_lower=0.0, cutoff_upper=cutoff, max_z=NUM_ATOM_TYPE
        )

        self.bd0_weight = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.bd1_weight = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.iff_weight = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        _init_linear_(self.bd0_weight)
        _init_linear_(self.bd1_weight)
        _init_linear_(self.iff_weight)

        # scaling for numerical stability
        self.iff_scale = 0.1

    def _train(self, batch):
        x0, x1, atype, abid = batch["x0"], batch["x1"], batch["atype"], batch["abid"]
        N = x0.shape[0]

        # uniformly sample time step
        t = np.random.rand()  # use the same time step for all mini-batches
        t_atom = torch.ones(N, 1, dtype=x0.dtype, device=x0.device) * t

        # sample noised data
        xt, _, _ = self.base.eu_matcher.sample_location_and_conditional_flow(x0, x1, t_atom)

        # intermediate force field
        pot0, pot1 = batch["pot0"], batch["pot1"]
        eu_out_fwd, eu_out_rev = self.base.fwd(atype, t_atom, xt, abid)

        # conv
        h, vec, _, _, _ = self.net(z=atype, t=t_atom, pos=xt, batch=abid)

        # boundary condition
        bd0_out = torch.sum(self.bd0_weight(h).unsqueeze(2) * vec.transpose(1, 2), dim=1)   # (N, 3)
        bd1_out = torch.sum(self.bd1_weight(h).unsqueeze(2) * vec.transpose(1, 2), dim=1)   # (N, 3)
        iff_out = torch.sum(self.iff_weight(h).unsqueeze(2) * vec.transpose(1, 2), dim=1)   # (N, 3)
        iff = (1 - t) * bd0_out.detach() + t * bd1_out.detach() + t * (1 - t) * iff_out
        loss_bound = F.mse_loss(bd0_out, -batch["force0"], reduction='mean') + \
                     F.mse_loss(bd1_out, -batch["force1"], reduction='mean')

        target = torch.exp(-(pot0 + pot1))[abid] * (
                self.base.eu_matcher.vector_field_to_score(eu_out_fwd, eu_out_rev, t) -
                self.base.eu_matcher.conditional_score(xt, x0, x1, t)
        )
        deno = 0
        for i in range(pot0.shape[0]):
            x0i, x1i, xti = x0[abid == i], x1[abid == i], xt[abid == i]
            deno += self.base.eu_matcher.likelihood(xti, x0i, x1i, t) * torch.exp(-(pot0[i] + pot1[i]))
        deno /= pot0.shape[0]
        target /= deno

        # print(f"t: {t}, force0: {batch['force0'].std()}, bd0: {bd0_out.std()},\n"
        #       f"target: {self.iff_scale * target.std()}, iff_out: {iff_out.std()}, iff: {iff.std()}")

        loss_iff = F.mse_loss(iff, (self.iff_scale * target).clip(min=-5.0, max=5.0), reduction='mean')

        loss = loss_iff + loss_bound

        return loss, (loss_iff, loss_bound)

    def ode(self, batch, ode_step=50, guidance=0.1):
        x0, atype, abid = batch["x0"], batch["atype"], batch["abid"]
        N = x0.shape[0]

        xt = x0.clone()

        ode_step = np.linspace(0, 1. - 1. / ode_step, ode_step)
        dt = ode_step[1] - ode_step[0]

        for t in ode_step:
            t_atom = torch.ones(N, 1, dtype=x0.dtype, device=x0.device) * t

            eu_out_fwd, eu_out_rev = self.base.fwd(atype, t_atom, xt, abid)

            # conv
            h, vec, _, _, _ = self.net(z=atype, t=t_atom, pos=xt, batch=abid)
            bd0_out = torch.sum(self.bd0_weight(h).unsqueeze(2) * vec.transpose(1, 2), dim=1)  # (N, 3)
            bd1_out = torch.sum(self.bd1_weight(h).unsqueeze(2) * vec.transpose(1, 2), dim=1)  # (N, 3)
            iff_out = torch.sum(self.iff_weight(h).unsqueeze(2) * vec.transpose(1, 2), dim=1)  # (N, 3)
            iff = (1 - t) * bd0_out + t * bd1_out + t * (1 - t) * iff_out
            iff /= self.iff_scale

            vt = eu_out_fwd - guidance * iff
            st = 1 # 1 + 10 * t
            xt = xt + st * vt * dt + self.base.eu_matcher.sigma * np.sqrt(dt) * torch.randn_like(xt) \
                if t != ode_step[-1] else xt + st * vt * dt

        return xt
