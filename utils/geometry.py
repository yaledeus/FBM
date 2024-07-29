import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from e3nn.o3 import rand_matrix
from .constants import *


def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign((torch.cross(v1, v2, dim=-1) * v0).sum(-1))
    dihed = sgn * torch.acos((n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    dihed = torch.nan_to_num(dihed)
    return dihed


def pairwise_dihedral(coord, edges):
    """
    Args:
        coord: (N, n_channel, 3).
        edges: (2, n_edges).
    Returns:
        Inter-residue Phi and Psi angles, (n_edges, 2).
    """
    src, dst = edges
    coord_N_src = coord[src, 0]  # (n_edge, 3)
    coord_N_dst = coord[dst, 0]
    coord_CA_src = coord[src, 1]
    coord_C_src = coord[src, 2]
    coord_C_dst = coord[dst, 2]

    ir_phi = dihedral_from_four_points(
        coord_C_dst,
        coord_N_src,
        coord_CA_src,
        coord_C_src
    )
    ir_psi = dihedral_from_four_points(
        coord_N_src,
        coord_CA_src,
        coord_C_src,
        coord_N_dst
    )
    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)

    return ir_dihed


def pairwise_bb_dist(bb):
    N, num_atoms, _ = bb.shape
    bb_expanded_i = bb.unsqueeze(1).unsqueeze(3)  # (N, 1, 4, 1, 3)
    bb_expanded_j = bb.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 4, 3)
    diff = bb_expanded_i - bb_expanded_j        # (N, N, 4, 4, 3)
    D = torch.sum(diff ** 2, dim=-1).sqrt()     # (N, N, 4, 4)
    return D


def rotate_by_axis(v, axis, theta):
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().item()
    v_r = v * np.cos(theta) + torch.cross(axis, v) * np.sin(theta) + \
          axis * torch.dot(v, axis) * (1 - np.cos(theta))
    return v_r


def recover_N_from_psi(coord_n, coord_ca, coord_c, psi):
    x = coord_ca - coord_n
    y = coord_c - coord_ca
    tor_axis = y / torch.linalg.norm(y)     # torsion angle N'-C-CA-N axis
    angle_axis = torch.cross(x, tor_axis)   # bond angle CA-C-N axis
    angle_axis = angle_axis / torch.linalg.norm(angle_axis)
    d0 = tor_axis * BOND_C_N     # initialize N'
    d1 = rotate_by_axis(d0, angle_axis, np.pi - ANGLE_CA_C_N)
    d2 = rotate_by_axis(d1, tor_axis, psi)
    coord_n_next = coord_c + d2
    return coord_n_next


def recover_CB(coord_c, coord_ca, coord_n):
    x = coord_ca - coord_n
    y = coord_c - coord_ca
    z = torch.cross(x, y)
    coord_cb = -0.57910144 * z + 0.5689693 * x - 0.5441217 * y + coord_ca
    return coord_cb


def kabsch_torch(P: torch.Tensor, Q: torch.Tensor):
    """
    :param P: (N, 3)
    :param Q: (N, 3)
    :return: P @ R + t, R, t such that minimize RMSD(PR + t, Q)
    """
    P = P.double()
    Q = Q.double()

    PC = torch.mean(P, dim=0)
    QC = torch.mean(Q, dim=0)

    # centering
    UP = P - PC
    UQ = Q - QC

    # Covariance matrix
    C = UP.T @ UQ
    V, S, W = torch.linalg.svd(C)

    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0

    # avoid inplace modify
    V_R = V.clone()
    if d:
        V_R[:, -1] = -V[:, -1]

    R: torch.Tensor = V_R @ W

    t = QC - PC @ R  # (3,)

    return (UP @ R + QC), R, t


"""
# test for kabsch algorithm
# Test Case 1: Simple test case
A = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float)
B = A @ torch.diag(torch.tensor([1, 1, -1]).float()) # Rotate A 180 degrees around Z-axis and flip Z
C, R, t = kabsch_torch(B, A)
assert np.allclose(C, A)

# Test Case 2: Random point cloud with uniform translation and rotation
np.random.seed(0)
A = torch.tensor(np.random.rand(100, 3), dtype=torch.float)
R = rand_matrix()
t = torch.tensor(np.random.rand(3), dtype=torch.float)
B = A @ R + t
C, R_est, t_est = kabsch_torch(B, A)
assert np.allclose(C, A, atol=1e-3)

# Test Case 3: Identity transformation
A = torch.tensor(np.random.rand(100, 3), dtype=torch.float)
B = A
C, R_est, t_est = kabsch_torch(B, A)
assert np.allclose(C, A, atol=1e-4)
assert np.allclose(R_est, np.eye(3), atol=1e-4)
assert np.allclose(t_est, np.zeros(3), atol=1e-4)
"""


def kabsch_numpy(P: np.ndarray, Q: np.ndarray):
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)

    PC = np.mean(P, axis=0)
    QC = np.mean(Q, axis=0)

    UP = P - PC
    UQ = Q - QC

    C = UP.T @ UQ
    V, S, W = np.linalg.svd(C)

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        V[:, -1] = -V[:, -1]

    R: np.ndarray = V @ W

    t = QC - PC @ R  # (3,)

    return (UP @ R + QC).astype(np.float32), R.astype(np.float32), t.astype(np.float32)


def recon_bb(psi, R, rbid):
    """
    This function is to reconstruct Cartesian coordinates of backbone atoms with
    :param psi: backbone dihedral, (B, 1)
    :param R: rotation matrix, (B, 3, 3)
    :param rbid: residue batch id, (B,)
    :return: Cartesian coordinates of Ns, (B, 3)
    """
    device = psi.device
    # replace backbone coordinates with PEPTIDE PLANE
    bb = PEPTIDE_PLANE_TORCH.type_as(psi).to(device).unsqueeze(0).repeat(psi.shape[0], 1, 1)
    g_idx = 0
    for i in range(rbid[-1] + 1):   # i-th protein
        B = torch.sum(rbid == i)
        for b in range(B - 1):    # #residue of i-th protein
            N_next = recover_N_from_psi(coord_n=bb[g_idx+b, 0], coord_ca=bb[g_idx+b, 1], coord_c=bb[g_idx+b, 2],
                                        psi=psi[g_idx+b+1])
            bb[g_idx+b+1] = bb[g_idx+b+1] @ R[g_idx+b+1] + N_next
        g_idx += B

    return bb


def recon(x, psi, R, bb_index, rmask, gmask):
    """
    This function is to reconstruct Cartesian coordinates of all atoms with
    :param x: local coordinates, (N, 3)
    :param psi: backbone dihedral, (B, 1)
    :param R: rotation matrix, (B, 3, 3)
    :param bb_index: backbone index, (B, 4)
    :param rmask: residue mask, (N,)
    :param gmask: global node mask, (N,)
    :return: Cartesian coordinates without global node, (N', 3)
    """
    device = x.device
    N, B = x.shape[0], psi.shape[0]
    x0 = x[gmask]  # without global node, (N', 3)
    rmask_ng = rmask[gmask] # (N',)
    x1 = torch.zeros_like(x0)  # (N', 3)
    # replace backbone coordinates with PEPTIDE PLANE
    x0[bb_index.view(-1)] = PEPTIDE_PLANE_TORCH.type_as(x).to(device).repeat(B, 1)
    # rotate the first frame
    x1[rmask_ng == 0] = x0[rmask_ng == 0] # @ R[0]
    # autoregressive update
    for b in range(B - 1):
        N_next = recover_N_from_psi(coord_n=x1[bb_index[b, 0]], coord_ca=x1[bb_index[b, 1]], coord_c=x1[bb_index[b, 2]],
                                    psi=psi[b+1])
        x1[rmask_ng == b+1] = x0[rmask_ng == b+1] @ R[b+1] + N_next

    return x1, x0


if __name__ == "__main__":
    """
    ca = torch.tensor([-1.1011,  0.2592,  0.0224], dtype=torch.float)
    c = torch.tensor([-1.0299,  0.1896, -0.0876], dtype=torch.float)
    n = torch.tensor([-1.0994,  0.174 ,  0.1408], dtype=torch.float)
    n1 = torch.tensor([-0.9468,  0.0818, -0.0556], dtype=torch.float)
    psi = dihedral_from_four_points(n, ca, c, n1)
    n_pred = recover_N_from_psi(c, ca, n, psi)
    print(f"psi: {psi}, ground truth N: {n1}, pred: {n_pred}")
    """
