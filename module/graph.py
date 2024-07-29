import torch
from torch import nn
import numpy as np


def _topk(input, k, dim=None, largest=True):
    """
    This function allows for repeated indices in case k is out of range.
    """
    if input.shape[1] >= k:
        return torch.topk(input, k, dim=dim, largest=largest)
    else:
        sorted_values, sorted_indices = torch.sort(input, descending=largest, dim=dim)
        num_repeats = (k // input.shape[1]) + 1
        sorted_values = sorted_values.repeat(1, num_repeats)[:, :k]
        sorted_indices = sorted_indices.repeat(1, num_repeats)[:, :k]
        return sorted_values, sorted_indices


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def _construct_knn_graph(X, rule_mats, k_neighbors):
    """
    :param X: (N, 3), coordinates
    :param rule_mats: list of (N, N), valid edges after each filtering
    :param k_neighbors: neighbors of each node
    """
    src_dst = torch.nonzero(sequential_and(*rule_mats))  # (Ef, 2), full possible edges represented in (src, dst)
    BIGINT = 1e10  # assign a large distance to invalid edges
    N = X.shape[0]
    dist = X[src_dst]  # (Ef, 2, 3)
    dist = dist[:, 0] - dist[:, 1]      # (Ef, 3)
    dist = torch.norm(dist, dim=-1)     # (Ef,)
    src_dst = src_dst.transpose(0, 1)  # (2, Ef)
    dist[src_dst[0] == src_dst[1]] += BIGINT    # rule out i2i
    dist = (torch.ones(N, N, dtype=dist.dtype, device=dist.device) * BIGINT).index_put_(tuple([k for k in src_dst]), dist)
    # dist_neighbors: (N, topk), dst: (N, topk)
    dist_neighbors, dst = _topk(dist, k_neighbors, dim=-1, largest=False)  # (N, topk)
    del dist  # release memory
    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)
    edges = torch.stack([src, dst])
    return edges  # (2, E)


def _construct_cutoff_graph(X, rule_mats, cutoff):
    """
    :param X: (N, 3), coordinates
    :param rule_mats: list of (N, N), valid edges after each filtering
    :param cutoff: cutoff threshold
    """
    src_dst = torch.nonzero(sequential_and(*rule_mats))  # (Ef, 2), full possible edges represented in (src, dst)
    BIGINT = 1e10  # assign a large distance to invalid edges
    N = X.shape[0]
    dist = X[src_dst]  # (Ef, 2, 3)
    dist = dist[:, 0] - dist[:, 1]  # (Ef, 3)
    dist = torch.norm(dist, dim=-1)  # (Ef,)
    src_dst = src_dst.transpose(0, 1)  # (2, Ef)
    dist[src_dst[0] == src_dst[1]] += BIGINT  # rule out i2i
    dist = (torch.ones(N, N, dtype=dist.dtype, device=dist.device) * BIGINT).index_put_(tuple([k for k in src_dst]), dist)
    edges = torch.nonzero(dist < cutoff).transpose(0, 1) # (2, E)
    del dist
    return edges


def construct_edges(xlocal, n, abid, rbid, rmask, gmask,
                    cutoff=0.4, k_neighbors=5):
    N = abid.shape[0]
    # same batch (for atoms)
    same_abid = abid.unsqueeze(-1).repeat(1, N)
    same_abid = same_abid == same_abid.transpose(0, 1) # (N, N)
    # same residue
    same_res = rmask.unsqueeze(-1).repeat(1, N)
    same_res = same_res == same_res.transpose(0, 1) # (N, N)
    # rule out global node
    same_loc = gmask.unsqueeze(-1).repeat(1, N)
    same_loc = same_loc == same_loc.transpose(0, 1) # (N, N)

    intra_edges = _construct_cutoff_graph(
        xlocal,
        [same_abid, same_res, same_loc],
        cutoff
    )
    # global node is fully-connected to all atoms in the same residue
    glob_edges = torch.nonzero(sequential_and(same_abid, same_res, ~same_loc)).transpose(0, 1).to(xlocal.device)
    intra_edges = torch.cat([glob_edges, intra_edges], dim=1)

    # same batch (for residues)
    B = rbid.shape[0]
    same_rbid = rbid.unsqueeze(-1).repeat(1, B)
    same_rbid = same_rbid == same_rbid.transpose(0, 1)  # (B, B)
    # inter global node, NOTE THAT the indexes are based on global nodes!
    inter_edges = _construct_knn_graph(
        n,  # N coordinates, (B, 3)
        [same_rbid],
        k_neighbors
    )

    return intra_edges, inter_edges
