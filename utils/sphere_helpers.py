import torch


def _vector_norm(v):
    """
    This function normalizes vector v to unit vector nv.
    :param v: (B, *)
    """
    nv = v / torch.norm(v, dim=-1, keepdim=True)
    return nv


def sp_expmap(base, tangent):
    """
    This function is the exponential map from tangent space at base.
    :param base: (B, 3)
    :param tangent: (B, 3)
    """
    v_norm = torch.norm(tangent, dim=-1, keepdim=True)  # (B, 1)
    nv = tangent / v_norm
    return torch.cos(v_norm) * base + torch.sin(v_norm) * nv


def sp_logmap(base, target):
    """
    This function is the logarithmic map from base to target.
    :param base: (B, 3)
    :param target: (B, 3)
    """
    dot = torch.sum(base * target, dim=-1, keepdim=True)    # (B, 1)
    tangent = _vector_norm(target - dot * base)
    return torch.arccos(dot) * tangent


def sp_proj(base, v):
    """
    This function projects vector v onto tangent space at base.
    :param base: (B, 3)
    :param v: (B, 3)
    """
    dot = torch.sum(base * v, dim=-1, keepdim=True) # (B, 1)
    return v - dot * base
