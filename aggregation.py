"""
aggregation.py — Token aggregation and feature extraction.

Strategy:
  - Aggregate: mean-pool the LAST N real tokens (the response region)
    from the last transformer layer.
  - Geometric features: per-layer L2 norms over response tokens, plus
    inter-layer cosine drift summarising representation evolution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


RESPONSE_TOKENS = 32  # number of trailing real tokens to pool over


def _response_mean(layer: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool the last RESPONSE_TOKENS real tokens of `layer`."""
    mask = mask.to(device=layer.device, dtype=layer.dtype)
    real_idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if real_idx.numel() == 0:
        return torch.zeros(layer.size(-1), device=layer.device, dtype=layer.dtype)
    last_real = real_idx[-RESPONSE_TOKENS:]
    return layer[last_real].mean(dim=0)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    attention_mask = attention_mask.to(hidden_states.device)
    return _response_mean(hidden_states[-1], attention_mask)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    attention_mask = attention_mask.to(hidden_states.device)
    pooled_per_layer = []
    for layer_idx in range(hidden_states.size(0)):
        pooled_per_layer.append(_response_mean(hidden_states[layer_idx], attention_mask))
    pooled_stack = torch.stack(pooled_per_layer, dim=0)

    norms = pooled_stack.norm(dim=-1)

    drifts = []
    for i in range(1, pooled_stack.size(0)):
        cos = F.cosine_similarity(
            pooled_stack[i].unsqueeze(0),
            pooled_stack[i - 1].unsqueeze(0),
        ).squeeze(0)
        drifts.append(1.0 - cos)
    drifts = torch.stack(drifts) if drifts else torch.zeros(0, device=norms.device)

    seq_len_real = attention_mask.sum().to(norms.dtype).unsqueeze(0)

    return torch.cat([norms, drifts, seq_len_real], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)
    return agg_features
