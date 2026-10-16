"""Rectified flow helpers (R1 plan §5)."""

from __future__ import annotations

from typing import Callable, List, Optional

import torch


def sample_xt(x_clean: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """x_t = (1-t)*x_clean + t*eps. t broadcastable to x."""
    while t.ndim < x_clean.ndim:
        t = t.unsqueeze(-1)
    return (1.0 - t) * x_clean + t * eps


def velocity_target(x_clean: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """v* = eps - x_clean."""
    return eps - x_clean


@torch.no_grad()
def euler_sample(
    model_fn: Callable,
    x_init: torch.Tensor,
    steps: int = 20,
    **model_kwargs,
) -> torch.Tensor:
    """Euler from t=1 to t=0. model_fn(x, t, **kw) -> velocity."""
    x = x_init
    ts = torch.linspace(1.0, 0.0, steps + 1, device=x.device, dtype=torch.float32)
    for i in range(steps):
        t = ts[i].expand(x.shape[0])
        t_next = ts[i + 1]
        v = model_fn(x, t, **model_kwargs)
        dt = t_next - ts[i]  # negative
        x = x + dt * v
    return x
