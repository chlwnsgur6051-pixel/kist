"""T2 training paradigm: full STDP + nontarget (US-off) inhibition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingSpec:
    training_id: int
    name: str
    plasticity_gate: str  # "all" | "us_mask"
    train_shunt: bool
    gate_us_on_shunt: bool = False


# T2 only for this share package.
T2 = TrainingSpec(
    2,
    "stdp_nontarget_inhibition",
    plasticity_gate="all",
    train_shunt=True,
)


@dataclass
class TrialContext:
    us_mask: np.ndarray
    t_us: float


def effective_stdp_factor(
    factor: float,
    spec: TrainingSpec,
    post_j: int,
    ctx: TrialContext,
) -> float:
    if spec.plasticity_gate == "all":
        return float(factor)
    if spec.plasticity_gate == "us_mask":
        if ctx.us_mask[post_j] < 0.5:
            return 1.0
        return float(factor)
    return 1.0


def us_step_bounds(
    dt: float,
    t_us: float,
    us_duration: float,
    n_steps: int | None = None,
) -> tuple[int, int]:
    us_step = max(0, int(round(t_us / dt)))
    us_steps = max(1, int(round(us_duration / dt)))
    us_step_end = us_step + us_steps
    if n_steps is not None:
        us_step_end = min(us_step_end, n_steps)
    return us_step, us_step_end


def shunt_active(
    spec: TrainingSpec,
    step: int,
    hidden_idx: int,
    us_mask: np.ndarray,
    *,
    shunt_lo_step: int,
    shunt_hi_step: int,
    us_step: int,
    us_step_end: int,
) -> bool:
    """Per-trial shunt: inhibit US-off (target=0) hidden cells."""
    if not spec.train_shunt:
        return False
    if not (shunt_lo_step <= step < shunt_hi_step):
        return False
    if us_mask[hidden_idx] >= 0.5:
        return False
    return True


def shunt_active_training_epoch(
    spec: TrainingSpec,
    step: int,
    hidden_idx: int,
    us_mask: np.ndarray | None,
    us_step: int,
    us_step_end: int,
    *,
    in_pair_gap: bool,
) -> bool:
    """Epoch timeline: gaps shunt all hidden; within pair, shunt US-off only (T2)."""
    if not spec.train_shunt:
        return False
    if in_pair_gap:
        return True
    assert us_mask is not None
    if us_mask[hidden_idx] >= 0.5:
        return False
    return True
