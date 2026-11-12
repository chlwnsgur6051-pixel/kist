"""Pure-Python reference of the corrected model (gap_mode=1), independent of stc_core.c.

LIF: every tick of the association interval (trial and gap) uses the same
Forward-Euler update, threshold test, reset and refractory test.
STDP: the archived baseline_v1 ContinuousAllPairs.process, unmodified, given
all post events of the interval (trial and gap ticks).
Weight bound: W = clip(W + dW, -b, b) at the end of each association interval.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np

from stc_model import BASELINE_V1, N, TRIAL_TICKS


def archived_module(rate):
    if "numba" not in sys.modules and importlib.util.find_spec("numba") is None:
        shim = types.ModuleType("numba"); shim.njit = lambda f: f
        sys.modules["numba"] = shim
    spec = importlib.util.spec_from_file_location("archived_sweep_model_ref",
                                                  BASELINE_V1 / "src/sweep_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LAMBDA = rate
    return mod


def interval(spec, x, w, y, v, refr, tail, base, teacher_on):
    eff = np.empty(N)
    for j in range(N):
        c = 0.
        for i in range(N):
            c += x[i] * w[i, j]
        eff[j] = c
    q = float(np.exp(-0.1 / spec.tau_s)) if spec.exponential else 0.
    events = [[] for _ in range(N)]
    for k in range(spec.interval):
        start = base + k - 1
        if spec.exponential:
            if spec.truncate and k == 46:
                tail[:] = 0.
            if k == 11:
                tail += spec.factor * eff
        for j in range(N):
            active = start >= refr[j]
            if spec.exponential:
                cs = tail[j]
            else:
                cs = eff[j] if 11 <= k <= spec.cs_end else 0.
            teach = 30. if (teacher_on and k == 0 and y[j] == 1) else 0.
            if active:
                v[j] += 0.1 / 2. * (-v[j] + cs + teach)
            else:
                v[j] = 0.
            if active and v[j] > 1.:
                events[j].append(k)
                v[j] = 0.
                refr[j] = base + k + 10
        if spec.exponential:
            tail *= q
    return events


def to_arrays(events):
    width = max(1, max(len(e) for e in events))
    ticks = np.full((N, width), -1, dtype=np.int64)
    counts = np.zeros(N, dtype=np.int64)
    for j, e in enumerate(events):
        ticks[j, :len(e)] = e
        counts[j] = len(e)
    return counts, ticks


def train(spec, x, y, orders, epochs):
    assert spec.gap_mode == 1
    mod = archived_module(spec.rate)
    stdp = mod.ContinuousAllPairs()
    w = np.zeros((N, N)); v = np.zeros(N); refr = np.full(N, -1000000000, dtype=np.int64)
    tail = np.zeros(N); tick = trial = 0
    log = []
    for ep in range(epochs):
        for a in orders[ep % len(orders)]:
            trial += 1
            ev = interval(spec, x[a], w, y[a], v, refr, tail, tick, True)
            counts, ticks = to_arrays(ev)
            delta = stdp.process(x[a], counts, ticks, tick, trial)
            w = w + delta
            if spec.bound is not None:
                w = np.clip(w, -spec.bound, spec.bound)
            log.append(dict(epoch=ep + 1, events=[list(e) for e in ev], delta=delta.copy()))
            tick += spec.interval
        # teacher-free recall on copies
        vv, rr, tt = v.copy(), refr.copy(), tail.copy()
        pred = np.zeros_like(y)
        for a in range(len(x)):
            ev = interval(spec, x[a], w, np.zeros(N, dtype=np.int64), vv, rr, tt,
                          tick + a * spec.interval, False)
            pred[a] = [int(any(0 < t < TRIAL_TICKS for t in e)) for e in ev]
        log[-1]["recall"] = pred.copy()
        log[-1]["weight"] = w.copy()
    return log, w, v, refr, tail
