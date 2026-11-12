"""Spike-Timing Correction baseline_v2 model driver (also reproduces baseline_v1 in legacy mode).

The per-tick LIF loop and the all-pairs STDP event loop are in stc_core.c.
This module owns the training/recall schedule, weight bounds and bookkeeping.

Model specs
-----------
STEP_V1          archived baseline_v1: step CS [1.0,4.5) ms, lambda 0.5,
                 interval 15.1 ms (gap 50 ticks), legacy gap, no bound.
EXP100_V1        previous exponential model: I0 exp(-(t-1)/8), natural decay,
                 lambda 0.25, interval 100 ms (gap 899 ticks), legacy gap.
Corrected variants are built with `spec(..., gap_mode=1)`.
"""
from __future__ import annotations

import ctypes
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "stc_core.c"
LIBRARY = HERE.parent / "build" / "stc_core.so"
V2 = HERE.parent
BASELINE_V1 = V2.parent / "baseline_v1"   # read-only: archived STDP class used by reference_model.py
N = 15
MAXEV = 160
TRIAL_TICKS = 101
DT = 0.1
P_D = ctypes.POINTER(ctypes.c_double)
P_I = ctypes.POINTER(ctypes.c_int64)


class Params(ctypes.Structure):
    _fields_ = [("exponential", ctypes.c_int), ("truncate", ctypes.c_int),
                ("cs_end", ctypes.c_int), ("gap_mode", ctypes.c_int),
                ("interval", ctypes.c_int64), ("factor", ctypes.c_double),
                ("q", ctypes.c_double), ("rate", ctypes.c_double),
                ("noise_sigma", ctypes.c_double)]


class History(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64 * (N * 1024)) for name in
                ("pre_tick", "pre_trial", "post_tick", "post_trial", "post_gap")] + [
        ("pre_count", ctypes.c_int64 * N), ("post_count", ctypes.c_int64 * N),
        ("total_pairs", ctypes.c_int64), ("cross_pairs", ctypes.c_int64)]


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        if not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
            subprocess.run(["gcc", "-O2", "-fno-fast-math", "-ffp-contract=off", "-shared",
                            "-fPIC", str(SOURCE), "-lm", "-o", str(LIBRARY)], check=True)
        L = ctypes.CDLL(str(LIBRARY))
        L.run_interval.argtypes = [ctypes.POINTER(Params), P_I, P_D, P_I, P_D, P_I, P_D,
                                   ctypes.c_int64, ctypes.c_int, P_D, P_I, P_I, P_I, P_D, P_D]
        L.run_interval.restype = ctypes.c_int
        L.stdp_process.argtypes = [ctypes.POINTER(History), ctypes.POINTER(Params), P_I, P_I,
                                   P_I, ctypes.c_int64, ctypes.c_int64, P_D, P_D]
        L.stdp_process.restype = ctypes.c_int
        _LIB = L
    return _LIB


def _pi(a):
    assert a.dtype == np.int64 and a.flags.c_contiguous
    return a.ctypes.data_as(P_I)


def _pd(a):
    assert a.dtype == np.float64 and a.flags.c_contiguous
    return a.ctypes.data_as(P_D)


@dataclass(frozen=True)
class Spec:
    name: str
    exponential: bool
    interval: int            # ticks per association
    rate: float
    gap_mode: int            # 0 legacy, 1 corrected
    bound: float | None = None   # symmetric weight bound, None = unbounded
    tau_s: float = 8.0
    factor: float = 1.0
    truncate: bool = False
    cs_end: int = 45

    def params(self, noise_sigma=0.0):
        q = float(np.exp(-DT / self.tau_s)) if self.exponential else 0.0
        return Params(int(self.exponential), int(self.truncate), self.cs_end, self.gap_mode,
                      self.interval, self.factor, q, self.rate, noise_sigma)

    @property
    def interval_ms(self):
        return self.interval * DT


STEP_V1 = Spec("Step v1 (15.1 ms, legacy gap, unbounded)", False, 151, 0.5, 0)
EXP100_V1 = Spec("Exponential v1 (100 ms, legacy gap, unbounded)", True, 1000, 0.25, 0)

# ---- baseline_v2 (official from 2026-11-12) ----
STEP_V2 = Spec("Step v2 (15.0 ms, corrected gap, W in [-1,1])", False, 150, 0.5, 1, 1.0)
EXP_V2 = Spec("Exponential v2 (100 ms, tau_s 8 ms, corrected gap, W in [-1,1])", True, 1000, 0.25, 1, 1.0)


def variant(base: Spec, name=None, **changes):
    s = replace(base, **changes)
    return replace(s, name=name or s.name)


# ---------------------------------------------------------------- data

def load_reference_dataset(name):
    with np.load(V2 / "datasets" / f"{name.lower()}_K10.npz") as d:
        return (d["input_binary"].astype(np.int64), d["target_binary"].astype(np.int64),
                d["presentation_orders"].astype(np.int64))


def window(dt_ms):
    u = dt_ms / 1.286893592
    return .170690168 * (1. - u * u) * math.exp(-.5 * u * u)


# ---------------------------------------------------------------- state

class State:
    def __init__(self):
        self.v = np.zeros(N)
        self.refr = np.full(N, -1000000000, dtype=np.int64)
        self.tail = np.zeros(N)
        self.hist = History()
        self.tick = 0
        self.trial = 0

    def neuron_copy(self):
        return self.v.copy(), self.refr.copy(), self.tail.copy()


class Interval:
    """Result buffers of one association interval."""

    def __init__(self):
        self.ev = np.full((N, MAXEV), -1, dtype=np.int64)
        self.count = np.zeros(N, dtype=np.int64)
        self.pred = np.zeros(N, dtype=np.int64)
        self.maxima = np.zeros(N)
        self.stats = np.zeros(3)

    def events(self, j):
        return self.ev[j, :self.count[j]]


def run_interval(params, x, w, y, v, refr, tail, base, teacher_on, noise=None, out=None):
    out = out or Interval()
    code = lib().run_interval(ctypes.byref(params), _pi(x), _pd(w), _pi(y), _pd(v), _pi(refr),
                              _pd(tail), int(base), int(teacher_on),
                              _pd(noise) if noise is not None else None,
                              _pi(out.ev), _pi(out.count), _pi(out.pred), _pd(out.maxima),
                              _pd(out.stats))
    if code:
        raise RuntimeError(f"run_interval failed ({code}): spike buffer overflow")
    return out


def stdp(state, params, x_or_pre, iv, base, trial, delta=None, stats=None):
    pre = np.where(x_or_pre != 0, 10, -1).astype(np.int64)
    delta = np.empty((N, N)) if delta is None else delta
    stats = np.empty(5) if stats is None else stats
    code = lib().stdp_process(ctypes.byref(state.hist), ctypes.byref(params), _pi(pre),
                              _pi(iv.ev), _pi(iv.count), int(base), int(trial), _pd(delta),
                              _pd(stats))
    if code:
        raise RuntimeError(f"stdp_process failed ({code}): history overflow")
    return delta, stats


# ---------------------------------------------------------------- recall

def recall(spec, x, y, w, state, noise_sigma=0.0, seed=None, keep=False):
    """Teacher-free sequential readout on a COPY of the continuous state.

    Bit = 1 if the output neuron spikes in the response window k=1..100.
    Gap spikes (corrected mode) are counted separately and affect the state.
    """
    params = spec.params(noise_sigma)
    v, refr, tail = state.neuron_copy()
    rng = np.random.default_rng(seed) if noise_sigma else None
    zero = np.zeros(N, dtype=np.int64)
    K = len(x)
    pred = np.zeros((K, N), dtype=np.int64)
    maxima = np.zeros((K, N))
    counts = np.zeros((K, N), dtype=np.int64)
    first = np.full((K, N), np.nan)
    gap_spikes = gap_exceed = 0
    events = [] if keep else None
    for a in range(K):
        noise = rng.standard_normal((100, N)) * noise_sigma if rng is not None else None
        iv = run_interval(params, x[a], w, zero, v, refr, tail, state.tick + a * spec.interval,
                          False, noise)
        pred[a] = iv.pred
        maxima[a] = iv.maxima
        for j in range(N):
            e = iv.events(j)
            r = e[(e > 0) & (e < TRIAL_TICKS)]
            counts[a, j] = len(r)
            if len(r):
                first[a, j] = r[0] * DT
        gap_exceed += int(iv.stats[0])
        gap_spikes += int(iv.stats[2])
        if keep:
            events.append([iv.events(j).copy() for j in range(N)])
    exact = int(np.all(pred == y, axis=1).sum())
    return dict(pred=pred, maxima=maxima, counts=counts, first=first, exact=exact,
                bit=float(np.mean(pred == y)), fp=int(np.sum((pred == 1) & (y == 0))),
                fn=int(np.sum((pred == 0) & (y == 1))), gap_spikes=gap_spikes,
                gap_exceed=gap_exceed, events=events)


# ---------------------------------------------------------------- training

def train(spec, x, y, orders, epochs, *, state=None, w=None, epoch_offset=0,
          noise_sigma=0.0, noise_rng=None, keep_events=False, keep_weights=(),
          detail_epochs=(), stop_on_legacy_exceed=False, progress=None):
    """Continuous training; ΔW applied once at the end of each association interval.

    orders: array (E, K); epoch e uses orders[(e-1) % E].
    Returns (rows, w, state, extra).
    """
    params = spec.params(noise_sigma)
    state = state or State()
    w = np.zeros((N, N)) if w is None else w
    K = orders.shape[1]
    lo, hi = (-spec.bound, spec.bound) if spec.bound is not None else (None, None)
    rows = []
    extra = dict(events=None, counts=None, weights={}, details={}, first_clip=None,
                 first_gap_spike=None, first_legacy_exceed=None)
    if keep_events:
        extra["events"] = np.full((epochs, K, N, 12), -1, dtype=np.int16)
        extra["counts"] = np.zeros((epochs, K, N), dtype=np.int16)
    iv = Interval()
    delta = np.empty((N, N))
    sstats = np.empty(5)
    at_bound_prev = np.zeros((N, N), dtype=bool)
    for e_local in range(epochs):
        epoch = epoch_offset + e_local + 1
        order = orders[(epoch - 1) % len(orders)]
        before = w.copy()
        m = dict(early=0, positive_window=0, extra=0, natural=0, missing=0, teacher_fail=0,
                 gap_exceed=0, gap_spikes=0, gap_vmax=-np.inf, gap_dw=0., gap_dw_abs=0.,
                 gap_pairs=0, ltp=0., ltd=0., clip_events=0, nonzero_updates=0,
                 clip_excess=0., outward_at_bound=0, raw_update_sq=0., clipped_raw_max=0.)
        clipped_any = np.zeros((N, N), dtype=bool)
        raw_sum = np.zeros((N, N))
        gap_times = []
        detail = [] if epoch in detail_epochs else None
        for pos, a in enumerate(order):
            state.trial += 1
            noise = (noise_rng.standard_normal((100, N)) * noise_sigma
                     if noise_sigma else None)
            base = state.tick
            w_before_assoc = w.copy() if detail is not None else None
            run_interval(params, x[a], w, y[a], state.v, state.refr, state.tail, base, True,
                         noise, iv)
            for j in range(N):
                ev = iv.events(j)
                teacher = len(ev) > 0 and ev[0] == 0
                if teacher != bool(y[a, j]):
                    m["teacher_fail"] += 1
                nat = ev[(ev > 0) & (ev < TRIAL_TICKS)]
                m["natural"] += len(nat)
                m["extra"] += max(0, len(nat) - 1)
                m["early"] += int(np.sum(nat < 40))
                m["positive_window"] += int(np.sum(np.abs(nat - 10) * DT < 1.286893592))
                if y[a, j] and not len(nat):
                    m["missing"] += 1
                g = ev[ev >= TRIAL_TICKS]
                if len(g):
                    gap_times.extend((g * DT).tolist())
                    if extra["first_gap_spike"] is None:
                        extra["first_gap_spike"] = dict(epoch=epoch, position=pos, association=int(a),
                                                        output=j, time_ms=float(g[0] * DT))
                if keep_events:
                    t = ev[ev < TRIAL_TICKS][:12]
                    extra["events"][e_local, pos, j, :len(t)] = t
                    extra["counts"][e_local, pos, j] = len(ev[ev < TRIAL_TICKS])
            m["gap_exceed"] += int(iv.stats[0])
            m["gap_spikes"] += int(iv.stats[2])
            m["gap_vmax"] = max(m["gap_vmax"], iv.stats[1])
            if iv.stats[0] and extra["first_legacy_exceed"] is None:
                extra["first_legacy_exceed"] = dict(epoch=epoch, position=pos, association=int(a),
                                                    steps=int(iv.stats[0]))
            stdp(state, params, x[a], iv, base, state.trial, delta, sstats)
            m["ltp"] += sstats[0]; m["ltd"] += sstats[1]
            m["gap_dw"] += sstats[2]; m["gap_dw_abs"] += sstats[3]; m["gap_pairs"] += int(sstats[4])
            raw_sum += delta
            nz = delta != 0.
            m["nonzero_updates"] += int(nz.sum())
            if lo is None:
                w += delta
            else:
                attempt = w + delta
                outside = (attempt > hi) | (attempt < lo)
                if outside.any():
                    m["clip_events"] += int(outside.sum())
                    clipped_any |= outside
                    m["clip_excess"] += float(np.sum(np.abs(attempt - np.clip(attempt, lo, hi))))
                    m["clipped_raw_max"] = max(m["clipped_raw_max"], float(np.abs(delta[outside]).max()))
                    on_bound = ((w >= hi) & (delta > 0)) | ((w <= lo) & (delta < 0))
                    m["outward_at_bound"] += int(on_bound.sum())
                    if extra["first_clip"] is None:
                        i, j = map(int, np.argwhere(outside)[0])
                        extra["first_clip"] = dict(epoch=epoch, position=pos, association=int(a),
                                                   input=i, output=j, w_old=float(w[i, j]),
                                                   delta=float(delta[i, j]))
                np.clip(attempt, lo, hi, out=w)
            if not np.isfinite(w).all():
                raise FloatingPointError(f"non-finite weight at epoch {epoch}")
            if detail is not None:
                detail.append(dict(position=pos, association=int(a), base=base,
                                   events=[iv.events(j).copy() for j in range(N)],
                                   delta=delta.copy(), w_before=w_before_assoc, w_after=w.copy(),
                                   maxima=iv.maxima.copy(), gap_stats=iv.stats.copy()))
            state.tick += spec.interval
        r = recall(spec, x, y, w, state)
        upd = w - before
        at_bound = (w >= hi) | (w <= lo) if lo is not None else np.zeros((N, N), dtype=bool)
        target0 = y == 0
        row = dict(epoch=epoch, exact_count=r["exact"], exact_accuracy_percent=100. * r["exact"] / len(x),
                   bit_accuracy_percent=100. * r["bit"], false_positive_bits=r["fp"],
                   false_negative_bits=r["fn"], weight_min=float(w.min()), weight_max=float(w.max()),
                   weight_norm=float(np.linalg.norm(w)), weight_update_norm=float(np.linalg.norm(upd)),
                   attempted_update_norm=float(np.linalg.norm(raw_sum)),
                   weight_change_max=float(np.abs(upd).max()),
                   early_natural_spikes=m["early"], positive_window_natural_spikes=m["positive_window"],
                   natural_spikes=m["natural"], additional_natural_spikes=m["extra"],
                   missing_target1_spikes=m["missing"], teacher_spike_failures=m["teacher_fail"],
                   training_gap_spikes=m["gap_spikes"], training_gap_threshold_exceedances=m["gap_exceed"],
                   training_gap_max_voltage=float(m["gap_vmax"]),
                   mean_gap_spike_time_ms=float(np.mean(gap_times)) if gap_times else "",
                   recall_gap_spikes=r["gap_spikes"], recall_gap_threshold_exceedances=r["gap_exceed"],
                   gap_stdp_contribution=m["gap_dw"], gap_stdp_contribution_abs=m["gap_dw_abs"],
                   gap_stdp_pairs=m["gap_pairs"], ltp_sum=m["ltp"], ltd_sum=m["ltd"],
                   clip_events=m["clip_events"], clipped_synapses=int(clipped_any.sum()),
                   nonzero_updates=m["nonzero_updates"],
                   clip_fraction=(m["clip_events"] / m["nonzero_updates"]) if m["nonzero_updates"] else 0.,
                   clip_excess_sum=m["clip_excess"], clipped_raw_update_max=m["clipped_raw_max"],
                   outward_updates_at_bound=m["outward_at_bound"],
                   synapses_at_bound=int(at_bound.sum()),
                   min_target0_margin=float((1 - r["maxima"][target0]).min()) if target0.any() else "")
        rows.append(row)
        at_bound_prev = at_bound
        if epoch in keep_weights:
            extra["weights"][epoch] = w.copy()
        if detail is not None:
            extra["details"][epoch] = dict(train=detail, recall=recall(spec, x, y, w, state, keep=True))
        if progress and epoch % progress == 0:
            print(spec.name, epoch, r["exact"], flush=True)
        if stop_on_legacy_exceed and m["gap_exceed"]:
            extra["stopped"] = epoch
            break
    return rows, w, state, extra


def first_and_sustained(rows, K=10):
    full = [int(r["exact_count"]) == K for r in rows]
    first = next((i + 1 for i, f in enumerate(full) if f), None)
    sustained = None
    if full and full[-1]:
        last_bad = max((i for i, f in enumerate(full) if not f), default=-1)
        sustained = last_bad + 2
    return first, sustained
