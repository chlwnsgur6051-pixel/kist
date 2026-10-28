"""Clocked LIF trial dynamics, copied from the validated simulator path.

Only the validated clocked mode is exposed. The Euler update, event ordering,
threshold comparison, reset, and refractory calculations are unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LIFConfig:
    dt_ms: float = 0.1
    tau_membrane_ms: float = 2.0
    tau_synaptic_current_ms: float = 1.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    refractory_ms: float = 1.0
    input_gain: float = 1.0
    threshold: float = 0.01
    teacher_current: float = 30.0


def simulate_trial(x, w, cfg: LIFConfig, timing, teacher=None, record_trace=False):
    """One binary cue, one clocked prediction, optional later teacher current.

    The target can influence only the teacher phase. Every active input neuron
    fires once at timing['pre']; input-zero neurons contribute no current.
    """
    x=np.asarray(x);w=np.asarray(w)
    assert x.shape==(15,) and w.shape==(15,15)
    if teacher is not None:assert np.asarray(teacher).shape==(15,)
    steps=int(round((timing['teacher']+2.)/cfg.dt_ms))+1
    pre_step=int(round(timing['pre']/cfg.dt_ms))
    read_step=int(round(timing['prediction']/cfg.dt_ms))
    teacher_step=int(round(timing['teacher']/cfg.dt_ms))
    refractory_steps=int(round(cfg.refractory_ms/cfg.dt_ms))
    v=np.full(15,cfg.v_rest,dtype=float)
    syn=np.zeros(15)
    refractory_until=np.zeros(15,dtype=np.int32)
    events=[[] for _ in range(15)]
    kinds=[[] for _ in range(15)]
    trace_v=np.empty((steps,15),dtype=np.float32) if record_trace else None
    trace_i=np.empty((steps,15),dtype=np.float32) if record_trace else None
    teacher_emitted=np.zeros(15,dtype=np.int8)
    suppressed=np.zeros(15,dtype=np.int8)
    injected_current=cfg.input_gain*(x@w)
    for step in range(steps):
        t=round(step*cfg.dt_ms,10)
        if step==pre_step:syn+=injected_current
        teacher_pulse=np.zeros(15)
        if teacher is not None and step==teacher_step:
            teacher_pulse=cfg.teacher_current*teacher
            suppressed=((teacher!=0)&(step<refractory_until)).astype(np.int8)
        active=step>=refractory_until
        v[active]+=cfg.dt_ms/cfg.tau_membrane_ms*(-(v[active]-cfg.v_rest)+syn[active]+teacher_pulse[active])
        v[~active]=cfg.v_reset
        syn*=math.exp(-cfg.dt_ms/cfg.tau_synaptic_current_ms)
        firing=(step==read_step)&active&(v>cfg.threshold)
        if teacher is not None and step==teacher_step:
            firing=firing|(active&(v>cfg.threshold)&(teacher!=0))
        if np.any(firing):
            for j in np.flatnonzero(firing):
                kind='prediction' if step<teacher_step else ('teacher' if teacher is not None and step==teacher_step and teacher[j] else 'extra')
                events[j].append(t);kinds[j].append(kind)
                if kind=='teacher':teacher_emitted[j]=1
            v[firing]=cfg.v_reset
            refractory_until[firing]=step+refractory_steps
        if record_trace:
            trace_v[step]=v
            trace_i[step]=syn+teacher_pulse
    prediction=np.asarray([any(k=='prediction' for k in kind) for kind in kinds],dtype=np.int8)
    return {'prediction':prediction,'event_times':events,'event_kinds':kinds,
            'teacher_emitted':teacher_emitted,'teacher_suppressed':suppressed,
            'input_current':injected_current,'voltage_trace':trace_v,'current_trace':trace_i}
