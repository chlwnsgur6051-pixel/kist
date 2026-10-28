"""Teacher-free Clocked LIF recall with dynamic association count K."""
from __future__ import annotations

import math
import numpy as np


def recall_batch(x, w, cfg, timing):
    """Replicate the validated K-dynamic Euler recall, threshold > 0.01."""
    K=len(x);v=np.zeros((K,15));syn=np.zeros((K,15))
    until=np.zeros((K,15),dtype=np.int32)
    dt=cfg.dt_ms;pre=int(round(timing['pre']/dt))
    read=int(round(timing['prediction']/dt))
    steps=int(round((timing['teacher']+2.)/dt))+1
    injected=cfg.input_gain*(x@w)
    pred=np.zeros((K,15),dtype=np.int8)
    voltage=np.empty_like(v)
    for step in range(steps):
        if step==pre:syn+=injected
        active=step>=until
        v[active]+=dt/cfg.tau_membrane_ms*(-v[active]+syn[active])
        v[~active]=cfg.v_reset
        syn*=math.exp(-dt/cfg.tau_synaptic_current_ms)
        firing=active&(v>cfg.threshold)&(step==read)
        if step==read:voltage[:]=v
        pred[firing]=1
        v[firing]=cfg.v_reset
        until[firing]=step+int(round(cfg.refractory_ms/dt))
    assert w.shape==(15,15) and pred.shape==(K,15)
    return pred,voltage
