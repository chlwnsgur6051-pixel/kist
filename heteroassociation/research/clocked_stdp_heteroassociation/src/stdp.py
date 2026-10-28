"""Validated centered inverted Mexican-hat spike-pair window."""
from __future__ import annotations

import math
import numpy as np


def window(delta_t_ms, parameters):
    u=delta_t_ms/parameters['width_ms']
    return parameters['k']*parameters['amplitude']*(u*u-1)*math.exp(-u*u/2)


def event_pair_update(x, trial, timing, parameters):
    """All-to-all actual pre/post pairs; no prediction-error mask is used."""
    delta=np.zeros((15,15),dtype=float)
    active=np.flatnonzero(x)
    gross_ltp=gross_ltd=0.0
    pair_count=0
    for j in range(15):
        for t,kind in zip(trial['event_times'][j],trial['event_kinds'][j]):
            assert kind in ('prediction','teacher')
            contribution=window(t-timing['pre'],parameters)
            delta[active,j]+=contribution
            pair_count+=len(active)
            if contribution>0:gross_ltp+=len(active)*contribution
            if contribution<0:gross_ltd+=len(active)*(-contribution)
    return delta,{'gross_ltp':gross_ltp,'gross_ltd':gross_ltd,'pair_count':pair_count}
