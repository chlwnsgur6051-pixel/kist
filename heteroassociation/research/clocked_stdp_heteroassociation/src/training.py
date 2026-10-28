"""Sequential clocked-LIF learning; weights change only through spike pairs."""
import numpy as np
from .model import LIFConfig, simulate_trial
from .stdp import event_pair_update
from .recall import recall_batch

def config_lif(config):
    return LIFConfig(**config['lif'])

def train(x,y,config,epochs=None,order=None):
    k=len(x)
    assert x.shape==y.shape==(k,15)
    cfg=config_lif(config)
    timing=config['timing_ms']
    window=config['window']
    epochs=config['epochs'] if epochs is None else epochs
    order=list(range(k)) if order is None else list(order)
    assert sorted(order)==list(range(k))
    w=np.zeros((15,15),dtype=np.float64)
    weights=np.empty((epochs+1,15,15),dtype=np.float64)
    exact=np.empty(epochs+1)
    bit=np.empty(epochs+1)
    correctness=np.empty((epochs+1,k),dtype=np.int8)
    weights[0]=w
    pred,_=recall_batch(x,w,cfg,timing)
    exact[0]=np.mean(np.all(pred==y,axis=1))
    bit[0]=np.mean(pred==y)
    correctness[0]=np.all(pred==y,axis=1)
    best_epoch=0
    first_100=None
    teacher_due=teacher_emitted=0
    gross_ltp=gross_ltd=0.0
    pair_count=0
    for epoch in range(1,epochs+1):
        for p in order:
            trial=simulate_trial(x[p],w,cfg,timing,teacher=y[p])
            delta,audit=event_pair_update(x[p],trial,timing,window)
            assert not np.any(delta[x[p]==0])
            assert not any(trial['teacher_emitted'][y[p]==0])
            w+=delta
            teacher_due+=int(np.sum(y[p]))
            teacher_emitted+=int(np.sum(trial['teacher_emitted']))
            gross_ltp+=audit['gross_ltp']
            gross_ltd+=audit['gross_ltd']
            pair_count+=audit['pair_count']
        weights[epoch]=w
        pred,_=recall_batch(x,w,cfg,timing)
        exact[epoch]=np.mean(np.all(pred==y,axis=1))
        bit[epoch]=np.mean(pred==y)
        correctness[epoch]=np.all(pred==y,axis=1)
        if exact[epoch]>exact[best_epoch]+1e-14: best_epoch=epoch
        if exact[epoch]==1 and first_100 is None:first_100=epoch
    return dict(weight_by_epoch=weights,exact_by_epoch=exact,bit_by_epoch=bit,
        association_correct_by_epoch=correctness,final_prediction=pred,
        best_epoch=best_epoch,first_100_epoch=first_100,
        teacher_due=teacher_due,teacher_emitted=teacher_emitted,
        gross_ltp=gross_ltp,gross_ltd=gross_ltd,pair_count=pair_count)
