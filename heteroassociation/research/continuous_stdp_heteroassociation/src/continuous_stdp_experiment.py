"""Hardware-oriented continuous LIF + continuous all-to-all STDP history.

Retains LIF membrane/refractory and STDP event history across association boundaries,
while applying accumulated STDP once after each association. Only normal spike reset
is performed; no artificial trial or epoch reset. 1000 epochs, Original/Modified K10.

Source datasets, order, and reference metrics are bundled in 20261102_105031.zip /
nonnegative_weight_results.zip.  Existing fixed LIF/STDP constants retained.

Local integration [-0.1, 10.0] ms contains 101 intervals; next association
starts at 15.1 ms (10.1ms association processing + 5ms guard interval).
All spike pair weights with time distance < 50ms computed exactly by event
history, pairs >=50ms contribute zero in float64 for this Gaussian kernel.
"""
from __future__ import annotations
import argparse, csv, io, json, math, time, zipfile
from pathlib import Path
import numpy as np
from numba import njit

SOURCE_DIR = Path(__file__).resolve().parent
DIR = SOURCE_DIR
REF = SOURCE_DIR / 'reference_20261102_105031.zip'
PREVIOUS = SOURCE_DIR / 'reference_nonnegative_weight_results.zip'
S = 1.286893592
C = .170690168
LAMBDA = 1.0
DT = .1
TAU = 2.
VTH = 1.
CAP = 1.282590743781507
TEACHER_I = 30.
TRIAL_TICKS = 101
GAP_TICKS = 50     # 5ms extra idle gap, 15.1ms teacher-to-teacher
WINDOW_HISTORY_TICKS = 500  # 50ms: outside this kernel underflows to zero in float64
NEPOCH = 1000

@njit

def lif_trial(x, w, y, volt, refr_until, base_tick, teacher_on):
    """Mutate continuous neuron state; return actual post-event local tick array."""
    pred = np.zeros(15, dtype=np.int64)
    counts = np.zeros(15, dtype=np.int64)
    post_ticks = np.full((15, 12), -1, dtype=np.int64)
    eff = np.zeros(15)
    for j in range(15):
        current=0.
        for i in range(15):
            current+=x[i]*w[i,j]
        eff[j]=min(current,CAP)
    for k in range(TRIAL_TICKS):
        # global start and end for interval k
        start=base_tick+k-1
        for j in range(15):
            active=start>=refr_until[j]
            if active:
                cs = eff[j] if 11 <= k <=40 else 0.
                teach = TEACHER_I if (teacher_on and k==0 and y[j]==1) else 0.
                volt[j]+=DT/TAU*(-volt[j]+cs+teach)
            else:
                volt[j]=0.
            if active and volt[j]>VTH:
                assert counts[j] < 12
                post_ticks[j, counts[j]] = k
                counts[j] += 1
                if k>0: pred[j]=1
                volt[j]=0.
                refr_until[j]=base_tick+k+10 # 1 ms refractory
    return pred, counts, post_ticks

@njit

def zero_current_gap(volt, refr, start_tick, ticks):
    for k in range(ticks):
        t = start_tick+k
        for j in range(15):
            if t>=refr[j]:
                volt[j]*=(1.-DT/TAU)
            else:
                volt[j]=0.

def window(dt):
    u = dt/S
    return C*(1.-u*u)*math.exp(-.5*u*u)

class ContinuousAllPairs:
    """Global spike history. Pre and post event against all preceding opposite events.

    Keeps history of last 50ms; F(>=50ms)=0 in double precision. Pairs
    crossing trial boundaries are included, also if pre/post roles reversed.
    """
    def __init__(self):
        self.pre = [[] for _ in range(15)]
        self.post = [[] for _ in range(15)]
        self.cross_count=0
        self.cross_abs=0.
        self.cross_signed=0.
        self.cross_max=0.
        self.total_count=0
    def prune(self, now):
        cut = now-WINDOW_HISTORY_TICKS
        for b in self.pre+self.post:
            b[:] = [(t,trial) for t,trial in b if t>cut]
    def process(self, x, counts, post_ticks, base_tick, trial_number):
        # Chronological events: teacher at 0, pre at 1, natural post >=1.
        # At a simultaneous pre/post timestamp, pre is processed first.
        events=[]
        for j in range(15):
            for k in range(counts[j]):
                events.append((int(base_tick+post_ticks[j,k]), 1, j))
        for i in range(15):
            if x[i]: events.append((base_tick+10, 0, i))
        events.sort(key=lambda z:(z[0], z[1]))
        delta=np.zeros((15,15))
        self.prune(base_tick)
        for tick,kind,idx in events:
            if kind==0:
                for j in range(15):
                    for pt, earlier_trial in self.post[j]:
                        dw=LAMBDA*window((pt-tick)*DT)
                        delta[idx,j]+=dw
                        self.total_count+=1
                        if earlier_trial!=trial_number:
                            self.cross_count+=1
                            self.cross_abs+=abs(dw)
                            self.cross_signed+=dw
                            self.cross_max=max(self.cross_max,abs(dw))
                self.pre[idx].append((tick,trial_number))
            else:
                for i in range(15):
                    for pt, earlier_trial in self.pre[i]:
                        dw=LAMBDA*window((tick-pt)*DT)
                        delta[i,idx]+=dw
                        self.total_count+=1
                        if earlier_trial!=trial_number:
                            self.cross_count+=1
                            self.cross_abs+=abs(dw)
                            self.cross_signed+=dw
                            self.cross_max=max(self.cross_max,abs(dw))
                self.post[idx].append((tick,trial_number))
        return delta

def load_data(name):
    with zipfile.ZipFile(REF) as z:
        with np.load(io.BytesIO(z.read(f'20261102_105031/npz/{name.lower()}_K10_seed0_Basic_Correction.npz'))) as d:
            X=d['input'].astype(np.int64); Y=d['target'].astype(np.int64)
        records=list(csv.DictReader(io.StringIO(z.read('20261102_105031/csv/presentation_orders.csv').decode())))
        r=next(r for r in records if r['dataset']==name and r['K']=='10' and r['seed']=='0')
        orders=np.array([[int(p) for p in seq.split(',')] for seq in r['orders'].split(';')],dtype=np.int64)
    assert X.shape==Y.shape==(10,15) and orders.shape==(1000,10)
    return X,Y,orders

def previous_history():
    with zipfile.ZipFile(PREVIOUS) as z:
        return list(csv.DictReader(io.StringIO(z.read('nonnegative_weight_experiment/epoch_history.csv').decode())))

def recall_sequential(X,Y,W,volt,refr,next_tick):
    """Counterfactual read-only probe on COPY of neuron state; no weight changes.
    Probe is not inserted into the training timeline or eligibility history.
    """
    vv=volt.copy(); rr=refr.copy()
    found=np.zeros_like(Y)
    for idx in range(10):
        tick=next_tick+idx*(TRIAL_TICKS+GAP_TICKS)
        pred,_,_=lif_trial(X[idx],W,Y[idx],vv,rr,tick,False)
        found[idx]=pred
        if GAP_TICKS: zero_current_gap(vv,rr,tick+TRIAL_TICKS,GAP_TICKS)
    return int(np.sum(np.all(found==Y,axis=1))), float(np.mean(found==Y))

def recall_independent(X,Y,W):
    """Standard historical probe: independent rest starts, used ONLY for comparison."""
    found=np.zeros_like(Y)
    for idx in range(10):
        v=np.zeros(15); r=np.full(15,-1000000000,dtype=np.int64)
        pred,_,_=lif_trial(X[idx],W,Y[idx],v,r,0,False)
        found[idx]=pred
    return int(np.sum(np.all(found==Y,axis=1)))

def train(name,floor):
    X,Y,orders=load_data(name)
    W=np.zeros((15,15)); v=np.zeros(15); refr=np.full(15,-1000000000,dtype=np.int64)
    stdp=ContinuousAllPairs()
    history=[]; t=time.time(); tick=0; trial_num=0; early_spike_count=0; other_spike_count=0
    for ep,order in enumerate(orders,1):
        for assoc in order:
            trial_num+=1
            pred,counts,post_ticks=lif_trial(X[assoc],W,Y[assoc],v,refr,tick,True)
            for j in range(15):
                # All Teacher positive bits should spike at time zero; if not, report.
                if Y[assoc,j] and post_ticks[j,0]!=0:
                    raise AssertionError(f'Teacher did not fire at 0: {name},{ep},{assoc},{j}')
                if not Y[assoc,j] and counts[j]>0 and post_ticks[j,0]==0:
                    raise AssertionError('No-target teacher at zero')
                for k in range(counts[j]):
                    et=post_ticks[j,k]
                    if et not in (0,40):
                        other_spike_count+=1
                        if et<40: early_spike_count+=1
            delta=stdp.process(X[assoc],counts,post_ticks,tick,trial_num)
            W +=delta
            if floor: np.maximum(W,0,out=W)
            if GAP_TICKS:
                zero_current_gap(v,refr,tick+TRIAL_TICKS,GAP_TICKS)
            tick+=TRIAL_TICKS+GAP_TICKS
        seq, bit=recall_sequential(X,Y,W,v,refr,tick)
        indep=recall_independent(X,Y,W)
        history.append(dict(dataset=name,model='W >= 0' if floor else 'Baseline',epoch=ep,
                            exact_count_continuous_recall=seq,exact_count_independent_recall=indep,
                            bit_accuracy_continuous_recall=bit))
        if ep%250==0:
            print(f'{name} {"W>=0" if floor else "Baseline"}: epoch{ep} continuous={seq}/10 isolated={indep}/10 elapsed {time.time()-t:.1f}s',flush=True)
    # Use continuous recall for comparison and present independent recall as diagnostic.
    for key in ['exact_count_continuous_recall','exact_count_independent_recall']:
        full=[e['epoch'] for e in history if e[key]==10]
        first=min(full) if full else None
        stable=next((e['epoch'] for e in reversed(history) if e[key]!=10),0)+1 if history[-1][key]==10 else None
        print(name, 'W>=0' if floor else 'Baseline',key, 'first',first, 'stable',stable, 'last',history[-1][key],flush=True)
    def first_for(k): return next((e['epoch'] for e in history if e[k]==10),None)
    def stable_for(k):
        if history[-1][k]!=10: return None
        bad=next((e['epoch'] for e in reversed(history) if e[k]!=10),0)
        return bad+1
    summary=dict(dataset=name,model='W >= 0' if floor else 'Baseline',
                 first_10_continuous=first_for('exact_count_continuous_recall'),
                 stable_start_continuous=stable_for('exact_count_continuous_recall'),
                 final_continuous=history[-1]['exact_count_continuous_recall'],
                 first_10_independent=first_for('exact_count_independent_recall'),
                 stable_start_independent=stable_for('exact_count_independent_recall'),
                 final_independent=history[-1]['exact_count_independent_recall'],
                 non_4ms_natural_spike_count=other_spike_count,early_natural_spike_count=early_spike_count,
                 cross_association_pairs=stdp.cross_count,
                 cross_association_sum_absolute_dw=stdp.cross_abs,
                 cross_association_sum_signed_dw=stdp.cross_signed,
                 cross_association_max_abs_single_dw=stdp.cross_max,
                 final_min_weight=float(W.min()),final_max_weight=float(W.max()),
                 simulated_time_ms=tick*DT,seconds=time.time()-t)
    return history,summary,W

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def main():
    assert GAP_TICKS >= 0
    assert abs(LAMBDA*window(-1)-.05)<1e-7
    assert abs(LAMBDA*window(3)+.05)<1e-7
    # Validation of single-trial LIF firing schedule on representative positive weights.
    X,Y,_=load_data('Original')
    w=np.full((15,15),CAP/X[0].sum())
    v=np.zeros(15);r=np.full(15,-1000000,dtype=np.int64)
    pred,cnts,times=lif_trial(X[0],w,Y[0],v,r,0,True)
    assert all(times[j,0]==0 if Y[0,j]==1 else times[j,0]==40 for j in range(15))
    assert all(cnts[j]==(2 if Y[0,j] else 1) for j in range(15))
    print('Single-trial baseline timing validated')
    allhist=[];summary=[];outputs={}
    for name in ('Original','Modified'):
        for floor in (False,True):
            h,s,W=train(name,floor)
            allhist.extend(h);summary.append(s)
            outputs[name+('_floor' if floor else '_baseline')]=W
    write_csv(DIR/'new_epoch_history.csv',allhist)
    write_csv(DIR/'new_summary.csv',summary)
    np.savez_compressed(DIR/'final_weights.npz',**outputs)
    oldrows=previous_history()
    previous_map={(r['dataset'],'W >= 0' if r['condition']=='Lower bound W>=0' else 'Baseline',int(r['epoch'])):int(r['exact_count'])
                  for r in oldrows if r['condition'] in ('Baseline','Lower bound W>=0')}
    comparisons=[]
    for h in allhist:
        prior=previous_map[h['dataset'],h['model'],int(h['epoch'])]
        comparisons.append(dict(**h,previous_exact_count=prior,
            delta_continuous=h['exact_count_continuous_recall']-prior,
            delta_independent=h['exact_count_independent_recall']-prior))
    write_csv(DIR/'epoch_comparison.csv',comparisons)
    print('SUMMARY',json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    for name in ('Original','Modified'):
        for model in ('Baseline','W >= 0'):
            diff=[c for c in comparisons if c['dataset']==name and c['model']==model]
            print('COMPARISON',name,model,'continuous mismatch epochs',sum(e['delta_continuous']!=0 for e in diff),'independent mismatch epochs',sum(e['delta_independent']!=0 for e in diff),flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap-ms', type=float, default=5.,
                        help='Extra no-input idle gap after the 10.1ms presentation; default 5ms')
    parser.add_argument('--out-dir', type=Path, default=SOURCE_DIR,
                        help='Directory for this run CSV and weights')
    args=parser.parse_args()
    if args.gap_ms < 0 or abs(args.gap_ms/DT-round(args.gap_ms/DT)) > 1e-7:
        parser.error('Gap must be nonnegative and a multiple of 0.1 ms.')
    GAP_TICKS=int(round(args.gap_ms/DT))
    DIR=args.out_dir.resolve()
    DIR.mkdir(parents=True, exist_ok=True)
    main()

