"""STDP 연관기억 (temporal 2-phase 입력, 단일 출력 1비트).

스펙
----
- 입력: 숫자 15픽셀. 2-phase 인코딩 -> 픽셀=1 이면 t=T_ON(2) 에 스파이크,
        픽셀=0 이면 t=T_OFF(7) 에 스파이크. 창 [0, T_MAX(10)].
- 출력: 뉴런 1개 (letter 픽셀 OUT_NEURON 을 1비트로 추론).
- 학습: STDP (spike-timing, 국소 규칙). gradient 아님.
        교사(T2): 타깃=1 -> 출력 발화 유도(US, t_teach 에 강제 발화),
                  타깃=0 -> 침묵(shunt). recall 은 교사 없이 CS 만.
- 페어 수: N_PAIRS 로 설정.

STDP:  Δw_i = +A+ exp(-Δt/τ)  (Δt = t_post - t_pre > 0, 강화)
             -A- exp( Δt/τ)  (Δt < 0, 약화)
2-phase 라 타깃1 학습 시: ON 픽셀(t=2, post=3 이전)은 강화, OFF 픽셀(t=7, post 이후)은 약화.

실행: python "템퍼럴_stdp/stdp_temporal.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import binary_character as data

RESULTS = HERE / "results_stdp_temporal_fixed_input"
RESULTS.mkdir(parents=True, exist_ok=True)
np.random.seed(0)

# ===== 설정 =====
N_PAIRS = 10           # 사용할 페어 수 (앞에서부터 N개 숫자)
OUT_NEURON = 7         # 추론할 letter 픽셀 (7=XNOR, 5/8 균형, 0/12 자명)
# 2-phase 인코딩
T_MAX = 10.0
T_ONE = 3.0
T_ZERO = 2.0
DT = 0.1

T_TEACH_1 = 4.0
T_TEACH_0 = 1.0

STEP_ONE = int(round(T_ONE / DT))
STEP_ZERO = int(round(T_ZERO / DT))

STEP_TEACH_1 = int(round(T_TEACH_1 / DT))
STEP_TEACH_0 = int(round(T_TEACH_0 / DT))

# 출력 LIF 뉴런
TAU_M = 3.0          # 막전위 시상수
THRESHOLD = 1.0
V_RESET = 0.0
# STDP
A_PLUS = 0.02
A_MINUS = 0.02
TAU_STDP = 2.0
W_INIT = 0.05
W_MIN, W_MAX = -1.5, 1.5

EPOCHS = 60
# target=0 timing intervention (B1)
T_NEG_INACTIVE = 0.0   # input bit=0
T_NEG_ACTIVE = 4.0     # input bit=1

STEP_NEG_INACTIVE = int(round(T_NEG_INACTIVE / DT))
STEP_NEG_ACTIVE = int(round(T_NEG_ACTIVE / DT))
# ================

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]
Y01 = letter_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]

n_steps = int(round(T_MAX / DT))
#STEP_ON = int(round(T_ON / DT))
#STEP_OFF = int(round(T_OFF / DT))
#STEP_TEACH = int(round(T_TEACH / DT))
DECAY = np.exp(-DT / TAU_M)


def encode(bits):
    spike_steps = []

    for bit in bits:
        if bit == 1:
            spike_steps.append([STEP_ONE])
        else:
            spike_steps.append([STEP_ZERO])

    return spike_steps


def simulate(w, pre_spikes, teacher_step=None):
    v = 0.0
    post_spikes = []

    decay = np.exp(-DT / TAU_M)

    for step in range(int(T_MAX / DT) + 1):
        v *= decay

        # presynaptic input
        for i, spikes in enumerate(pre_spikes):
            if step in spikes:
                v += w[i]

        # natural postsynaptic spike
        if v >= THRESHOLD:
            post_spikes.append(step)
            v = V_RESET

        # teacher spike
        if teacher_step is not None and step == teacher_step:
            post_spikes.append(step)
            v = V_RESET

    return post_spikes


def stdp_update(w, spike_steps, post):
    """pre(픽셀별 1스파이크) x post 쌍에 STDP 적용."""
    for i in range(15):
        t_pre = spike_steps[i][0] * DT
        for ps in post:
            dt = ps * DT - t_pre
            if dt > 0:
                w[i] += A_PLUS * np.exp(-dt / TAU_STDP)
            elif dt < 0:
                w[i] -= A_MINUS * np.exp(dt / TAU_STDP)
    return np.clip(w, W_MIN, W_MAX)


def predict(w):
    pred = []

    for x in X01:
        pre_spikes = encode(x)

        post_spikes = simulate(
            w,
            pre_spikes,
            teacher_step=None
        )

        pred.append(1 if len(post_spikes) > 0 else 0)

    return np.array(pred, dtype=int)

def train():
    natural_spike_count = 0
    trial_count = 0
    multi_post_trial_count = 0
    diag = {
        0: {"trials": 0, "natural": 0, "multi": 0},
        1: {"trials": 0, "natural": 0, "multi": 0},
    }
    tgt = Y01[:, OUT_NEURON].astype(int)
    w = np.full(15, W_INIT)
    acc_hist, w_hist = [], np.zeros((EPOCHS + 1, 15))
    w_hist[0] = w
    for ep in range(EPOCHS):
        for p in np.random.permutation(N_PAIRS):
            x = X01[p]
            y = int(tgt[p])

            pre_spikes = encode(x)

            if y == 1:
                teacher_step = STEP_TEACH_1
            else:
                teacher_step = STEP_TEACH_0

            # simulation은 하되
            post_spikes = simulate(
                w,
                pre_spikes,
                teacher_step=teacher_step
            )

            # 이번 실험에서는 natural post를 STDP에 사용하지 않음.
            # teacher spike만 사용.
            w = stdp_update(
                w,
                pre_spikes,
                [teacher_step]
            )
            
        pred_ep = predict(w)
        acc_hist.append(int((pred_ep == tgt).sum()))
        w_hist[ep + 1] = w
        
    return w, tgt, acc_hist, w_hist

def main():
    w, tgt, acc_hist, w_hist = train()
    pred = predict(w)

    acc = int((pred == tgt).sum())

    print(f"=== Fixed-input STDP (out {OUT_NEURON}, N_PAIRS={N_PAIRS}) ===")
    print("timing:")
    print(f"  input 1  -> {T_ONE} ms")
    print(f"  input 0  -> {T_ZERO} ms")
    print(f"  target 1 -> {T_TEACH_1} ms")
    print(f"  target 0 -> {T_TEACH_0} ms")
    print()

    print("target :", tgt.tolist())
    print("pred   :", pred.tolist())

    for i in range(N_PAIRS):
        pre_spikes = encode(X01[i])
        post = simulate(w, pre_spikes, teacher_step=None)

        print(
            f"digit {i}: "
            f"post_steps={post}, "
            f"pred={pred[i]}, "
            f"target={tgt[i]}, "
            f"{'O' if pred[i] == tgt[i] else 'X'}"
        )

    print()
    print("correct:", acc, "/", N_PAIRS)
    print("weights:")
    print(w)
        
    

if __name__ == "__main__":
    main()
