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

RESULTS = HERE / "results_stdp"
RESULTS.mkdir(parents=True, exist_ok=True)
np.random.seed(0)

# ===== 설정 =====
N_PAIRS = 10           # 사용할 페어 수 (앞에서부터 N개 숫자)
OUT_NEURON = 7         # 추론할 letter 픽셀 (7=XNOR, 5/8 균형, 0/12 자명)
# 2-phase 인코딩
T_MAX = 10.0
T_ON = 2.0            # 픽셀=1 -> 이 시각에 스파이크
T_OFF = 7.0          # 픽셀=0 -> 이 시각에 스파이크
DT = 0.1
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
# 교사(T2)
T_TEACH = 4.5        # 타깃1 일 때 US 강제 발화 시각 (T_ON 과 T_OFF 사이)
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
STEP_ON = int(round(T_ON / DT))
STEP_OFF = int(round(T_OFF / DT))
STEP_TEACH = int(round(T_TEACH / DT))
DECAY = np.exp(-DT / TAU_M)


def encode(bits, teach=None):
    """
    B2 symmetric timing

    target=1 / recall:
        bit 1 -> 2 ms
        bit 0 -> 7 ms

    target=0 training:
        bit 0 -> 2 ms
        bit 1 -> 7 ms
    """

    if teach == "timing_ltd":
        return np.where(
            bits > 0.5,
            STEP_OFF,   # active 1 -> late -> LTD
            STEP_ON     # inactive 0 -> early -> LTP
        ).astype(int)

    return np.where(
        bits > 0.5,
        STEP_ON,
        STEP_OFF
    ).astype(int)


def simulate(w, bits, teach=None):
    """LIF 시뮬. teach: None(recall) / 'fire'(타깃1 US) / 'shunt'(타깃0 침묵).
    반환: post_spike_steps(list), spike_steps(픽셀별 pre 스텝), v_trace."""
    spike_steps = encode(bits, teach=teach)
    v = 0.0
    post, trace = [], np.zeros(n_steps)
    for step in range(n_steps):
        I = w[spike_steps == step].sum()           # 이 스텝에 도착한 입력들의 가중합
        v = v * DECAY + I
        if teach in ("fire", "timing_ltd") and step == STEP_TEACH:
            v = THRESHOLD + 1.0
        if v >= THRESHOLD:
            post.append(step)
            v = V_RESET
        trace[step] = v
    return post, spike_steps, trace


def stdp_update(w, spike_steps, post):
    """pre(픽셀별 1스파이크) x post 쌍에 STDP 적용."""
    for i in range(15):
        t_pre = spike_steps[i] * DT
        for ps in post:
            dt = ps * DT - t_pre
            if dt > 0:
                w[i] += A_PLUS * np.exp(-dt / TAU_STDP)
            elif dt < 0:
                w[i] -= A_MINUS * np.exp(dt / TAU_STDP)
    return np.clip(w, W_MIN, W_MAX)


def predict(w):
    """교사 없이 CS 만 -> 발화하면 1."""
    preds, traces = [], []
    for i in range(N_PAIRS):
        post, _, tr = simulate(w, X01[i], teach=None)
        preds.append(1 if len(post) > 0 else 0)
        traces.append(tr)
    return np.array(preds), traces

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
            teach = "fire" if tgt[p] == 1 else "timing_ltd"

            post, ss, _ = simulate(w, X01[p], teach=teach)

            trial_count += 1
            y = int(tgt[p])

            natural_posts = [
                ps for ps in post
                if ps != STEP_TEACH
            ]

            natural_spike_count += len(natural_posts)

            diag[y]["trials"] += 1
            diag[y]["natural"] += len(natural_posts)

            if len(post) > 1:
                multi_post_trial_count += 1
                diag[y]["multi"] += 1

            w = stdp_update(w, ss, post)
        
#    print("\n=== Training spike diagnostic ===")
#    print(f"total trials       : {trial_count}")
#    print(f"natural post spikes: {natural_spike_count}")
#    print(f"multi-post trials  : {multi_post_trial_count}")

#    for y in [0, 1]:
#        print(
#            f"target={y}: "
#            f"trials={diag[y]['trials']}, "
#            f"natural={diag[y]['natural']}, "
#            f"multi={diag[y]['multi']}"
#        )
    return w, tgt, acc_hist, w_hist

def recall_peak(w, bits):
    spike_steps = encode(bits, teach=None)

    v = 0.0
    peak_v = -np.inf

    for step in range(n_steps):
        I = w[spike_steps == step].sum()
        v = v * DECAY + I

        if v > peak_v:
            peak_v = v

    return peak_v

def main():
    w, tgt, acc_hist, w_hist = train()
    pred, traces = predict(w)
    acc = int((pred == tgt).sum())
    t_axis = np.arange(n_steps) * DT

    print(f"=== STDP temporal (out {OUT_NEURON}, N_PAIRS={N_PAIRS}) ===")
    print(f"타깃 = {tgt.tolist()}")
    for i in range(N_PAIRS):
        post, _, _ = simulate(w, X01[i], teach=None)
        print(f" digit {i}: post_spikes={len(post)}  pred={pred[i]}  target={tgt[i]}  "
              f"{'O' if pred[i]==tgt[i] else 'X'}")
    print(f"정확도: {acc}/{N_PAIRS}")

    ncol = min(5, N_PAIRS); nrow = int(np.ceil(N_PAIRS / ncol))

    # 1) 스파이크 래스터 (입력 2-phase + 출력)
    fig1, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow),
                              sharex=True, squeeze=False)
    flat = axes.ravel()
    for i in range(N_PAIRS):
        axp = flat[i]
        ss = encode(X01[i])
        on = np.where(X01[i] > 0.5)[0]; off = np.where(X01[i] <= 0.5)[0]
        axp.scatter(np.full_like(on, T_ON), on, s=18, color="tab:blue")     # 1 -> t_ON
        axp.scatter(np.full_like(off, T_OFF), off, s=18, color="tab:red")    # 0 -> t_OFF
        post, _, _ = simulate(w, X01[i], teach=None)
        for ps in post:
            axp.axvline(ps * DT, color="green", lw=1.2, alpha=0.6)
        ok = "O" if pred[i] == tgt[i] else "X"
        axp.set_title(f"d{i} tgt={tgt[i]} pred={pred[i]} {ok}", fontsize=8,
                      color=("green" if pred[i] == tgt[i] else "red"))
        axp.set_xlim(0, T_MAX); axp.set_ylim(-1, 15)
    for k in range(N_PAIRS, len(flat)):
        flat[k].axis("off")
    fig1.suptitle("Spike raster: input(blue=1@t2, red=0@t7) + output(green)", fontsize=12)
    fig1.supxlabel("time"); fig1.supylabel("input pixel / output")
    fig1.tight_layout(); fig1.savefig(RESULTS / f"stdp_n{OUT_NEURON}_1_raster.png", dpi=120)

    # 2) 출력 막전위 (페어별 한 평면)
    fig2, ax = plt.subplots(figsize=(9, 5))
    for i in range(N_PAIRS):
        ax.plot(t_axis, traces[i], lw=1.2,
                color=("tab:red" if tgt[i] == 1 else "tab:blue"), alpha=0.8, label=f"{i}->{tgt[i]}")
    ax.axhline(THRESHOLD, color="k", ls=":", lw=1, label="threshold")
    ax.axvline(T_ON, color="tab:blue", ls="--", alpha=0.3); ax.axvline(T_OFF, color="tab:red", ls="--", alpha=0.3)
    ax.set_xlabel("time"); ax.set_ylabel("membrane v")
    ax.set_title(f"Output membrane per digit (recall)  red=tgt1")
    ax.legend(fontsize=7, ncol=2)
    fig2.tight_layout(); fig2.savefig(RESULTS / f"stdp_n{OUT_NEURON}_2_membrane.png", dpi=120)

    # 3) 가중치 5x3 + 에폭별 변화
    fig3, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.5))
    W = w.reshape(5, 3); m = np.abs(W).max() or 1e-9
    im = a0.imshow(W, cmap="RdBu_r", vmin=-m, vmax=m)
    for r in range(5):
        for c in range(3):
            a0.text(c, r, f"{W[r,c]:.2f}", ha="center", va="center", fontsize=8)
    a0.set_xticks([]); a0.set_yticks([]); a0.set_title("weights (5x3)")
    fig3.colorbar(im, ax=a0, fraction=0.046)
    for j in range(15):
        a1.plot(w_hist[:, j], lw=1.0, color=plt.cm.viridis(j / 14))
    a1.axhline(0, color="k", lw=0.6); a1.set_xlabel("epoch"); a1.set_ylabel("w"); a1.set_title("weights over epochs")
    fig3.tight_layout(); fig3.savefig(RESULTS / f"stdp_n{OUT_NEURON}_3_weights.png", dpi=120)

    # 4) 결과 요약 + 학습곡선
    fig4, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.6))
    xp = np.arange(N_PAIRS)
    a0.bar(xp - 0.2, tgt, 0.4, label="target", color="tab:gray")
    a0.bar(xp + 0.2, pred, 0.4, label="pred", color="tab:orange")
    for i in range(N_PAIRS):
        a0.text(i, 1.05, "O" if pred[i] == tgt[i] else "X", ha="center",
                color=("green" if pred[i] == tgt[i] else "red"), fontweight="bold")
    a0.set_xticks(xp); a0.set_ylim(0, 1.25); a0.set_title(f"result {acc}/{N_PAIRS}"); a0.legend(fontsize=8)
    a1.plot(acc_hist, color="tab:green"); a1.set_ylim(0, N_PAIRS + 0.5)
    a1.set_xlabel("epoch"); a1.set_ylabel("acc"); a1.set_title("learning curve")
    fig4.tight_layout(); fig4.savefig(RESULTS / f"stdp_n{OUT_NEURON}_4_summary.png", dpi=120)

    print(f"그림: {RESULTS}")
    plt.show()
    
    print("final weights =", w)
    
    print("\n=== Recall membrane peaks ===")
    for p in range(N_PAIRS):
        peak = recall_peak(w, X01[p])
        print(
            f"digit {p}: "
            f"peak={peak:.4f}  "
            f"target={tgt[p]}"
        )
        
    

if __name__ == "__main__":
    main()
