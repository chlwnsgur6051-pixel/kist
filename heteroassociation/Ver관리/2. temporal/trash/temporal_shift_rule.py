"""(CS,US) 조합으로 출력 스파이크를 앞/뒤로 이동시키는 학습 규칙.

설계 (요청안)
------------
각 시냅스 i 에 대해 (CS_i = 입력 픽셀, US = 타깃):
    Δw_i = eta * (2*US - 1) * (2*CS_i - 1)     # bipolar Hebbian 증분
이 규칙이 '이 패턴의 출력 스파이크'를 이동:
    CS,US = 11, 01  ->  앞쪽(이른 발화, "1")
    CS,US = 00, 10  ->  뒤쪽(늦은 발화, "0")
즉 방향은 US 가 결정하고, bipolar 입력이라 CS=0 픽셀도 기여한다.

출력은 latency 코딩(이르면 1, 늦으면 0). 반복 적용하면 bipolar 상관행렬
연관기억으로 수렴한다. 여기서는 '스파이크가 에폭마다 앞/뒤로 움직이는' 과정을 시각화.

실행: python "템퍼럴_stdp/temporal_shift_rule.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import binary_character as data

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
EVO = RESULTS / "evolution"          # 에폭별 스냅샷 프레임 저장
EVO.mkdir(parents=True, exist_ok=True)
np.random.seed(0)

# ===== 설정 =====
OUT_NEURON = 7          # letter 픽셀 (0/12 자명, 5/8 균형, 7=XNOR)
T_MAX = 10.0
T_CENTER = 5.0
THRESHOLD = 1.0
RAMP = THRESHOLD / T_CENTER
GAIN = 0.15             # D_norm -> latency 이득 (< RAMP)
ETA = 0.01              # 학습률 (스파이크 이동 속도)
EPOCHS = 300
EVO_EVERY = 5           # 몇 에폭마다 스냅샷 프레임 저장할지
# 입력 스파이크 타이밍 표현(2-phase): 비트=1 -> 이른, 비트=0 -> 늦은
T_A_IN = 1.0
T_B_IN = 3.0
# ================

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(10, 15).astype(np.float64)      # 0/1 입력
Y01 = letter_images.reshape(10, 15).astype(np.float64)     # 0/1 타깃
Xb = 2 * X01 - 1                                            # bipolar 입력 (CS)
digit_names = [str(i) for i in range(10)]

# 정규화 스케일(가시화 안정용): 학습 후 |D| 규모로 고정하지 않고 러닝 max 사용
D_SCALE = 15.0 * 0.5    # 대략적 스케일 (Σ x_b^2=15, lr 반영). latency 매핑용.


def drive_to_latency(D):
    Dn = D / D_SCALE
    return np.clip(THRESHOLD / np.maximum(RAMP + GAIN * Dn, 1e-3), 0, T_MAX)


def simulate_ramp(Dn):
    n = int(round(T_MAX / (T_MAX / 200)))
    dt = T_MAX / n
    v, trace, lat = 0.0, [], None
    for step in range(n):
        v += (RAMP + GAIN * Dn) * dt
        if lat is None and v >= THRESHOLD:
            lat = step * dt; trace.append(THRESHOLD)
        else:
            trace.append(min(v, THRESHOLD * 1.1))
    return np.array(trace), (T_MAX if lat is None else lat)


def plot_timing_frame(w, tgt, out_neuron, epoch):
    """현재 가중치로 '입력 타이밍(2-phase) + 출력 스파이크(latency)' 프레임을 EVO 에 저장."""
    D = Xb @ w
    lat = drive_to_latency(D)
    pred = (lat < T_CENTER).astype(int)
    acc = int((pred == tgt).sum())
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), sharex=True, sharey=True)
    for i, axp in enumerate(axes.ravel()):
        for nrn in range(15):
            if X01[i, nrn] > 0.5:
                axp.scatter(T_A_IN, nrn, s=22, color="tab:blue", zorder=3)
            else:
                axp.scatter(T_B_IN, nrn, s=22, color="tab:red", zorder=3)
        axp.scatter(lat[i], 17, s=80, marker="o",
                    color=("tab:red" if pred[i] == 1 else "tab:gray"), edgecolors="k", zorder=4)
        axp.axhline(15.5, color="lightgray", lw=0.8)
        axp.axvline(T_A_IN, color="tab:blue", ls=":", alpha=0.4)
        axp.axvline(T_B_IN, color="tab:red", ls=":", alpha=0.4)
        axp.axvline(T_CENTER, color="k", ls="--", lw=0.8, alpha=0.7)
        ok = "O" if pred[i] == tgt[i] else "X"
        axp.set_title(f"digit {i} tgt={tgt[i]} pred={pred[i]} {ok}", fontsize=9,
                      color=("green" if pred[i] == tgt[i] else "red"))
        axp.set_xlim(0, T_MAX); axp.set_ylim(-1, 19)
        axp.set_yticks([0, 5, 10, 14, 17]); axp.set_yticklabels(["0", "5", "10", "14", "OUT"], fontsize=7)
    fig.suptitle(f"epoch {epoch:04d}   acc {acc}/10   (out {out_neuron})  "
                 "input timing(2-phase) + output latency", fontsize=12)
    fig.supxlabel("time"); fig.supylabel("neuron (input 0-14, OUT)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(EVO / f"shift_n{out_neuron}_ep{epoch:04d}.png", dpi=110)
    plt.close(fig)


def train(out_neuron):
    tgt = Y01[:, out_neuron]                                # 0/1
    tgt_i = tgt.astype(int)
    us_b = 2 * tgt - 1                                      # bipolar US, [10]
    w = np.zeros(15)
    lat_hist = np.zeros((EPOCHS + 1, 10))                   # 에폭별 각 숫자 latency
    w_hist = np.zeros((EPOCHS + 1, 15))                     # 에폭별 15 가중치

    lat_hist[0] = drive_to_latency(Xb @ w)
    w_hist[0] = w
    plot_timing_frame(w, tgt_i, out_neuron, 0)             # 에폭 0 스냅샷
    for ep in range(EPOCHS):
        for i in np.random.permutation(10):
            # Δw_i = eta * (2US-1) * (2CS_i-1)  = eta * us_b[i] * Xb[i]
            w += ETA * us_b[i] * Xb[i]
        lat_hist[ep + 1] = drive_to_latency(Xb @ w)
        w_hist[ep + 1] = w
        if (ep + 1) % EVO_EVERY == 0:                       # 5에폭마다 스냅샷
            plot_timing_frame(w, tgt_i, out_neuron, ep + 1)
    return w, tgt_i, lat_hist, w_hist


# ---------------------------------------------------------------------------
def main():
    w, tgt, lat_hist, w_hist = train(OUT_NEURON)
    D = Xb @ w
    lat = drive_to_latency(D)
    pred = (lat < T_CENTER).astype(int)
    acc = int((pred == tgt).sum())
    Dn = D / D_SCALE

    print(f"=== (CS,US) shift rule, 출력 뉴런 {OUT_NEURON} ===")
    print(f"타깃 = {tgt.tolist()}")
    for i in range(10):
        print(f" digit {i}: D={D[i]:+.3f}  latency={lat[i]:.2f}  "
              f"pred={pred[i]}  target={tgt[i]}  {'O' if pred[i]==tgt[i] else 'X'}")
    print(f"정확도: {acc}/10")

    # 1) 페어별 입력 타이밍(2-phase) + 출력(latency), 입력 마커 크기=|가중치|
    wabs_max = float(np.abs(w).max()) or 1e-9
    fig1, axes = plt.subplots(2, 5, figsize=(15, 6.4), sharex=True, sharey=True)
    for i, axp in enumerate(axes.ravel()):
        for nrn in range(15):
            sz = 8 + 70 * (abs(w[nrn]) / wabs_max)      # 마커 크기 = 가중치 크기
            if X01[i, nrn] > 0.5:
                axp.scatter(T_A_IN, nrn, s=sz, color="tab:blue", zorder=3)
            else:
                axp.scatter(T_B_IN, nrn, s=sz, color="tab:red", zorder=3)
        axp.scatter(lat[i], 17, s=70, marker="o",
                    color=("tab:red" if pred[i] == 1 else "tab:gray"), edgecolors="k", zorder=4)
        axp.axhline(15.5, color="lightgray", lw=0.8)
        axp.axvline(T_CENTER, color="k", ls="--", lw=0.8, alpha=0.7)
        ok = "O" if pred[i] == tgt[i] else "X"
        axp.set_title(f"digit {i} tgt={tgt[i]} pred={pred[i]} {ok}", fontsize=9,
                      color=("green" if pred[i] == tgt[i] else "red"))
        axp.set_xlim(0, T_MAX); axp.set_ylim(-1, 19)
        axp.set_yticks([0, 5, 10, 14, 17]); axp.set_yticklabels(["0", "5", "10", "14", "OUT"], fontsize=7)
    fig1.suptitle("Final input timing (2-phase) + output latency  "
                  "[marker size = |weight|]", fontsize=12)
    fig1.supxlabel("time"); fig1.supylabel("neuron (input 0-14, OUT)")
    fig1.tight_layout(); fig1.savefig(RESULTS / f"shift_n{OUT_NEURON}_1_timing.png", dpi=120)

    # 2) per-digit 바: 출력 latency (interval fig2 spread-bar 대응)
    fig2, ax = plt.subplots(figsize=(9, 3.6))
    colors = ["tab:red" if tgt[i] == 1 else "tab:blue" for i in range(10)]
    ax.bar(np.arange(10), lat, color=colors, alpha=0.85)
    ax.axhline(T_CENTER, color="k", ls="--", lw=1, label=f"decision t={T_CENTER}")
    for i in range(10):
        ok = pred[i] == tgt[i]
        ax.text(i, lat[i] + 0.12, "O" if ok else "X", ha="center",
                color=("green" if ok else "red"), fontweight="bold")
    ax.set_xticks(np.arange(10)); ax.set_xticklabels(digit_names)
    ax.set_xlabel("digit"); ax.set_ylabel("output latency"); ax.set_ylim(0, T_MAX)
    ax.set_title(f"Output latency per digit (out {OUT_NEURON}, acc {acc}/10)  red=tgt1 blue=tgt0")
    ax.legend(fontsize=9)
    fig2.tight_layout(); fig2.savefig(RESULTS / f"shift_n{OUT_NEURON}_2_latency.png", dpi=120)

    # 3) 15개 가중치의 에폭별 변화 — 한 평면 (interval fig3 tau_time 대응)
    fig3, ax = plt.subplots(figsize=(9, 5))
    eps = np.arange(EPOCHS + 1)
    colors2 = plt.cm.viridis(np.linspace(0, 1, 15))
    for j in range(15):
        ax.plot(eps, w_hist[:, j], lw=1.2, color=colors2[j], label=f"w{j}")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("epoch"); ax.set_ylabel("weight value")
    ax.set_title(f"15 weights over epochs (out {OUT_NEURON})  (>0 흥분 / <0 억제)")
    ax.legend(fontsize=6, ncol=5, loc="upper left")
    fig3.tight_layout(); fig3.savefig(RESULTS / f"shift_n{OUT_NEURON}_3_weights_time.png", dpi=120)

    # 4) 페어별 출력 멤브레인 (ramp) — interval fig4 대응
    n_ramp = 200
    t_axis = np.arange(n_ramp) * (T_MAX / n_ramp)
    fig4, axes = plt.subplots(2, 5, figsize=(15, 6.4), sharex=True, sharey=True)
    for i, axp in enumerate(axes.ravel()):
        tr, _ = simulate_ramp(Dn[i])
        axp.plot(t_axis[:len(tr)], tr, lw=1.5,
                 color=("tab:red" if tgt[i] == 1 else "tab:blue"))
        axp.axhline(THRESHOLD, color="k", ls=":", lw=1)
        axp.axvline(T_CENTER, color="gray", ls="--", lw=0.8)
        ok = "O" if pred[i] == tgt[i] else "X"
        axp.set_title(f"digit {i} tgt={tgt[i]} pred={pred[i]} {ok}", fontsize=9,
                      color=("green" if pred[i] == tgt[i] else "red"))
        axp.set_xlim(0, T_MAX); axp.set_ylim(0, THRESHOLD * 1.15)
    fig4.suptitle(f"Output membrane (ramp) per pair (out {OUT_NEURON})  "
                  "steeper -> earlier crossing = '1'", fontsize=12)
    fig4.supxlabel("time"); fig4.supylabel("output membrane v")
    fig4.tight_layout(); fig4.savefig(RESULTS / f"shift_n{OUT_NEURON}_4_out_membrane.png", dpi=120)

    print(f"그림: {RESULTS}")
    print(f"에폭별 스냅샷({EVO_EVERY}에폭마다): {EVO}")
    plt.show()


if __name__ == "__main__":
    main()
