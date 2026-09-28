"""두 구역(1-구역 / 0-구역) 대칭 클러스터링 temporal 모델.

아이디어 (요청안)
----------------
시간 창에 두 앵커: 1-구역 중심 MU_ONE(이른), 0-구역 중심 MU_ZERO(늦은).
각 입력 픽셀은 학습되는 발화 시각 tau_i.

학습 규칙(패턴 p, 활성 집합 A_p, 타깃 t_p):
  - t_p=1 : 활성 스파이크를 1-구역(MU_ONE)으로 모음  -> 0-구역에선 자연히 분산
  - t_p=0 : 반대로 0-구역(MU_ZERO)으로 모음          -> 1-구역에서 분산
  = tau[A] += ETA * (mu_tgt - tau[A])   (목표 구역으로 당김 = 거기 모이고 반대는 멀어짐)

판정: 활성 스파이크 평균 시각이 어느 구역에 가까운가.
  mean(tau[A]) < 중간점 -> 1-구역(이른) -> 1,  아니면 -> 0.

interval(한 구역만)과 달리 양쪽 다 목표 구역이 있어 대칭 -> majority 붕괴 완화.
15 출력 픽셀 전부 학습 + recall 그리드.

실행: python "템퍼럴_stdp/temporal_tworegion.py"
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

RESULTS = HERE / "results_tworegion"
RESULTS.mkdir(parents=True, exist_ok=True)
np.random.seed(0)

# ===== 설정 =====
N_PAIRS = 10
OUT_NEURON = 7          # 대표 출력(스파이크 타이밍 그림용). ALL_OUTPUTS 면 recall 은 전체.
ALL_OUTPUTS = True
T_MAX = 10.0
MU_ONE = 3.0            # 1-구역 중심 (이른)
MU_ZERO = 7.0           # 0-구역 중심 (늦은)
MID = (MU_ONE + MU_ZERO) / 2.0
INIT_SPREAD = 3.0
ETA = 0.1
EPOCHS = 200
# 판정용 coincidence 뉴런 (봉우리 위치로 구역 결정)
DT_SIM = 0.05
OUT_TAU = 0.5
# ================

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]
Y01 = letter_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]
ACTIVE = [np.where(X01[i] > 0.5)[0] for i in range(N_PAIRS)]


def _cluster_spread(tau, grp, direction):
    """direction=+1: 그룹 평균으로 모음(cluster).  -1: 평균에서 멀어짐(spread)."""
    if len(grp) < 2:
        return
    mean = tau[grp].mean()
    tau[grp] += direction * ETA * (mean - tau[grp])


def learn_tau(tgt):
    """구역별로 나눠서 업데이트.
       타깃1: 1-구역(tau<MID) 스파이크는 모으고, 0-구역(tau>=MID) 스파이크는 분산.
       타깃0: 반대 (0-구역 모으고, 1-구역 분산)."""
    tau = np.clip(MID + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD, 0, T_MAX)
    tau_hist = np.zeros((EPOCHS + 1, 15)); tau_hist[0] = tau
    for ep in range(EPOCHS):
        for p in np.random.permutation(N_PAIRS):
            A = ACTIVE[p]
            if len(A) == 0:
                continue
            A1 = A[tau[A] < MID]      # 현재 1-구역에 있는 활성 스파이크
            A0 = A[tau[A] >= MID]     # 현재 0-구역
            if tgt[p] == 1:
                _cluster_spread(tau, A1, +1)   # 1-구역 모으고
                _cluster_spread(tau, A0, -1)   # 0-구역 분산
            else:
                _cluster_spread(tau, A0, +1)   # 0-구역 모으고
                _cluster_spread(tau, A1, -1)   # 1-구역 분산
            tau = np.clip(tau, 0, T_MAX)
        tau_hist[ep + 1] = tau
    return tau, tau_hist


def coincidence_peak(tau_active):
    """활성 스파이크의 coincidence 막전위 봉우리 '시각'. 뭉친 구역에서 봉우리."""
    if len(tau_active) == 0:
        return MID
    n = int(round(T_MAX / DT_SIM))
    decay = np.exp(-DT_SIM / OUT_TAU)
    I = np.zeros(n)
    for t in tau_active:
        s = int(round(t / DT_SIM))
        if 0 <= s < n:
            I[s] += 1.0
    v, best_v, best_t = 0.0, -1.0, MID
    for step in range(n):
        v = v * decay + I[step]
        if v > best_v:
            best_v, best_t = v, step * DT_SIM
    return best_t


def predict(tau):
    """coincidence 봉우리가 어느 구역인가. 봉우리 시각 < MID(이른=1구역) -> 1."""
    preds, peaks = [], []
    for i in range(N_PAIRS):
        pt = coincidence_peak(tau[ACTIVE[i]])
        peaks.append(pt)
        preds.append(1 if pt < MID else 0)
    return np.array(preds), np.array(peaks)


# ---------------------------------------------------------------------------
def plot_spike_timing(tau, tgt, pred, means, out_neuron):
    """페어별 활성 스파이크 시각 + 두 구역 앵커 (화면표시)."""
    ncol = min(5, N_PAIRS); nrow = int(np.ceil(N_PAIRS / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for i in range(N_PAIRS):
        ax = flat[i]; A = ACTIVE[i]
        ax.scatter(tau[A], A, s=24, color=("tab:red" if tgt[i] == 1 else "tab:blue"), zorder=3)
        ax.axvline(MU_ONE, color="tab:red", ls=":", alpha=0.5)     # 1-구역
        ax.axvline(MU_ZERO, color="tab:blue", ls=":", alpha=0.5)   # 0-구역
        ax.axvline(MID, color="k", ls="--", lw=0.8)                # 판정 경계
        ax.scatter([means[i]], [15.5], s=60, marker="v",
                   color=("tab:red" if pred[i] == 1 else "tab:gray"), edgecolors="k", zorder=4)
        ok = "O" if pred[i] == tgt[i] else "X"
        ax.set_title(f"d{i} tgt={tgt[i]} pred={pred[i]} {ok}", fontsize=8,
                     color=("green" if pred[i] == tgt[i] else "red"))
        ax.set_xlim(0, T_MAX); ax.set_ylim(-1, 16); ax.set_yticks([0, 5, 10, 14])
    for k in range(N_PAIRS, len(flat)):
        flat[k].axis("off")
    fig.suptitle(f"out {out_neuron}: spike timing  (red=1-region@{MU_ONE}, "
                 f"blue=0-region@{MU_ZERO}, v=coincidence peak)", fontsize=11)
    fig.supxlabel("spike time"); fig.supylabel("input pixel")
    fig.tight_layout()
    fig.savefig(RESULTS / f"tworegion_n{out_neuron}_timing.png", dpi=120)


def plot_recall_grid(pred_all):
    fig, axes = plt.subplots(3, N_PAIRS, figsize=(1.45 * N_PAIRS, 4.6), squeeze=False)
    for k in range(N_PAIRS):
        inp = X01[k].reshape(5, 3); tg = Y01[k].reshape(5, 3); pr = pred_all[k].reshape(5, 3)
        axes[0][k].imshow(inp, cmap="gray_r", vmin=0, vmax=1)
        axes[1][k].imshow(tg, cmap="gray_r", vmin=0, vmax=1)
        axes[2][k].imshow(pr, cmap="gray_r", vmin=0, vmax=1)
        exact = np.array_equal(pr, tg); bit = float((pr == tg).mean())
        axes[0][k].set_title(f"{k}->{chr(65 + k)}", fontsize=9, color=("green" if exact else "red"))
        axes[2][k].set_xlabel(f"{bit:.2f}", fontsize=8, color=("green" if exact else "red"))
        for r in range(3):
            axes[r][k].set_xticks([]); axes[r][k].set_yticks([])
    axes[0][0].set_ylabel("input"); axes[1][0].set_ylabel("target"); axes[2][0].set_ylabel("pred")
    n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(N_PAIRS))
    fig.suptitle(f"Recall (two-region)  exact {n_exact}/{N_PAIRS}", fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / "tworegion_recall_grid.png", dpi=120)


def main():
    if not ALL_OUTPUTS:
        tgt = Y01[:, OUT_NEURON].astype(int)
        tau, _ = learn_tau(tgt)
        pred, means = predict(tau)
        print(f"out {OUT_NEURON}: acc {int((pred==tgt).sum())}/{N_PAIRS}")
        plot_spike_timing(tau, tgt, pred, means, OUT_NEURON)
        plt.show()
        return

    pred_all = np.zeros((N_PAIRS, 15), dtype=int)
    per_pixel = np.zeros(15, dtype=int)
    rep = None
    print("=== two-region, 출력 15픽셀 ===")
    for j in range(15):
        tgt = Y01[:, j].astype(int)
        tau, _ = learn_tau(tgt)
        pred, means = predict(tau)
        pred_all[:, j] = pred
        per_pixel[j] = int((pred == tgt).sum())
        if j == OUT_NEURON:
            rep = (tau.copy(), tgt.copy(), pred.copy(), means.copy())
        print(f" out {j:2d}: {per_pixel[j]}/{N_PAIRS}")

    n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(N_PAIRS))
    bit = float((pred_all == Y01.astype(int)).mean())
    print(f"\n전체: exact letters={n_exact}/{N_PAIRS}, bit_acc={bit:.3f}")

    plot_recall_grid(pred_all)
    # per-pixel acc
    fig, ax = plt.subplots(figsize=(4.5, 5.2))
    grid = per_pixel.reshape(5, 3)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=N_PAIRS)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{grid[r,c]}/{N_PAIRS}", ha="center", va="center", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Per-pixel acc (two-region)\nexact {n_exact}/{N_PAIRS}, bit {bit:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(RESULTS / "tworegion_per_pixel_acc.png", dpi=120)

    if rep is not None:
        plot_spike_timing(*rep, OUT_NEURON)
    print(f"그림: {RESULTS}")
    plt.show()


if __name__ == "__main__":
    main()
