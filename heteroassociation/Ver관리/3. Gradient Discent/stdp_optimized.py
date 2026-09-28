"""비교군: Closed-form Hebbian으로 τ와 w를 최적화.

메커니즘
--------
- τ (spike timing): interval rule (cluster/spread)로 학습
- w (weight): closed-form Hebbian (one-shot 계산)
- 생물학적 규칙, error correction 아님

모델
----
멤브레인: v(t) = Σ_i w_i * K(t - tau_i),  K = exp(-(t-tau_i)/OUT_TAU) if t>=tau_i else 0
판정: score = max_t v(t),  발화 <=> score >= theta

학습 규칙
---------
τ (interval, 반복 학습):
  tgt=1: tau[A] -= eta_cluster * dev  (모으기)
  tgt=0: tau[A] += eta_spread * dev   (벌리기)

w (closed-form Hebbian, one-shot):
  w = Σ_patterns (2*tgt - 1) * (2*x - 1)  (전체 패턴의 상관)
  → 한 번 계산으로 최적값 도달

실행: python "템퍼럴_stdp/stdp_optimized.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import binary_character as data

RESULTS = HERE / "results_stdp_optimized"
RESULTS.mkdir(parents=True, exist_ok=True)
PER_OUT = RESULTS / "per_output"
PER_OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(0)
torch.manual_seed(0)

# ===== 설정 =====
N_PAIRS = 10
T_MAX = 10.0
T_CENTER = 5.0
INIT_SPREAD = 3.0
DT = 0.05
OUT_TAU = 0.4
GAIN = 4.0
# τ 학습 (interval rule)
ETA_CLUSTER = 0.01
ETA_SPREAD = 0.01
EPOCHS_TAU = 500
# w: closed-form (one-shot, epoch 무관)
# ================

n_steps = int(round(T_MAX / DT))
t_grid = np.arange(n_steps) * DT

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]
Y01 = letter_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]


def build_kernel(tau):
    """tau: [15] numpy. 반환 K: [n_steps, 15]"""
    tg = t_grid.reshape(-1, 1)  # [T,1]
    d = tg - tau.reshape(1, -1)  # [T,15]
    K = np.where(d >= 0, np.exp(-d / OUT_TAU), np.zeros_like(d))
    return K


def membrane_trace(tau, w, active):
    """활성 픽셀의 멤브레인 v(t)."""
    decay = np.exp(-DT / OUT_TAU)
    I = np.zeros(n_steps)
    for i in active:
        s = int(round(tau[i] / DT))
        if 0 <= s < n_steps:
            I[s] += w[i]
    v, tr = 0.0, np.zeros(n_steps)
    for step in range(n_steps):
        v = v * decay + I[step]
        tr[step] = v
    return tr


def learn_hebbian_closed_form():
    """Closed-form Hebbian: w를 한 번에 계산. τ는 interval rule로 학습."""
    tgt_all = Y01.astype(int)  # [D, 15]

    # 1) w: closed-form Hebbian (one-shot)
    x_bi = 2 * X01 - 1  # [D, 15] bipolar
    w = np.zeros(15)
    for j in range(15):
        tgt_bi = 2 * tgt_all[:, j] - 1  # [D] bipolar
        w += np.sum(tgt_bi[:, None] * x_bi, axis=0)  # [15]
    w = w / N_PAIRS  # 정규화

    # 2) τ: interval rule (반복 학습)
    tau = np.clip(T_CENTER + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD, 0, T_MAX)
    acc_hist = []

    for epoch in range(EPOCHS_TAU):
        for p in np.random.permutation(N_PAIRS):
            for j in range(15):
                A = np.where(X01[p] > 0.5)[0]
                if len(A) == 0:
                    continue
                tgt = tgt_all[p, j]
                dev = tau[A] - tau[A].mean()
                eta = ETA_CLUSTER if tgt == 1 else ETA_SPREAD
                if tgt == 1:
                    tau[A] -= eta * dev
                else:
                    tau[A] += eta * dev
                tau = np.clip(tau, 0, T_MAX)

        # theta 업데이트: 각 출력별 최적 threshold
        theta_all = np.zeros(15)
        for j in range(15):
            scores = []
            for i in range(N_PAIRS):
                A = np.where(X01[i] > 0.5)[0]
                tr = membrane_trace(tau, w, A)
                scores.append(tr.max())
            scores = np.array(scores)
            pos_scores = scores[tgt_all[:, j] == 1]
            neg_scores = scores[tgt_all[:, j] == 0]
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                theta_all[j] = (pos_scores.mean() + neg_scores.mean()) / 2
            elif len(pos_scores) > 0:
                theta_all[j] = pos_scores.mean()

        # 정확도 계산
        pred_all = np.zeros((N_PAIRS, 15), dtype=int)
        for i in range(N_PAIRS):
            for j in range(15):
                A = np.where(X01[i] > 0.5)[0]
                tr = membrane_trace(tau, w, A)
                pred_all[i, j] = 1 if tr.max() >= theta_all[j] else 0

        n_exact = sum(np.array_equal(pred_all[k], tgt_all[k]) for k in range(N_PAIRS))
        acc_hist.append(n_exact)

        if (epoch + 1) % 50 == 0:
            bit = float((pred_all == tgt_all).mean())
            print(f" tau-epoch {epoch:4d}: exact {n_exact}/{N_PAIRS}, bit {bit:.3f}")

    return tau, w, theta_all, np.array(acc_hist)


def predict_all(tau, w, theta_all):
    """전체 예측."""
    pred_all = np.zeros((N_PAIRS, 15), dtype=int)
    for i in range(N_PAIRS):
        for j in range(15):
            A = np.where(X01[i] > 0.5)[0]
            tr = membrane_trace(tau, w, A)
            pred_all[i, j] = 1 if tr.max() >= theta_all[j] else 0
    return pred_all


def plot_recall_grid(pred_all):
    fig, axes = plt.subplots(3, N_PAIRS, figsize=(1.45 * N_PAIRS, 4.6), squeeze=False)
    for k in range(N_PAIRS):
        inp = X01[k].reshape(5, 3)
        tg = Y01[k].astype(int).reshape(5, 3)
        pr = pred_all[k].reshape(5, 3)
        axes[0][k].imshow(inp, cmap="gray_r", vmin=0, vmax=1)
        axes[1][k].imshow(tg, cmap="gray_r", vmin=0, vmax=1)
        axes[2][k].imshow(pr, cmap="gray_r", vmin=0, vmax=1)
        exact = np.array_equal(pr, tg)
        bit = float((pr == tg).mean())
        axes[0][k].set_title(f"{k}->{chr(65 + k)}", fontsize=9, color=("green" if exact else "red"))
        axes[2][k].set_xlabel(f"{bit:.2f}", fontsize=8, color=("green" if exact else "red"))
        for r in range(3):
            axes[r][k].set_xticks([])
            axes[r][k].set_yticks([])
    axes[0][0].set_ylabel("input")
    axes[1][0].set_ylabel("target")
    axes[2][0].set_ylabel("pred")
    n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(N_PAIRS))
    fig.suptitle(f"Recall (Hebbian)  exact {n_exact}/{N_PAIRS}", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "hebbian_recall_grid.png", dpi=120)


def plot_per_pixel_acc(per_pixel, n_exact, bit):
    fig, ax = plt.subplots(figsize=(4.5, 5.2))
    grid = per_pixel.reshape(5, 3)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=N_PAIRS)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{grid[r, c]}/{N_PAIRS}", ha="center", va="center", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Per-pixel acc (Hebbian)\nexact {n_exact}/{N_PAIRS}, bit {bit:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / "hebbian_per_pixel_acc.png", dpi=120)


def main():
    print("=== Hebbian (τ interval + w closed-form) ===")
    print("w: closed-form Hebbian (one-shot 계산)")
    print(f"τ: interval rule ({EPOCHS_TAU} epochs)\n")
    tau, w, theta_all, acc_hist = learn_hebbian_closed_form()

    pred_all = predict_all(tau, w, theta_all)
    per_pixel = np.zeros(15, dtype=int)

    for j in range(15):
        per_pixel[j] = int((pred_all[:, j] == Y01[:, j].astype(int)).sum())
        print(f" out {j:2d}: {per_pixel[j]}/{N_PAIRS}")

    n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(N_PAIRS))
    bit = float((pred_all == Y01.astype(int)).mean())
    print(f"\n전체: exact letters={n_exact}/{N_PAIRS}, bit_acc={bit:.3f}")

    plot_recall_grid(pred_all)
    plot_per_pixel_acc(per_pixel, n_exact, bit)

    # 학습 곡선
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(acc_hist, lw=2, color="tab:green")
    ax.set_xlabel("tau epoch")
    ax.set_ylabel("exact matches")
    ax.set_title(f"Learning curve (Hebbian: w closed-form + τ interval)")
    ax.set_ylim([0, N_PAIRS + 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "hebbian_learning_curve.png", dpi=120)

    print(f"\n그림: {RESULTS}")
    plt.show()


if __name__ == "__main__":
    main()
