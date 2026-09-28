"""100% Gradient Descent v1: τ와 w를 번갈아가며 최적화 (alternating update).

메커니즘
--------
입력(digit, CS) →[τ,w]→ 출력 coincidence 봉우리 → 타깃(letter, HS) 비교
τ와 w를 autograd로 번갈아가며 업데이트 (각 에폭에서: τ먼저 → w그다음).

모델
----
멤브레인: v(t) = Σ_i w_i * K(t - tau_i),  K = exp(-(t-tau_i)/OUT_TAU) if t>=tau_i else 0
판정: score = max_t v(t),  발화 <=> score >= theta

손실: BCE with logits

실행: python "템퍼럴_stdp/GD_v1.py"
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

RESULTS = HERE / "results_gd_v1"
RESULTS.mkdir(parents=True, exist_ok=True)
PER_OUT = RESULTS / "per_output"
PER_OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(0)
torch.manual_seed(0)

# ===== 설정 =====
N_PAIRS = 5
T_MAX = 10.0
T_CENTER = 5.0
INIT_SPREAD = 3.0
DT = 0.05
OUT_TAU = 0.4
GAIN = 4.0
# 학습
EPOCHS = 500
LR_TAU = 0.01      # τ 학습률
LR_W = 0.01        # w 학습률
EARLY_STOP = True  # 10/10 달성하면 조기 종료 (True/False)
# ================

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(N_PAIRS, 15).astype(np.float64)
Y01 = letter_images.reshape(N_PAIRS, 15).astype(np.float64)

n_steps = int(round(T_MAX / DT))
t_grid = np.arange(n_steps) * DT


def build_kernel(tau_t):
    """tau_t: [15] torch tensor. 반환 K: [n_steps, 15]"""
    tg = torch.tensor(t_grid, dtype=torch.float32).view(-1, 1)  # [T,1]
    d = tg - tau_t.view(1, -1)  # [T,15]
    K = torch.where(d >= 0, torch.exp(-d / OUT_TAU), torch.zeros_like(d))
    return K


def learn_gd_alternating():
    """τ, w를 번갈아가며 gradient descent."""
    tgt_all = torch.tensor(Y01, dtype=torch.float32)  # [D, 15]
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)  # [D,15]

    # 초기값
    tau = torch.tensor(
        np.clip(T_CENTER + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD, 0, T_MAX),
        dtype=torch.float32,
        requires_grad=True
    )
    w = (torch.randn(15) * 0.1).requires_grad_(True)
    theta = torch.zeros(15, requires_grad=True)  # [15] 각 출력별

    opt_tau = torch.optim.Adam([tau], lr=LR_TAU)
    opt_w = torch.optim.Adam([w, theta], lr=LR_W)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_hist = []

    for epoch in range(EPOCHS):
        # 1) τ 업데이트 (w, theta 고정)
        opt_tau.zero_grad()
        K = build_kernel(torch.clamp(tau, 0, T_MAX))  # [T,15]
        wm = w.view(1, -1) * masks  # [D,15]
        membrane = wm @ K.t()  # [D,T]
        score = membrane.max(dim=1).values  # [D]
        logit = GAIN * (score - theta.view(1, -1))
        loss_tau = loss_fn(logit, tgt_all)
        loss_tau.backward()
        opt_tau.step()

        # τ 범위 클립
        with torch.no_grad():
            tau.clamp_(0, T_MAX)

        # 2) w 업데이트 (τ 고정)
        opt_w.zero_grad()
        K = build_kernel(torch.clamp(tau, 0, T_MAX).detach())  # [T,15] 고정
        wm = w.view(1, -1) * masks  # [D,15]
        membrane = wm @ K.t()  # [D,T]
        score = membrane.max(dim=1).values  # [D]
        logit = GAIN * (score - theta.view(1, -1))
        loss_w = loss_fn(logit, tgt_all)
        loss_w.backward()
        opt_w.step()

        avg_loss = float((loss_tau + loss_w) / 2)
        loss_hist.append(avg_loss)

        # 매 에폭 accuracy 체크
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                K_eval = build_kernel(torch.clamp(tau, 0, T_MAX))
                wm = w.view(1, -1) * masks
                score = (wm @ K_eval.t()).max(dim=1).values
                pred = (score.view(-1, 1) >= theta.view(1, -1)).int().numpy()
                n_exact = sum(np.array_equal(pred[k], Y01[k].astype(int)) for k in range(N_PAIRS))
            print(f" epoch {epoch:4d}: loss {avg_loss:.4f}, exact {n_exact}/{N_PAIRS}")

        # 모든 출력이 정확하면 스킵
        if EARLY_STOP and (epoch + 1) % 50 == 0:
            with torch.no_grad():
                K_eval = build_kernel(torch.clamp(tau, 0, T_MAX))
                wm = w.view(1, -1) * masks
                score = (wm @ K_eval.t()).max(dim=1).values
                pred = (score.view(-1, 1) >= theta.view(1, -1)).int().numpy()
                n_exact = sum(np.array_equal(pred[k], Y01[k].astype(int)) for k in range(N_PAIRS))
            if n_exact == N_PAIRS:
                print(f" >> {N_PAIRS}/{N_PAIRS} 달성! epoch {epoch+1}에서 종료")
                break

    return tau.detach().numpy(), w.detach().numpy(), theta.detach().numpy(), np.array(loss_hist)


def membrane_trace(tau, w, active, theta):
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


def predict_all(tau, w, theta_all):
    """전체 예측."""
    pred_all = np.zeros((N_PAIRS, 15), dtype=int)
    for i in range(N_PAIRS):
        for j in range(15):
            A = np.where(X01[i] > 0.5)[0]
            tr = membrane_trace(tau, w, A, theta_all[j])
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
    fig.suptitle(f"Recall (GD v1 - alternating)  exact {n_exact}/{N_PAIRS}", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "gd_v1_recall_grid.png", dpi=120)


def plot_per_pixel_acc(per_pixel, n_exact, bit):
    fig, ax = plt.subplots(figsize=(4.5, 5.2))
    grid = per_pixel.reshape(5, 3)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=N_PAIRS)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{grid[r, c]}/{N_PAIRS}", ha="center", va="center", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Per-pixel acc (GD v1 - alternating)\nexact {n_exact}/{N_PAIRS}, bit {bit:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / "gd_v1_per_pixel_acc.png", dpi=120)


def main():
    print("=== GD v1: Alternating update (τ → w → τ → w ...) ===\n")
    tau, w, theta_all, loss_hist = learn_gd_alternating()

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

    print(f"\n그림: {RESULTS}")
    plt.show()


if __name__ == "__main__":
    main()
