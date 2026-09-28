"""τ(스파이크 시각) + w(멤브레인 영향력 가중치) 결합 모델.

동기: interval(τ=발화시각, spread 판정)은 '공유 τ 충돌 + 중요도 가중 없음'으로 막힘.
      -> 각 입력 스파이크가 출력 멤브레인에 주는 영향력을 가중치 w 로 두고,
         w 를 surrogate gradient(autograd)로 최적화한다. (충돌 픽셀은 w 가 자동 downweight)

모델 (출력 픽셀 j):
  - 입력 픽셀 i(활성)가 시각 tau_i 에 스파이크
  - 출력 멤브레인 v(t) = Σ_i w_i * K(t - tau_i),  K = 짧은 tau 지수 PSP (coincidence)
  - 점수 = max_t v(t),  발화 <=> 점수 >= theta
  - readout(surrogate) = sigmoid(gain*(점수 - theta))

학습:
  1) tau : interval 클러스터링 규칙(타깃1 모으기 / 타깃0 벌리기) — 정렬(delay) 담당
  2) w   : autograd + Adam 으로 BCE 최소화 — 중요도 담당 (봉우리에 기여하는 픽셀만 강조)

실행: python "템퍼럴_stdp/temporal_weighted.py"
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

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
PER_OUT = RESULTS / "per_output_weighted"
PER_OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(0)
torch.manual_seed(0)

# ===== 설정 =====
N_PAIRS = 10
T_MAX = 10.0
T_CENTER = 5.0
INIT_SPREAD = 3.0
DT = 0.05
OUT_TAU = 0.4            # 짧을수록 coincidence 엄격
GAIN = 4.0              # sigmoid 기울기
# tau 클러스터링(정렬)
ETA_CLUSTER = 0.05
ETA_SPREAD = 0.05
EPOCHS_TAU = 200
# w 최적화(중요도, autograd)
EPOCHS_W = 10000
LR_W = 0.0005
# ================

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]
Y01 = letter_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]
ACTIVE = [np.where(X01[i] > 0.5)[0] for i in range(N_PAIRS)]

n_steps = int(round(T_MAX / DT))
t_grid = np.arange(n_steps) * DT


# ---------------------------------------------------------------------------
# 1) tau 클러스터링 (interval 규칙) — 정렬 담당
# ---------------------------------------------------------------------------
def learn_tau(tgt):
    tau = np.clip(T_CENTER + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD, 0, T_MAX)
    for _ in range(EPOCHS_TAU):
        for p in np.random.permutation(N_PAIRS):
            A = ACTIVE[p]
            if len(A) == 0:
                continue
            dev = tau[A] - tau[A].mean()
            eta = ETA_CLUSTER if tgt[p] == 1 else ETA_SPREAD
            if tgt[p] == 1:
                tau[A] -= eta * dev
            else:
                tau[A] += eta * dev
            tau = np.clip(tau, 0, T_MAX)
    return tau


# ---------------------------------------------------------------------------
# 2) w 최적화 (autograd) — 중요도 담당
# ---------------------------------------------------------------------------
def build_kernel(tau):
    """Kbase[t, i] = exp(-(t - tau_i)/OUT_TAU) for t>=tau_i else 0.  torch [T,15]."""
    tg = torch.tensor(t_grid, dtype=torch.float32).view(-1, 1)     # [T,1]
    tau_t = torch.tensor(tau, dtype=torch.float32).view(1, -1)     # [1,15]
    d = tg - tau_t
    K = torch.where(d >= 0, torch.exp(-d / OUT_TAU), torch.zeros_like(d))
    return K                                                       # [T,15]


def learn_w(tau, tgt):
    K = build_kernel(tau)                                          # [T,15]
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)          # [D,15]
    target = torch.tensor(tgt, dtype=torch.float32)               # [D]
    # w=0 초기화 금지: 멤브레인 평탄 -> max 미분 퇴화 -> w 안 배움.
    w = (torch.randn(15) * 0.1).requires_grad_(True)
    theta = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, theta], lr=LR_W)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_hist = []
    for _ in range(EPOCHS_W):
        opt.zero_grad()
        # membrane[d,t] = Σ_i (w_i * mask[d,i]) * K[t,i]
        wm = w.view(1, -1) * masks                                # [D,15]
        membrane = wm @ K.t()                                      # [D,T]
        score = membrane.max(dim=1).values                        # [D]
        logit = GAIN * (score - theta)
        loss = loss_fn(logit, target)
        loss.backward()
        opt.step()
        loss_hist.append(float(loss))
    with torch.no_grad():
        wm = w.view(1, -1) * masks
        score = (wm @ K.t()).max(dim=1).values
        pred = (score >= theta).int().numpy()
    return w.detach().numpy(), float(theta.detach()), pred, np.array(loss_hist), K.numpy()


# ---------------------------------------------------------------------------
def membrane_trace(tau, w, active):
    """시각화용: 활성 픽셀 가중 멤브레인 v(t)."""
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


def plot_output(j, tau, w, theta, tgt, pred, loss_hist):
    # 멤브레인(페어별) + 가중치 + 손실 3-패널
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(N_PAIRS):
        tr = membrane_trace(tau, w, ACTIVE[i])
        axes[0].plot(t_grid, tr, lw=1.0,
                     color=("tab:red" if tgt[i] == 1 else "tab:blue"), alpha=0.7)
    axes[0].axhline(theta, color="k", ls=":", lw=1, label=f"theta={theta:.2f}")
    axes[0].set_title(f"out {j}: weighted membrane (red=tgt1)"); axes[0].legend(fontsize=8)
    axes[0].set_xlabel("time"); axes[0].set_ylabel("v")
    W = w.reshape(5, 3); m = np.abs(W).max() or 1e-9
    im = axes[1].imshow(W, cmap="RdBu_r", vmin=-m, vmax=m)
    for r in range(5):
        for c in range(3):
            axes[1].text(c, r, f"{W[r,c]:.1f}", ha="center", va="center", fontsize=7)
    axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].set_title("weights w (5x3)")
    axes[2].plot(loss_hist, color="tab:red"); axes[2].set_title("BCE loss"); axes[2].set_xlabel("epoch")
    acc = int((pred == tgt).sum())
    fig.suptitle(f"out {j}  acc {acc}/{N_PAIRS}", fontsize=12)
    fig.tight_layout(); fig.savefig(PER_OUT / f"weighted_n{j:02d}.png", dpi=110)
    plt.close(fig)


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
    fig.suptitle(f"Recall (tau+w)  exact {n_exact}/{N_PAIRS}", fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / "weighted_recall_grid.png", dpi=120)


def main():
    pred_all = np.zeros((N_PAIRS, 15), dtype=int)
    per_pixel = np.zeros(15, dtype=int)
    print("=== tau + w (surrogate) 결합 모델, 출력 15픽셀 ===")
    for j in range(15):
        tgt = Y01[:, j].astype(int)
        tau = learn_tau(tgt)                        # 1) 정렬
        w, theta, pred, loss_hist, _ = learn_w(tau, tgt)   # 2) 중요도(autograd)
        pred_all[:, j] = pred
        per_pixel[j] = int((pred == tgt).sum())
        plot_output(j, tau, w, theta, tgt, pred, loss_hist)
        print(f" out {j:2d}: {per_pixel[j]}/{N_PAIRS}")

    n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(N_PAIRS))
    bit = float((pred_all == Y01.astype(int)).mean())
    print(f"\n전체: exact letters={n_exact}/{N_PAIRS}, bit_acc={bit:.3f}")

    plot_recall_grid(pred_all)
    # per-pixel acc 히트맵
    fig, ax = plt.subplots(figsize=(4.5, 5.2))
    grid = per_pixel.reshape(5, 3)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=N_PAIRS)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{grid[r,c]}/{N_PAIRS}", ha="center", va="center", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Per-pixel acc (tau+w)\nexact {n_exact}/{N_PAIRS}, bit {bit:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(RESULTS / "weighted_per_pixel_acc.png", dpi=120)

    print(f"그림: {RESULTS}  (출력별: {PER_OUT})")
    plt.show()


if __name__ == "__main__":
    main()
