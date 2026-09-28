"""GD v1 (alternating) vs v2 (simultaneous) loss 비교.

v1: τ → w → τ → w ... (번갈아)
v2: (τ, w 동시) → (τ, w 동시) ... (동시)

실행: python "템퍼럴_stdp/compare_gd_versions.py"
"""
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

RESULTS = HERE / "results_gd_compare"
RESULTS.mkdir(parents=True, exist_ok=True)

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
EPOCHS = 500
LR_TAU = 0.01
LR_W = 0.01
# ================

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(N_PAIRS, 15).astype(np.float64)
Y01 = letter_images.reshape(N_PAIRS, 15).astype(np.float64)

n_steps = int(round(T_MAX / DT))
t_grid = np.arange(n_steps) * DT


def build_kernel(tau_t):
    tg = torch.tensor(t_grid, dtype=torch.float32).view(-1, 1)
    d = tg - tau_t.view(1, -1)
    K = torch.where(d >= 0, torch.exp(-d / OUT_TAU), torch.zeros_like(d))
    return K


def learn_v1_alternating():
    """v1: τ업데이트 → w업데이트 번갈아."""
    tgt_all = torch.tensor(Y01, dtype=torch.float32)
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)

    tau = torch.tensor(
        np.clip(T_CENTER + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD, 0, T_MAX),
        dtype=torch.float32,
        requires_grad=True
    )
    w = (torch.randn(15) * 0.1).requires_grad_(True)
    theta = torch.zeros(15, requires_grad=True)

    opt_tau = torch.optim.Adam([tau], lr=LR_TAU)
    opt_w = torch.optim.Adam([w, theta], lr=LR_W)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_hist = []

    for epoch in range(EPOCHS):
        # 1) τ 업데이트
        opt_tau.zero_grad()
        K = build_kernel(torch.clamp(tau, 0, T_MAX))
        wm = w.view(1, -1) * masks
        membrane = wm @ K.t()
        score = membrane.max(dim=1).values
        logit = GAIN * (score - theta.view(1, -1))
        loss_tau = loss_fn(logit, tgt_all)
        loss_tau.backward()
        opt_tau.step()

        with torch.no_grad():
            tau.clamp_(0, T_MAX)

        # 2) w 업데이트
        opt_w.zero_grad()
        K = build_kernel(torch.clamp(tau, 0, T_MAX).detach())
        wm = w.view(1, -1) * masks
        membrane = wm @ K.t()
        score = membrane.max(dim=1).values
        logit = GAIN * (score - theta.view(1, -1))
        loss_w = loss_fn(logit, tgt_all)
        loss_w.backward()
        opt_w.step()

        avg_loss = float((loss_tau + loss_w) / 2)
        loss_hist.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"v1 epoch {epoch+1:3d}: loss {avg_loss:.6f}")

    return np.array(loss_hist)


def learn_v2_simultaneous():
    """v2: τ, w 동시 업데이트."""
    tgt_all = torch.tensor(Y01, dtype=torch.float32)
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)

    tau = torch.tensor(
        np.clip(T_CENTER + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD, 0, T_MAX),
        dtype=torch.float32,
        requires_grad=True
    )
    w = (torch.randn(15) * 0.1).requires_grad_(True)
    theta = torch.zeros(15, requires_grad=True)

    opt_tau = torch.optim.Adam([tau], lr=LR_TAU)
    opt_w = torch.optim.Adam([w, theta], lr=LR_W)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_hist = []

    for epoch in range(EPOCHS):
        # 동시 업데이트
        opt_tau.zero_grad()
        opt_w.zero_grad()

        K = build_kernel(torch.clamp(tau, 0, T_MAX))
        wm = w.view(1, -1) * masks
        membrane = wm @ K.t()
        score = membrane.max(dim=1).values
        logit = GAIN * (score - theta.view(1, -1))
        loss = loss_fn(logit, tgt_all)

        loss.backward()
        opt_tau.step()
        opt_w.step()

        with torch.no_grad():
            tau.clamp_(0, T_MAX)

        avg_loss = float(loss)
        loss_hist.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"v2 epoch {epoch+1:3d}: loss {avg_loss:.6f}")

    return np.array(loss_hist)


def plot_comparison(loss_v1, loss_v2):
    """Loss 곡선 비교."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 전체 곡선
    ax = axes[0]
    ax.plot(loss_v1, lw=2, label="v1 (alternating)", color="tab:red", alpha=0.8)
    ax.plot(loss_v2, lw=2, label="v2 (simultaneous)", color="tab:blue", alpha=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("GD v1 vs v2: Full loss curve")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 확대 (처음 100 에폭)
    ax = axes[1]
    ax.plot(loss_v1[:100], lw=2, label="v1 (alternating)", color="tab:red", alpha=0.8)
    ax.plot(loss_v2[:100], lw=2, label="v2 (simultaneous)", color="tab:blue", alpha=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("GD v1 vs v2: First 100 epochs (zoom)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "gd_loss_comparison.png", dpi=120)
    print(f"\n그림 저장: {RESULTS / 'gd_loss_comparison.png'}")

    # 통계
    print("\n=== Loss 통계 ===")
    print(f"v1 초기: {loss_v1[0]:.6f} → 최종: {loss_v1[-1]:.6f} (감소: {loss_v1[0]-loss_v1[-1]:.6f})")
    print(f"v2 초기: {loss_v2[0]:.6f} → 최종: {loss_v2[-1]:.6f} (감소: {loss_v2[0]-loss_v2[-1]:.6f})")
    print(f"\n최종 loss 차이: v1={loss_v1[-1]:.6f}, v2={loss_v2[-1]:.6f} (v2가 {loss_v1[-1]-loss_v2[-1]:.6f} 더 낮음)")


def main():
    print("=== GD v1 (Alternating) ===")
    loss_v1 = learn_v1_alternating()

    print("\n=== GD v2 (Simultaneous) ===")
    loss_v2 = learn_v2_simultaneous()

    print("\n=== 비교 ===")
    plot_comparison(loss_v1, loss_v2)


if __name__ == "__main__":
    main()
