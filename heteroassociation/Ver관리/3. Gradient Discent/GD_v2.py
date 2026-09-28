"""100% Gradient Descent v2: Simultaneous τ & w optimization.

메커니즘
--------
입력(digit, CS) →[τ,w]→ 출력 coincidence 봉우리 → 타깃(letter, HS) 비교
τ와 w를 autograd로 동시에 최적화 (simultaneous update).

모델
----
멤브레인: v(t) = Σ_i w_i * K(t - tau_i),  K = exp(-(t-tau_i)/OUT_TAU) if t>=tau_i else 0
판정: score = max_t v(t),  발화 <=> score >= theta

특징 (v2)
---------
- HS (letter A~J) 비트맵: 균형잡힌 설계 (각 픽셀마다 1이 5개, 0이 5개)
- 최대 orthogonality → linear separability 향상
- τ, w 동시 업데이트 → 일관된 gradient

실행: python "템퍼럴_stdp/GD_v2.py"
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

import binary_character as data  # Original digit/letter patterns

RESULTS = HERE / "results_only_gd"
RESULTS.mkdir(parents=True, exist_ok=True)
PER_OUT = RESULTS / "per_output"
PER_OUT.mkdir(parents=True, exist_ok=True)
EVO = RESULTS / "evolution"
EVO.mkdir(parents=True, exist_ok=True)

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
# 학습
EPOCHS = 10000
LR_TAU = 0.001      # τ 학습률
LR_W = 0.01        # w 학습률
EARLY_STOP = True  # 10/10 달성하면 조기 종료 (True/False)
# ================

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]
Y01 = letter_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]

n_steps = int(round(T_MAX / DT))
t_grid = np.arange(n_steps) * DT


def build_kernel(tau_t):
    """tau_t: [15] torch tensor. 반환 K: [n_steps, 15]"""
    tg = torch.tensor(t_grid, dtype=torch.float32).view(-1, 1)  # [T,1]
    d = tg - tau_t.view(1, -1)  # [T,15]
    K = torch.where(d >= 0, torch.exp(-d / OUT_TAU), torch.zeros_like(d))
    return K


def learn_gd_all():
    """모든 15 출력을 동시에 학습. 각 출력별 τ, w, theta."""
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)  # [D,15]
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_hist = []

    # 각 출력별 파라미터
    tau_list = []
    w_list = []
    theta_list = []
    for _ in range(15):
        tau_list.append(torch.tensor(
            np.clip(T_CENTER + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD, 0, T_MAX),
            dtype=torch.float32,
            requires_grad=True
        ))
        w_list.append((torch.randn(15) * 0.1).requires_grad_(True))
        theta_list.append(torch.zeros(1, requires_grad=True))

    opt_tau = torch.optim.Adam(tau_list, lr=LR_TAU)
    opt_w = torch.optim.Adam(w_list + theta_list, lr=LR_W)

    for epoch in range(EPOCHS):
        # 동시 업데이트: 모든 출력에 대해 τ, w, theta 한 번에
        opt_tau.zero_grad()
        opt_w.zero_grad()

        loss_total = 0.0
        for j in range(15):
            tgt = torch.tensor(Y01[:, j], dtype=torch.float32)  # [D]
            K = build_kernel(torch.clamp(tau_list[j], 0, T_MAX))  # [T,15]
            wm = w_list[j].view(1, -1) * masks  # [D,15]
            membrane = wm @ K.t()  # [D,T]
            score = membrane.max(dim=1).values  # [D]
            logit = GAIN * (score - theta_list[j])  # [D]
            loss = loss_fn(logit, tgt)
            loss_total = loss_total + loss

        loss_total.backward()
        opt_tau.step()
        opt_w.step()

        # τ 범위 클립
        with torch.no_grad():
            for tau in tau_list:
                tau.clamp_(0, T_MAX)

        avg_loss = float(loss_total) / 15
        loss_hist.append(avg_loss)

        # 매 에폭 accuracy 체크
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                pred_all = np.zeros((N_PAIRS, 15), dtype=int)
                for j in range(15):
                    K_eval = build_kernel(torch.clamp(tau_list[j], 0, T_MAX))
                    wm = w_list[j].view(1, -1) * masks
                    score = (wm @ K_eval.t()).max(dim=1).values
                    pred_all[:, j] = (score >= theta_list[j]).int().numpy().flatten()
                n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(N_PAIRS))
            print(f" epoch {epoch:4d}: loss {avg_loss:.4f}, exact {n_exact}/{N_PAIRS}")

        # 모든 출력이 정확하면 스킵
        if EARLY_STOP and (epoch + 1) % 50 == 0:
            with torch.no_grad():
                pred_all = np.zeros((N_PAIRS, 15), dtype=int)
                for j in range(15):
                    K_eval = build_kernel(torch.clamp(tau_list[j], 0, T_MAX))
                    wm = w_list[j].view(1, -1) * masks
                    score = (wm @ K_eval.t()).max(dim=1).values
                    pred_all[:, j] = (score >= theta_list[j]).int().numpy().flatten()
                n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(N_PAIRS))
            if n_exact == N_PAIRS:
                print(f" >> 10/10 달성! epoch {epoch+1}에서 종료")
                break

    # 각 출력별로 numpy로 변환
    tau_all = np.stack([t.detach().numpy() for t in tau_list])  # [15, 15]
    w_all = np.stack([w.detach().numpy() for w in w_list])  # [15, 15]
    theta_all = np.stack([th.detach().numpy() for th in theta_list]).flatten()  # [15]
    return tau_all, w_all, theta_all, np.array(loss_hist)


def predict(tau_all, w_all, theta_all):
    """판정. tau_all, w_all: [15, 15], theta_all: [15]"""
    pred_all = np.zeros((N_PAIRS, 15), dtype=int)
    score_all = np.zeros((N_PAIRS, 15))
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)

    for j in range(15):
        tau_t = torch.tensor(tau_all[j], dtype=torch.float32)
        w_t = torch.tensor(w_all[j], dtype=torch.float32)
        K = build_kernel(tau_t)
        wm = w_t.view(1, -1) * masks
        score = (wm @ K.t()).max(dim=1).values.numpy()
        score_all[:, j] = score
        pred_all[:, j] = (score >= theta_all[j]).astype(int)

    return pred_all, score_all


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


def plot_timing_and_peaks(j, tau, w, theta_j):
    """입력 스파이크 타이밍(τ) + 멤브레인 봉우리 + 예측."""
    tau_t = torch.tensor(tau, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    K = build_kernel(tau_t)
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)
    wm = w_t.view(1, -1) * masks
    score = (wm @ K.t()).max(dim=1).values.numpy()
    pred = (score >= theta_j).astype(int)
    peaks = score
    tgt = Y01[:, j].astype(int)
    acc = int((pred == tgt).sum())

    ncol = min(5, N_PAIRS)
    nrow = int(np.ceil(N_PAIRS / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.5 * nrow),
                            sharex=True, sharey=True, squeeze=False)
    flat = axs.ravel()

    for i in range(N_PAIRS):
        ax = flat[i]
        A = np.where(X01[i] > 0.5)[0]
        ax.scatter(tau[A], A, s=26, color="tab:blue", zorder=3)  # 활성 픽셀 발화
        if len(A) > 1:
            ax.axvspan(tau[A].min(), tau[A].max(), color="tab:orange", alpha=0.12)
        fired = peaks[i] >= theta_j
        color = "tab:red" if fired else "tab:gray"
        ax.axvline(peaks[i], color=color, ls="-", lw=1.4, alpha=0.75, zorder=2)
        ok = "O" if pred[i] == tgt[i] else "X"
        ax.set_title(f"d{i} tgt={tgt[i]} pred={pred[i]} {ok}\npeak={peaks[i]:.2f}",
                    fontsize=8, color=("green" if pred[i] == tgt[i] else "red"))
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(-1, 15)
        ax.set_yticks([0, 5, 10, 14])

    for k in range(N_PAIRS, len(flat)):
        flat[k].axis("off")

    fig.suptitle(f"out {j}: spike timing (τ) & peaks  acc {acc}/{N_PAIRS}", fontsize=12)
    fig.supxlabel("spike time τ")
    fig.supylabel("input pixel")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(PER_OUT / f"gd_n{j:02d}_timing.png", dpi=110)
    plt.close(fig)


def plot_membranes(j, tau, w, theta_j):
    """각 패턴의 출력 멤브레인 v(t)."""
    tau_t = torch.tensor(tau, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    K = build_kernel(tau_t)
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)
    wm = w_t.view(1, -1) * masks
    score = (wm @ K.t()).max(dim=1).values.numpy()
    pred = (score >= theta_j).astype(int)
    tgt = Y01[:, j].astype(int)
    t_axis = np.arange(n_steps) * DT

    ncol = min(5, N_PAIRS)
    nrow = int(np.ceil(N_PAIRS / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.5 * nrow),
                            sharex=True, sharey=False, squeeze=False)
    flat = axs.ravel()

    for i in range(N_PAIRS):
        ax = flat[i]
        A = np.where(X01[i] > 0.5)[0]
        tr = membrane_trace(tau, w, A, theta_j)
        ax.plot(t_axis, tr, lw=1.4, color=("tab:red" if tgt[i] == 1 else "tab:blue"))
        ax.axhline(theta_j, color="k", ls=":", lw=1, alpha=0.6)
        fired = tr.max() >= theta_j
        ok = "O" if pred[i] == tgt[i] else "X"
        ax.set_title(f"d{i} {'FIRE' if fired else 'sil'} {ok}",
                    fontsize=8, color=("green" if pred[i] == tgt[i] else "red"))
        ax.set_xlim(0, T_MAX)

    for k in range(N_PAIRS, len(flat)):
        flat[k].axis("off")

    fig.suptitle(f"out {j}: output membrane v(t)  (threshold={theta_j:.2f})", fontsize=12)
    fig.supxlabel("time")
    fig.supylabel("v")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(PER_OUT / f"gd_n{j:02d}_membranes.png", dpi=110)
    plt.close(fig)


def plot_output(j, tau, w, theta_j, loss_hist):
    """출력 j 상세 그림."""
    tau_t = torch.tensor(tau, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    K = build_kernel(tau_t)
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)
    wm = w_t.view(1, -1) * masks
    score = (wm @ K.t()).max(dim=1).values.numpy()
    pred = (score >= theta_j).astype(int)
    peaks = score
    tgt = Y01[:, j].astype(int)
    acc = int((pred == tgt).sum())

    # 1) 손실 곡선
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(loss_hist, color="tab:red", lw=1.5)
    axes[0, 0].set_title(f"out {j}: BCE loss"); axes[0, 0].set_xlabel("epoch")

    # 2) 가중치 히트맵
    W = w.reshape(5, 3)
    m = np.abs(W).max() or 1e-9
    im = axes[0, 1].imshow(W, cmap="RdBu_r", vmin=-m, vmax=m)
    for r in range(5):
        for c in range(3):
            axes[0, 1].text(c, r, f"{W[r, c]:.2f}", ha="center", va="center", fontsize=7)
    axes[0, 1].set_title("weights w (5×3)"); axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
    plt.colorbar(im, ax=axes[0, 1])

    # 3) τ 히트맵
    T = tau.reshape(5, 3)
    im2 = axes[1, 0].imshow(T, cmap="viridis", vmin=0, vmax=T_MAX)
    for r in range(5):
        for c in range(3):
            axes[1, 0].text(c, r, f"{T[r, c]:.1f}", ha="center", va="center", fontsize=7)
    axes[1, 0].set_title("spike times τ (5×3)"); axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    plt.colorbar(im2, ax=axes[1, 0])

    # 4) 봉우리 막대
    colors = ["tab:red" if tgt[i] == 1 else "tab:blue" for i in range(N_PAIRS)]
    axes[1, 1].bar(np.arange(N_PAIRS), peaks, color=colors, alpha=0.85)
    axes[1, 1].axhline(theta_j, color="k", ls="--", lw=1, label=f"theta={theta_j:.2f}")
    for i in range(N_PAIRS):
        ok = pred[i] == tgt[i]
        axes[1, 1].text(i, peaks[i] + 0.02, "O" if ok else "X", ha="center",
                       color=("green" if ok else "red"), fontweight="bold")
    axes[1, 1].set_xticks(np.arange(N_PAIRS))
    axes[1, 1].set_ylabel("peak"); axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title(f"Output peaks (acc {acc}/{N_PAIRS})")

    fig.suptitle(f"out {j}: 100% GD (τ & w jointly optimized)", fontsize=12)
    fig.tight_layout(); fig.savefig(PER_OUT / f"gd_n{j:02d}_summary.png", dpi=110)
    plt.close(fig)


def plot_recall_grid(pred_all):
    """recall 그리드."""
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
    fig.suptitle(f"Recall (100% GD)  exact {n_exact}/{N_PAIRS}", fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / "gd_recall_grid.png", dpi=120)


def main():
    print("=== 100% Gradient Descent (모든 출력 동시 학습, 각 출력별 theta) ===")
    tau_all, w_all, theta_all, loss_hist = learn_gd_all()

    pred_all = np.zeros((N_PAIRS, 15), dtype=int)
    per_pixel = np.zeros(15, dtype=int)

    # 예측 계산
    for k in range(N_PAIRS):
        for j in range(15):
            A = np.where(X01[k] > 0.5)[0]
            tr = membrane_trace(tau_all[j], w_all[j], A, theta_all[j])
            pred_all[k, j] = 1 if tr.max() >= theta_all[j] else 0

    # 각 출력별 시각화 및 정확도
    for j in range(15):
        per_pixel[j] = int((pred_all[:, j] == Y01[:, j].astype(int)).sum())
        plot_output(j, tau_all[j], w_all[j], theta_all[j], loss_hist)
        plot_timing_and_peaks(j, tau_all[j], w_all[j], theta_all[j])
        plot_membranes(j, tau_all[j], w_all[j], theta_all[j])
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
            ax.text(c, r, f"{grid[r, c]}/{N_PAIRS}", ha="center", va="center", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Per-pixel acc (100% GD)\nexact {n_exact}/{N_PAIRS}, bit {bit:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(RESULTS / "gd_per_pixel_acc.png", dpi=120)

    print(f"\n그림 저장:")
    print(f"  요약: {PER_OUT}")
    print(f"  전체: {RESULTS}")
    plt.show()


if __name__ == "__main__":
    main()
