"""100% GD sweep: (n_pairs, lr_tau, lr_w, epochs) 격자로 실행.

각 케이스마다:
  - learn_gd 실행 (τ, w 동시 최적화)
  - 결과 시각화 저장

격자(GRID): n_pairs, lr_tau, lr_w, epochs
결과 디렉토리 태그: gd_pair={n}_lrtau={lt}_lrw={lw}_ep={ep}

특징: 기본 digit/letter 패턴 사용

실행: python "템퍼럴_stdp/sweep_only_gd.py"
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # sweep: 창 없이 저장만
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import binary_character as data  # Original digit/letter patterns

RESULTS_SWEEP = HERE / "results_gd_sweep"
RESULTS_SWEEP.mkdir(parents=True, exist_ok=True)
np.random.seed(0)
torch.manual_seed(0)

# ===== 격자 (여기 편집) =====
GRID = {
    "n_pairs":  [10],
    "lr_tau":   [0.0001, 0.001, 0.01],
    "lr_w":     [0.0001, 0.001, 0.01],
    "epochs":   [1000, 10000, 100000],
}
# ===== 고정 파라미터 =====
T_MAX = 10.0
T_CENTER = 5.0
INIT_SPREAD = 3.0
DT = 0.05
OUT_TAU = 0.4
GAIN = 4.0
EARLY_STOP = True  # 10/10 달성하면 조기 종료 (True/False)
# ================

n_steps = int(round(T_MAX / DT))
t_grid = np.arange(n_steps) * DT

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
DIG = digit_images.reshape(10, 15).astype(np.float64)
LET = letter_images.reshape(10, 15).astype(np.float64)


# ---------------------------------------------------------------------------
def build_kernel(tau_t):
    """tau_t: [15] torch tensor. 반환 K: [n_steps, 15]"""
    tg = torch.tensor(t_grid, dtype=torch.float32).view(-1, 1)
    d = tg - tau_t.view(1, -1)
    K = torch.where(d >= 0, torch.exp(-d / OUT_TAU), torch.zeros_like(d))
    return K


def learn_gd(X01, Y01, out_neuron, n_pairs, lr_tau, lr_w, epochs):
    """τ, w를 번갈아 gradient descent."""
    tgt = torch.tensor(Y01[:n_pairs, out_neuron], dtype=torch.float32)
    masks = torch.tensor(X01[:n_pairs] > 0.5, dtype=torch.float32)

    tau = torch.tensor(
        np.clip(T_CENTER + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD, 0, T_MAX),
        dtype=torch.float32,
        requires_grad=True
    )
    w = (torch.randn(15) * 0.1).requires_grad_(True)
    theta = torch.zeros(1, requires_grad=True)

    opt_tau = torch.optim.Adam([tau], lr=lr_tau)
    opt_w = torch.optim.Adam([w, theta], lr=lr_w)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        # 동시 업데이트: τ, w, theta 한 번에
        opt_tau.zero_grad()
        opt_w.zero_grad()

        K = build_kernel(torch.clamp(tau, 0, T_MAX))
        wm = w.view(1, -1) * masks
        membrane = wm @ K.t()
        score = membrane.max(dim=1).values
        logit = GAIN * (score - theta)
        loss = loss_fn(logit, tgt)

        loss.backward()
        opt_tau.step()
        opt_w.step()

        with torch.no_grad():
            tau.clamp_(0, T_MAX)

        # 10/10 달성하면 스킵 (EARLY_STOP=True일 때만)
        with torch.no_grad():
            K_eval = build_kernel(torch.clamp(tau, 0, T_MAX))
            wm = w.view(1, -1) * masks
            score = (wm @ K_eval.t()).max(dim=1).values
            pred = (score >= theta).int().numpy()
            acc = int((pred == tgt).sum())
        if EARLY_STOP and acc == n_pairs:
            break

    with torch.no_grad():
        K = build_kernel(torch.clamp(tau, 0, T_MAX))
        wm = w.view(1, -1) * masks
        score = (wm @ K.t()).max(dim=1).values
        pred = (score >= theta).int().numpy()
    return pred


def predict(X01, Y01, out_neuron, n_pairs, lr_tau, lr_w, epochs):
    """판정용 (출력 all)."""
    pred_all = np.zeros((n_pairs, 15), dtype=int)
    for j in range(15):
        pred_all[:, j] = learn_gd(X01, Y01, j, n_pairs, lr_tau, lr_w, epochs)
    return pred_all


def recall_grid(pred_all, X01, Y01, n_pairs, path, title):
    fig, axes = plt.subplots(3, n_pairs, figsize=(1.45 * n_pairs, 4.6), squeeze=False)
    for k in range(n_pairs):
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
    n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(n_pairs))
    fig.suptitle(f"{title}  exact {n_exact}/{n_pairs}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def per_pixel_fig(per_pixel, n_pairs, n_exact, bit, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 5.2))
    grid = per_pixel.reshape(5, 3)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=n_pairs)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{grid[r, c]}/{n_pairs}", ha="center", va="center", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nexact {n_exact}/{n_pairs}, bit {bit:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def stats(pred_all, Y01, n_pairs):
    n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(n_pairs))
    bit = float((pred_all == Y01.astype(int)).mean())
    return n_exact, bit


# ---------------------------------------------------------------------------
def run_case(cfg):
    n = cfg["n_pairs"]; lrt = cfg["lr_tau"]; lrw = cfg["lr_w"]; ep = cfg["epochs"]
    tag = f"gd_pair={n}_lrtau={lrt}_lrw={lrw}_ep={ep}"
    outdir = RESULTS_SWEEP / tag
    outdir.mkdir(parents=True, exist_ok=True)

    X01 = DIG[:n]; Y01 = LET[:n]

    pred_all = np.zeros((n, 15), dtype=int)
    per_pixel = np.zeros(15, dtype=int)

    for j in range(15):
        pred_all[:, j] = learn_gd(X01, Y01, j, n, lrt, lrw, ep)
        per_pixel[j] = int((pred_all[:, j] == Y01[:, j].astype(int)).sum())

    # 시각화
    n_ex, bt = stats(pred_all, Y01, n)
    recall_grid(pred_all, X01, Y01, n, outdir / "recall_grid.png", "Recall (100% GD)")
    per_pixel_fig(per_pixel, n, n_ex, bt, outdir / "per_pixel_acc.png", "Per-pixel acc")

    print(f"[{tag}]  exact {n_ex}/{n}, bit {bt:.3f}")
    return {"tag": tag, "exact": n_ex, "bit": round(bt, 3)}


def main():
    combos = [dict(zip(GRID.keys(), v)) for v in itertools.product(*GRID.values())]
    print(f"총 {len(combos)} 케이스\n")
    rows = [run_case(c) for c in combos]
    # 요약 CSV
    cols = ["tag", "exact", "bit"]
    lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]
    (RESULTS_SWEEP / "summary.csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n요약: {RESULTS_SWEEP / 'summary.csv'}")


if __name__ == "__main__":
    main()
