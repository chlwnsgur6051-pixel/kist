"""temporal_weighted (tau 딜레이 + w 가중치) 를 파라미터 격자로 일괄 실행하는 sweep.

각 케이스마다:
  1) tau(스파이크 딜레이) 클러스터링 끝 -> 딜레이 단계 시각화 저장 (delay_*)
  2) w(autograd) 학습 끝            -> 웨이트 단계 시각화 저장 (weight_*)

격자(GRID): n_pairs, eta_cluster, eta_spread, epochs_w, lr_w
결과 디렉토리 태그: pair={n}_ETA={ec},{es}_wlr={lr}_wEpoch={ew}

실행: python "템퍼럴_stdp/sweep_weighted.py"
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # sweep: 창 없이 저장만
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import binary_character as data

RESULTS_SWEEP = HERE / "results_weighted_sweep"
RESULTS_SWEEP.mkdir(parents=True, exist_ok=True)
np.random.seed(0)
torch.manual_seed(0)

# ===== 격자 (여기 편집) =====
GRID = {
    "n_pairs":     [10],
    "eta_cluster": [0.03, 0.04, 0.05],
    "eta_spread":  [0.03, 0.04, 0.05],
    "epochs_w":    [20000,100000],
    "lr_w":        [0.001,0.005,0.0001],
}
# ===== 고정 파라미터 =====
T_MAX = 10.0
T_CENTER = 5.0
INIT_SPREAD = 3.0
DT = 0.05
OUT_TAU = 0.4
GAIN = 4.0
EPOCHS_TAU = 200
SPREAD_THR = 2.0          # 딜레이 단계 판정(spread<THR -> 1)
# ================

n_steps = int(round(T_MAX / DT))
t_grid = np.arange(n_steps) * DT

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
DIG = digit_images.reshape(10, 15).astype(np.float64)
LET = letter_images.reshape(10, 15).astype(np.float64)


# ---------------------------------------------------------------------------
def learn_tau(X01, ACTIVE, tgt, n_pairs, eta_c, eta_s):
    tau = np.clip(T_CENTER + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD, 0, T_MAX)
    for _ in range(EPOCHS_TAU):
        for p in np.random.permutation(n_pairs):
            A = ACTIVE[p]
            if len(A) == 0:
                continue
            dev = tau[A] - tau[A].mean()
            if tgt[p] == 1:
                tau[A] -= eta_c * dev
            else:
                tau[A] += eta_s * dev
            tau = np.clip(tau, 0, T_MAX)
    return tau


def spread_pred(X01, tau, n_pairs):
    """딜레이 단계 판정: 활성 스파이크 spread < THR -> 1."""
    preds = []
    for i in range(n_pairs):
        A = np.where(X01[i] > 0.5)[0]
        s = 0.0 if len(A) <= 1 else float(tau[A].max() - tau[A].min())
        preds.append(1 if s < SPREAD_THR else 0)
    return np.array(preds)


def build_kernel(tau):
    tg = torch.tensor(t_grid, dtype=torch.float32).view(-1, 1)
    tau_t = torch.tensor(tau, dtype=torch.float32).view(1, -1)
    d = tg - tau_t
    return torch.where(d >= 0, torch.exp(-d / OUT_TAU), torch.zeros_like(d))


def learn_w(X01, tau, tgt, n_pairs, epochs_w, lr_w):
    K = build_kernel(tau)
    masks = torch.tensor(X01 > 0.5, dtype=torch.float32)
    target = torch.tensor(tgt, dtype=torch.float32)
    w = (torch.randn(15) * 0.1).requires_grad_(True)
    theta = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, theta], lr=lr_w)
    lf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs_w):
        opt.zero_grad()
        wm = w.view(1, -1) * masks
        score = (wm @ K.t()).max(dim=1).values
        loss = lf(GAIN * (score - theta), target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        wm = w.view(1, -1) * masks
        score = (wm @ K.t()).max(dim=1).values
        pred = (score >= theta).int().numpy()
    return w.detach().numpy(), float(theta.detach()), pred


# ---------------------------------------------------------------------------
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
    n = cfg["n_pairs"]; ec = cfg["eta_cluster"]; es = cfg["eta_spread"]
    ew = cfg["epochs_w"]; lr = cfg["lr_w"]
    tag = f"pair={n}_ETA={ec},{es}_wlr={lr}_wEpoch={ew}"
    outdir = RESULTS_SWEEP / tag
    outdir.mkdir(parents=True, exist_ok=True)

    X01 = DIG[:n]; Y01 = LET[:n]
    ACTIVE = [np.where(X01[i] > 0.5)[0] for i in range(n)]

    pred_tau = np.zeros((n, 15), dtype=int)   # 딜레이 단계 예측(spread)
    pred_w = np.zeros((n, 15), dtype=int)     # 웨이트 단계 예측
    ppx_tau = np.zeros(15, dtype=int)
    ppx_w = np.zeros(15, dtype=int)

    for j in range(15):
        tgt = Y01[:, j].astype(int)
        tau = learn_tau(X01, ACTIVE, tgt, n, ec, es)     # 1) 딜레이 조정
        pt = spread_pred(X01, tau, n)
        pred_tau[:, j] = pt; ppx_tau[j] = int((pt == tgt).sum())
        _, _, pw = learn_w(X01, tau, tgt, n, ew, lr)     # 2) 웨이트 학습
        pred_w[:, j] = pw; ppx_w[j] = int((pw == tgt).sum())

    # 1) 딜레이 단계 시각화
    et, bt = stats(pred_tau, Y01, n)
    recall_grid(pred_tau, X01, Y01, n, outdir / "delay_recall_grid.png", "DELAY(tau) recall")
    per_pixel_fig(ppx_tau, n, et, bt, outdir / "delay_per_pixel.png", "DELAY per-pixel acc")
    # 2) 웨이트 단계 시각화
    ew_ex, bw = stats(pred_w, Y01, n)
    recall_grid(pred_w, X01, Y01, n, outdir / "weight_recall_grid.png", "WEIGHT(tau+w) recall")
    per_pixel_fig(ppx_w, n, ew_ex, bw, outdir / "weight_per_pixel.png", "WEIGHT per-pixel acc")

    print(f"[{tag}]  delay: exact {et}/{n} bit {bt:.3f}  ->  weight: exact {ew_ex}/{n} bit {bw:.3f}")
    return {"tag": tag, "delay_exact": et, "delay_bit": round(bt, 3),
            "weight_exact": ew_ex, "weight_bit": round(bw, 3)}


def main():
    combos = [dict(zip(GRID.keys(), v)) for v in itertools.product(*GRID.values())]
    print(f"총 {len(combos)} 케이스\n")
    rows = [run_case(c) for c in combos]
    # 요약 CSV
    cols = ["tag", "delay_exact", "delay_bit", "weight_exact", "weight_bit"]
    lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]
    (RESULTS_SWEEP / "summary.csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n요약: {RESULTS_SWEEP / 'summary.csv'}")


if __name__ == "__main__":
    main()
