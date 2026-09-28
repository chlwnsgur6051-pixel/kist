"""단층 + Hebbian 저장(연관기억) + temporal(latency) 판독.

temporal_stdp.py(=supervised chronotron)와 짝을 이루는 'Hebbian 연관기억' 버전.
학습을 오차보정이 아니라 **외적(outer product) 한 방 저장**으로 한다.

- 입력: 숫자 15bit -> bipolar(0->-1, 1->+1)  (off 픽셀도 -1 로 기여 = complement 효과)
- 저장(Hebbian):  w_i = (1/K) * sum_k  t_bipolar_k * x_bipolar_ki
    (타깃도 bipolar. 이게 표준 상관행렬 연관기억 = BAM 계열)
- 회상 드라이브:  D_k = sum_i w_i * x_bipolar_ki
    D_k ≈ t_k * (15/K) + crosstalk(입력 겹침)   -> 겹침이 크면 부호가 틀림
- 출력(latency):  D 클수록 이른 발화(->"1"), 작을수록 늦은 발화(->"0")
    판정: latency < T_CENTER -> 1

핵심: 이 Hebbian 저장은 rank-9 겹치는 숫자에서 crosstalk 로 XNOR 픽셀(7)이 틀린다.
      -> 우리가 논의한 '단층 연관기억의 천장'을 temporal 프레임에서 그대로 재현.

실행: python "템퍼럴_stdp/temporal_hebbian.py"
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

# ===== 실험 설정 =====
OUT_NEURON = 1          # letter 픽셀. 0/12는 자명(항상1), 5/8 균형, 7=XNOR(어려움)
SWEEP_ALL = False       # True 면 15개 출력 전부 저장->픽셀별 정확도
RECALL_GRID = True      # SWEEP_ALL=False 여도 recall 비트맵 그리드는 같이 그림
                        # (글자 한 장을 그리려면 15픽셀을 다 저장해야 함)
T_MAX = 10.0
T_CENTER = 5.0
DT = 0.05
THRESHOLD = 1.0
RAMP = THRESHOLD / T_CENTER
GAIN = 0.15             # D_norm 을 latency 로 변환하는 이득 (< RAMP 유지)
# =====================

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(10, 15).astype(np.float64)      # [10,15] 0/1
Y01 = letter_images.reshape(10, 15).astype(np.float64)     # [10,15] 0/1
Xb = 2 * X01 - 1                                            # bipolar 입력 (-1/+1)
digit_names = [str(i) for i in range(10)]
K = 10


def hebbian_store(target_bits):
    """외적 한 방 저장: w_i = (1/K) sum_k t_bipolar_k * x_bipolar_ki."""
    tb = 2 * target_bits - 1                                # bipolar 타깃
    return (Xb.T @ tb) / K                                  # [15]


def recall_latency(w):
    """회상: 드라이브 D=Xb@w -> 정규화 -> latency. D 크면 빨리."""
    D = Xb @ w                                              # [10]
    Dn = D / (np.abs(D).max() + 1e-9)                       # [-1,1] 정규화
    lat = np.clip(THRESHOLD / np.maximum(RAMP + GAIN * Dn, 1e-3), 0, T_MAX)
    return D, lat


def simulate_ramp(drive_norm):
    """스텝별 막전위 v(t)=(RAMP+GAIN*Dn)*t, threshold 도달=latency. 궤적 반환."""
    n = int(round(T_MAX / DT))
    v, trace, lat = 0.0, [], None
    for step in range(n):
        v += (RAMP + GAIN * drive_norm) * DT
        if lat is None and v >= THRESHOLD:
            lat = step * DT; trace.append(THRESHOLD)
        else:
            trace.append(min(v, THRESHOLD * 1.1))
    return np.array(trace), (T_MAX if lat is None else lat)


# ---------------------------------------------------------------------------
def run_single(out_neuron):
    tgt = Y01[:, out_neuron].astype(int)
    w = hebbian_store(Y01[:, out_neuron])
    D, lat = recall_latency(w)
    Dn = D / (np.abs(D).max() + 1e-9)
    traces = [simulate_ramp(Dn[i])[0] for i in range(10)]
    pred = (lat < T_CENTER).astype(int)
    acc = int((pred == tgt).sum())
    t_axis = np.arange(len(traces[0])) * DT

    print(f"\n=== Hebbian, 출력 뉴런 {out_neuron} ===")
    print(f"타깃 = {tgt.tolist()}")
    for i in range(10):
        print(f" digit {i}: D={D[i]:+.3f}  latency={lat[i]:.2f}  "
              f"pred={pred[i]}  target={tgt[i]}  {'O' if pred[i]==tgt[i] else 'X'}")
    print(f"정확도: {acc}/10  (Hebbian 한 방 저장)")

    # 1) 가중치
    fig1, ax = plt.subplots(figsize=(4.2, 5))
    W = w.reshape(5, 3); m = np.abs(W).max() or 1e-9
    im = ax.imshow(W, cmap="RdBu_r", vmin=-m, vmax=m)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{W[r,c]:.2f}", ha="center", va="center", fontsize=8,
                    color=("white" if abs(W[r, c]) > 0.6 * m else "black"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Hebbian weights -> out {out_neuron}\n(red=+, blue=-)")
    fig1.colorbar(im, ax=ax, fraction=0.046)
    fig1.tight_layout(); fig1.savefig(RESULTS / f"heb_n{out_neuron}_1_weights.png", dpi=120)

    # 2) 막전위 램프
    fig2, ax = plt.subplots(figsize=(9, 5))
    for i in range(10):
        ax.plot(t_axis, traces[i], lw=1.5,
                color=("tab:red" if tgt[i] == 1 else "tab:blue"), alpha=0.8,
                label=f"{i}->{tgt[i]}")
    ax.axhline(THRESHOLD, color="k", ls=":", lw=1, label="threshold")
    ax.axvline(T_CENTER, color="gray", ls="--", lw=1, label=f"decision t={T_CENTER}")
    ax.set_xlabel("time"); ax.set_ylabel("membrane v")
    ax.set_title(f"Hebbian recall ramp (out {out_neuron})  earlier='1'(red)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig2.tight_layout(); fig2.savefig(RESULTS / f"heb_n{out_neuron}_2_membrane.png", dpi=120)

    # 3) latency 막대
    fig3, ax = plt.subplots(figsize=(9, 3.6))
    colors = ["tab:red" if tgt[i] == 1 else "tab:blue" for i in range(10)]
    ax.bar(np.arange(10), lat, color=colors, alpha=0.85)
    ax.axhline(T_CENTER, color="k", ls="--", lw=1)
    for i in range(10):
        ok = pred[i] == tgt[i]
        ax.text(i, lat[i] + 0.15, "O" if ok else "X", ha="center",
                color=("green" if ok else "red"), fontweight="bold")
    ax.set_xticks(np.arange(10)); ax.set_xticklabels(digit_names)
    ax.set_xlabel("digit"); ax.set_ylabel("latency"); ax.set_ylim(0, T_MAX)
    ax.set_title(f"Hebbian output latency (out {out_neuron}, acc {acc}/10)  red=1 blue=0")
    fig3.tight_layout(); fig3.savefig(RESULTS / f"heb_n{out_neuron}_3_latency.png", dpi=120)

    # 4) 10페어 스파이크 래스터
    fig4, axes = plt.subplots(2, 5, figsize=(15, 6.2), sharex=True, sharey=True)
    for i, axp in enumerate(axes.ravel()):
        active = np.where(X01[i] > 0.5)[0]
        axp.scatter(np.zeros_like(active), active, s=28, marker="o",
                    facecolors="none", edgecolors="tab:blue", linewidths=1.2, zorder=3)
        axp.scatter([lat[i]], [16], s=70, marker="o",
                    color=("tab:red" if pred[i] == 1 else "tab:gray"), edgecolors="k", zorder=4)
        axp.axhline(15, color="lightgray", lw=0.8)
        axp.axvline(T_CENTER, color="k", ls="--", lw=0.8)
        ok = "O" if pred[i] == tgt[i] else "X"
        axp.set_title(f"digit {i} tgt={tgt[i]} pred={pred[i]} {ok}", fontsize=9,
                      color=("green" if pred[i] == tgt[i] else "red"))
        axp.set_xlim(-0.6, T_MAX); axp.set_ylim(-1, 18)
        axp.set_yticks([0, 5, 10, 14, 16]); axp.set_yticklabels(["in0", "in5", "in10", "in14", "OUT"], fontsize=7)
    fig4.suptitle(f"10-pair spike raster (Hebbian, out {out_neuron})  in=pixels@t=0, out=latency",
                  fontsize=12)
    fig4.supxlabel("time"); fig4.supylabel("neuron")
    fig4.tight_layout(); fig4.savefig(RESULTS / f"heb_n{out_neuron}_4_spikes.png", dpi=120)

    return acc


def store_all_pixels():
    """15개 출력 픽셀 전부 저장 -> (W_all[15,15], pred_all[10,15])."""
    W_all = np.zeros((15, 15))
    for j in range(15):
        W_all[:, j] = hebbian_store(Y01[:, j])
    D_all = Xb @ W_all                                        # [10,15]
    Dn = D_all / (np.abs(D_all).max(axis=0, keepdims=True) + 1e-9)   # 출력별 정규화
    lat_all = np.clip(THRESHOLD / np.maximum(RAMP + GAIN * Dn, 1e-3), 0, T_MAX)
    return W_all, (lat_all < T_CENTER).astype(int)


def plot_recall_grid(pred_all):
    """input(숫자) / target(글자) / pred(글자) 5x3 비트맵 그리드."""
    fh = 5.1
    fig, axes = plt.subplots(3, 10, figsize=(14.5, fh), squeeze=False)
    for k in range(10):
        inp = X01[k].reshape(5, 3)
        tg = Y01[k].astype(int).reshape(5, 3)
        pr = pred_all[k].reshape(5, 3)
        axes[0][k].imshow(inp, cmap="gray_r", vmin=0, vmax=1)
        axes[1][k].imshow(tg, cmap="gray_r", vmin=0, vmax=1)
        axes[2][k].imshow(pr, cmap="gray_r", vmin=0, vmax=1)
        exact = np.array_equal(pr, tg); bit = float((pr == tg).mean())
        col = "green" if exact else "red"
        axes[0][k].set_title(f"{k}->{chr(65 + k)}", fontsize=9, color=col)
        axes[2][k].set_xlabel(f"{bit:.2f}", fontsize=8, color=col)
        for r in range(3):
            axes[r][k].set_xticks([]); axes[r][k].set_yticks([])
    axes[0][0].set_ylabel("input"); axes[1][0].set_ylabel("target"); axes[2][0].set_ylabel("pred")
    n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(10))
    bit = float((pred_all == Y01.astype(int)).mean())
    fig.suptitle(f"Recall: input -> target -> pred (Hebbian one-shot)\n"
                 f"exact {n_exact}/10, bit {bit:.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.62 / fh])
    fig.savefig(RESULTS / "heb_recall_grid.png", dpi=120)


def run_sweep_all():
    W_all, pred_all = store_all_pixels()
    per_pixel = (pred_all == Y01.astype(int)).sum(axis=0)
    for j in range(15):
        print(f"out neuron {j:2d}: {per_pixel[j]}/10")
    exact = int((pred_all == Y01.astype(int)).all(axis=1).sum())
    bit = float((pred_all == Y01.astype(int)).mean())
    print(f"\n전체: exact letters={exact}/10, bit_acc={bit:.3f}")

    plot_recall_grid(pred_all)

    fig, ax = plt.subplots(figsize=(4.5, 5.2))
    grid = per_pixel.reshape(5, 3)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=10)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{grid[r,c]}/10", ha="center", va="center", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Hebbian per-pixel acc\nexact {exact}/10, bit {bit:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(RESULTS / "heb_sweep_per_pixel_acc.png", dpi=120)


if __name__ == "__main__":
    if SWEEP_ALL:
        run_sweep_all()                              # recall 그리드 포함
    else:
        run_single(OUT_NEURON)
        if RECALL_GRID:                              # 단일 뉴런 실행에도 같이 첨부
            _, pred_all = store_all_pixels()
            n_exact = int((pred_all == Y01.astype(int)).all(axis=1).sum())
            bit = float((pred_all == Y01.astype(int)).mean())
            print(f"\n전체(15픽셀): exact letters={n_exact}/10, bit_acc={bit:.3f}")
            plot_recall_grid(pred_all)
    print(f"\n그림: {RESULTS}")
    plt.show()
