"""단층 + temporal(latency 출력) + STDP 학습 실험.

개념
----
- 입력: 우리 숫자 비트맵 15bit (make_digit_dataset).
- 출력: 단층 가중치(15-벡터) -> 출력 뉴런 1개. 은닉층 없음.
- 출력 코딩: latency (스파이크 시각).
    뉴런은 tonic ramp 로 중립(드라이브 0)이면 창 중앙 t=T_CENTER 에 발화.
    입력 드라이브 D = sum_i w_i*x_i 가 발화 시각을 평행이동:
      D 클수록 -> 이른 발화(0쪽) -> "1",  작을수록 -> 늦은 발화(-> "0").
    판정: latency < T_CENTER -> 1.
- 학습(temporal STDP / chronotron 식):
    타깃 스파이크 시각(1->이른, 0->늦은)을 주고, 실제 스파이크 시각 오차를
    줄이도록 시냅스를 국소 갱신. "너무 늦으면 활성 입력 강화(더 빨리 발화),
    너무 이르면 약화" -> spike-time-error 기반 로컬 규칙.

이 파일 하나로 학습+시뮬+시각화까지. 원본 모델/코드는 안 건드림.
실행: python "템퍼럴_stdp/temporal_stdp.py"
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
np.random.seed(0)

# ===== 실험 설정 =====
# 주의: letter 픽셀 0과 12는 A~J 전부 1이라 '항상 1'인 자명한 타깃(쓸모없음).
#       N_PAIRS=10 기준 균형 잡힌 것: 5,8(5:5).  어려운 픽셀: 7.
#       N_PAIRS 를 줄이면 픽셀별 1:0 비율이 달라진다(아래 실행 시 출력됨).
#       이 규칙은 오차보정이 없어서 타깃이 한쪽으로 치우치면 전부 그쪽으로 붕괴한다.
N_PAIRS = 2            # 사용할 (숫자->글자) 페어 수. 앞에서부터 N개. 최대 10.
OUT_NEURON = 2          # 볼 출력(letter) 픽셀. SWEEP_ALL=True 면 무시.
SWEEP_ALL = False       # True 면 15개 출력 뉴런 전부 학습해 픽셀별 정확도 표시
RECALL_GRID = True      # SWEEP_ALL=False 여도 recall 비트맵 그리드는 같이 그림
                        # (글자 한 장을 그리려면 15픽셀을 다 학습해야 해서 그만큼 느려짐)
T_MAX = 10.0            # 페어 윈도우 [0, T_MAX]
T_CENTER = 5.0          # 중립 드라이브 발화 시각(판정 경계)
T_TARGET_ONE = 2.0      # 타깃=1 -> 이른 발화 목표
T_TARGET_ZERO = 8.0     # 타깃=0 -> 늦은 발화 목표
DT = 0.05
THRESHOLD = 1.0
RAMP = THRESHOLD / T_CENTER       # 중립이면 latency = T_CENTER
LR = 0.02
EPOCHS = 400
W_CLIP = 2.0            # 오차보정 없으면 폭주하므로 가중치 범위 제한
# 래스터에 입력 스파이크를 찍을 시각. 모델엔 입력 딜레이가 없고 활성 픽셀이
# t=0 부터 상수 드라이브(D)를 넣으므로 0. 꺼진 픽셀은 모델이 안 쓰므로 안 그림.
T_IN = 0.0
# =====================

# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X = digit_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]   # [N,15] 입력
Y = letter_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]  # [N,15] 타깃(letter)
digit_names = [str(i) for i in range(N_PAIRS)]


def _grid_fig(n, cell=(3.0, 3.15), min_size=(10.0, 4.0)):
    """페어 수 n 에 맞춘 유연한 서브플롯 그리드.
    페어가 적어도 suptitle/범례가 잘리지 않도록 최소 캔버스 크기를 보장한다.
    반환: (fig, 사용할 axes 리스트, figsize)."""
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    figsize = (max(cell[0] * ncol, min_size[0]), max(cell[1] * nrow, min_size[1]))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for k in range(n, len(flat)):
        flat[k].axis("off")                       # 남는 칸 숨김
    return fig, flat[:n], figsize


# ---------------------------------------------------------------------------
# latency 뉴런
# ---------------------------------------------------------------------------
def latency_from_drive(D):
    """closed-form: latency = threshold / (ramp + D), [0,T_MAX] clamp. D 크면 빨리."""
    return np.clip(THRESHOLD / np.maximum(RAMP + D, 1e-3), 0, T_MAX)


def simulate_ramp(drive):
    """스텝별 막전위 v(t)=(ramp+drive)*t, threshold 도달 시각=latency. 궤적/latency."""
    n = int(round(T_MAX / DT))
    v, trace, lat = 0.0, [], None
    for step in range(n):
        v += (RAMP + drive) * DT
        if lat is None and v >= THRESHOLD:
            lat = step * DT
            trace.append(THRESHOLD)
        else:
            trace.append(min(v, THRESHOLD * 1.1))
    return np.array(trace), (T_MAX if lat is None else lat)


# ---------------------------------------------------------------------------
# temporal STDP 학습 (spike-time-error 기반 로컬 규칙)
# ---------------------------------------------------------------------------
def train_stdp(target_bits, epochs=EPOCHS, lr=LR):
    """순수 Hebbian(방향) 규칙 — 오차보정 없음(비생물학적이라 제거).
    타깃1 -> 활성 입력 강화(드라이브↑, 이른 발화 '가까워'),
    타깃0 -> 약화(드라이브↓, 늦은 발화 '멀어').  목표 latency/오차 사용 안 함."""
    w = np.random.randn(15) * 0.02
    acc_hist = []
    for ep in range(epochs):
        for i in np.random.permutation(N_PAIRS):
            sign = 1.0 if target_bits[i] > 0.5 else -1.0
            w = np.clip(w + lr * sign * X[i], -W_CLIP, W_CLIP)
        lat_all = latency_from_drive(X @ w)
        pred = (lat_all < T_CENTER).astype(int)
        acc_hist.append(int((pred == target_bits).sum()))
    return w, acc_hist


# ---------------------------------------------------------------------------
# 페어별 스파이크 래스터: 입력(활성 픽셀 @ t=0) + 출력(latency)
# 모델이 실제로 쓰는 것만 그린다 — 꺼진 픽셀은 D 에 기여도 학습 기여도 0 이라 제외.
# ---------------------------------------------------------------------------
def plot_spike_raster(tgt, lats, pred, out_neuron):
    fig, axs, (_, fh) = _grid_fig(N_PAIRS)
    for i, ax in enumerate(axs):
        on = np.where(X[i] > 0.5)[0]                          # 모델이 쓰는 픽셀(=1)
        # 활성 픽셀: t=0 부터 상수 드라이브 투입
        ax.scatter(np.full(len(on), T_IN), on, s=30, marker="o", color="tab:blue", zorder=3)
        ax.axvline(T_IN, color="tab:blue", ls=":", alpha=0.35)
        # 출력 스파이크 (latency 위치, 예측=1 빨강 / 0 회색)
        ax.scatter([lats[i]], [16], s=70, marker="o",
                   color=("tab:red" if pred[i] == 1 else "tab:gray"), edgecolors="k", zorder=4)
        ax.axhline(15, color="lightgray", lw=0.8)            # 입력/출력 구분선
        ax.axvline(T_CENTER, color="k", ls="--", lw=0.8)     # 판정 경계
        ok = "O" if pred[i] == tgt[i] else "X"
        ax.set_title(f"digit {i}  tgt={tgt[i]} pred={pred[i]} {ok}",
                     fontsize=9, color=("green" if pred[i] == tgt[i] else "red"))
        ax.set_xlim(-0.6, T_MAX); ax.set_ylim(-1, 18)
        ax.set_yticks([0, 5, 10, 14, 16])
        ax.set_yticklabels(["in0", "in5", "in10", "in14", "OUT"], fontsize=7)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", color="tab:blue", label=f"active input (=1) @ t={T_IN}"),
        Line2D([], [], marker="o", ls="", color="tab:red", label="output pred=1 (early)"),
        Line2D([], [], marker="o", ls="", color="tab:gray", label="output pred=0 (late)"),
        Line2D([], [], color="k", ls="--", label=f"decision t={T_CENTER}"),
    ]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{N_PAIRS}-pair spike raster (out neuron {out_neuron})  "
                 "input=active pixels only,  output=latency", fontsize=12)
    # 범례/제목 자리를 인치 기준으로 확보 (페어 수가 적어 캔버스가 작아도 안 잘리게).
    # supxlabel 은 범례 바로 위에 오도록 y 를 직접 지정 — 안 그러면 겹침.
    fig.supxlabel("time", y=0.42 / fh); fig.supylabel("neuron")
    fig.tight_layout(rect=[0, 0.85 / fh, 1, 1 - 0.45 / fh])
    fig.savefig(RESULTS / f"n{out_neuron}_5_spikes.png", dpi=120)


# ---------------------------------------------------------------------------
# 페어별 출력 멤브레인 v(t) — 숫자마다 한 칸씩 (fig2 겹쳐그리기의 분해 버전)
# ---------------------------------------------------------------------------
def plot_out_membrane(traces, lats, drives, tgt, pred, out_neuron):
    t_axis = np.arange(len(traces[0])) * DT
    fig, axs, (_, fh) = _grid_fig(N_PAIRS)
    for i, axp in enumerate(axs):
        color = "tab:red" if tgt[i] == 1 else "tab:blue"
        axp.plot(t_axis, traces[i], lw=1.6, color=color)
        axp.axhline(THRESHOLD, color="k", ls=":", lw=1)             # 임계
        axp.axvline(T_CENTER, color="gray", ls="--", lw=0.8)        # 판정 경계
        axp.axvline(lats[i], color=color, ls="-", lw=1.0, alpha=0.45)
        axp.scatter([lats[i]], [THRESHOLD], s=48, marker="v",       # 발화 시각
                    color=color, edgecolors="k", zorder=4)
        ok = "O" if pred[i] == tgt[i] else "X"
        axp.set_title(f"digit {i}  tgt={tgt[i]} pred={pred[i]} {ok}\n"
                      f"D={drives[i]:+.2f}   t*={lats[i]:.2f}", fontsize=9,
                      color=("green" if pred[i] == tgt[i] else "red"))
        axp.set_xlim(-0.2, T_MAX); axp.set_ylim(0, THRESHOLD * 1.2)
    fig.suptitle(f"Output membrane v(t) per digit (out {out_neuron})  "
                 "steeper ramp -> earlier crossing = '1'", fontsize=12)
    fig.supxlabel("time"); fig.supylabel("membrane v")
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.5 / fh])
    fig.savefig(RESULTS / f"n{out_neuron}_6_out_membrane.png", dpi=120)


# ---------------------------------------------------------------------------
# 단일 출력 뉴런 실험 + 시각화
# ---------------------------------------------------------------------------
def run_single(out_neuron):
    tgt = Y[:, out_neuron].astype(int)
    w, acc_hist = train_stdp(tgt)

    # 실제 ramp 시뮬로 latency/궤적
    traces, lats, drives = [], [], []
    for i in range(N_PAIRS):
        d = float(X[i] @ w)
        tr, lat = simulate_ramp(d)
        traces.append(tr); lats.append(lat); drives.append(d)
    lats = np.array(lats); drives = np.array(drives)
    pred = (lats < T_CENTER).astype(int)
    acc = int((pred == tgt).sum())
    t_axis = np.arange(len(traces[0])) * DT

    n_one = int(tgt.sum())
    print(f"\n=== 출력 뉴런 {out_neuron} (letter 픽셀 {out_neuron}), 페어 {N_PAIRS}개 ===")
    print(f"타깃 = {tgt.tolist()}  (1이 {n_one}개 / 0이 {N_PAIRS - n_one}개, "
          f"다수클래스 기준선 {max(n_one, N_PAIRS - n_one)}/{N_PAIRS})")
    for i in range(N_PAIRS):
        print(f" digit {i}: drive={drives[i]:+.3f}  latency={lats[i]:.2f}  "
              f"pred={pred[i]}  target={tgt[i]}  {'O' if pred[i]==tgt[i] else 'X'}")
    print(f"정확도: {acc}/{N_PAIRS}")

    # --- 시각화 4장 ---
    # 1) 가중치 5x3
    fig1, ax = plt.subplots(figsize=(4.2, 5))
    W = w.reshape(5, 3); m = np.abs(W).max() or 1e-9
    im = ax.imshow(W, cmap="RdBu_r", vmin=-m, vmax=m)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{W[r,c]:.2f}", ha="center", va="center", fontsize=8,
                    color=("white" if abs(W[r, c]) > 0.6 * m else "black"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Learned weights -> out neuron {out_neuron}\n(red=+, blue=-)")
    fig1.colorbar(im, ax=ax, fraction=0.046)
    fig1.tight_layout(); fig1.savefig(RESULTS / f"n{out_neuron}_1_weights.png", dpi=120)

    # 2) 막전위 램프 궤적
    fig2, ax = plt.subplots(figsize=(9, 5))
    for i in range(N_PAIRS):
        ax.plot(t_axis, traces[i], lw=1.5,
                color=("tab:red" if tgt[i] == 1 else "tab:blue"), alpha=0.8,
                label=f"{i}->{tgt[i]}")
    ax.axhline(THRESHOLD, color="k", ls=":", lw=1, label="threshold")
    ax.axvline(T_CENTER, color="gray", ls="--", lw=1, label=f"decision t={T_CENTER}")
    ax.set_xlabel("time"); ax.set_ylabel("membrane v")
    ax.set_title(f"Ramp membrane per digit (out {out_neuron})\n"
                 "earlier crossing='1'(red), later='0'(blue)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig2.tight_layout(); fig2.savefig(RESULTS / f"n{out_neuron}_2_membrane.png", dpi=120)

    # 3) latency 막대
    fig3, ax = plt.subplots(figsize=(9, 3.6))
    colors = ["tab:red" if tgt[i] == 1 else "tab:blue" for i in range(N_PAIRS)]
    ax.bar(np.arange(N_PAIRS), lats, color=colors, alpha=0.85)
    ax.axhline(T_CENTER, color="k", ls="--", lw=1, label=f"decision t={T_CENTER}")
    for i in range(N_PAIRS):
        ok = pred[i] == tgt[i]
        ax.text(i, lats[i] + 0.15, "O" if ok else "X", ha="center",
                color=("green" if ok else "red"), fontweight="bold")
    ax.set_xticks(np.arange(N_PAIRS)); ax.set_xticklabels(digit_names)
    ax.set_xlabel("digit"); ax.set_ylabel("output latency"); ax.set_ylim(0, T_MAX)
    ax.set_title(f"Output latency (out {out_neuron}, acc {acc}/{N_PAIRS})  red=1 blue=0")
    ax.legend(fontsize=8)
    fig3.tight_layout(); fig3.savefig(RESULTS / f"n{out_neuron}_3_latency.png", dpi=120)

    # 4) 학습 곡선 (정확도만; 오차 지표 제거)
    fig4, a0 = plt.subplots(figsize=(6, 3.4))
    a0.plot(acc_hist, color="tab:green")
    a0.set_title(f"exact-count / {N_PAIRS} (out {out_neuron})")
    a0.set_ylim(0, N_PAIRS + 0.5); a0.set_xlabel("epoch"); a0.set_ylabel("acc")
    fig4.tight_layout(); fig4.savefig(RESULTS / f"n{out_neuron}_4_learning.png", dpi=120)

    # 5) 페어별 스파이크 래스터
    plot_spike_raster(tgt, lats, pred, out_neuron)

    # 6) 페어별 출력 멤브레인 v(t)
    plot_out_membrane(traces, lats, drives, tgt, pred, out_neuron)

    return acc


# ---------------------------------------------------------------------------
# recall 비트맵 그리드: input(숫자) / target(글자) / pred(글자) 5x3
# 15픽셀 예측이 다 있어야 글자 한 장이 되므로 sweep 에서만 그린다.
# ---------------------------------------------------------------------------
def plot_recall_grid(pred_all):
    fh = 5.1
    fig, axes = plt.subplots(3, N_PAIRS, squeeze=False,
                             figsize=(max(1.45 * N_PAIRS, 5.6), fh))
    for k in range(N_PAIRS):
        inp = X[k].reshape(5, 3)
        tg = Y[k].astype(int).reshape(5, 3)
        pr = pred_all[k].reshape(5, 3)
        axes[0][k].imshow(inp, cmap="gray_r", vmin=0, vmax=1)
        axes[1][k].imshow(tg, cmap="gray_r", vmin=0, vmax=1)
        axes[2][k].imshow(pr, cmap="gray_r", vmin=0, vmax=1)
        exact = np.array_equal(pr, tg); bit = float((pr == tg).mean())
        col = "green" if exact else "red"
        axes[0][k].set_title(f"{k}->{chr(65 + k)}", fontsize=9, color=col)
        axes[2][k].set_xlabel(f"{bit:.2f}", fontsize=8, color=col)   # 글자별 bit 정확도
        for r in range(3):
            axes[r][k].set_xticks([]); axes[r][k].set_yticks([])
    axes[0][0].set_ylabel("input"); axes[1][0].set_ylabel("target"); axes[2][0].set_ylabel("pred")
    n_exact = sum(np.array_equal(pred_all[k], Y[k].astype(int)) for k in range(N_PAIRS))
    bit = float((pred_all == Y.astype(int)).mean())
    fig.suptitle(f"Recall: input -> target -> pred (temporal STDP)\n"
                 f"exact {n_exact}/{N_PAIRS}, bit {bit:.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.62 / fh])   # 2줄 제목 자리 확보
    fig.savefig(RESULTS / "recall_grid.png", dpi=120)


def train_all_pixels():
    """15개 출력 픽셀 전부 학습. 반환: (W_all[15,15], pred_all[N,15])."""
    W_all = np.zeros((15, 15))
    for j in range(15):
        W_all[:, j], _ = train_stdp(Y[:, j].astype(int))
    pred_all = (latency_from_drive(X @ W_all) < T_CENTER).astype(int)
    return W_all, pred_all


# ---------------------------------------------------------------------------
# 15개 출력 뉴런 전부 실험 -> 픽셀별 정확도 (letter 그리드)
# ---------------------------------------------------------------------------
def run_sweep_all():
    _, pred_all = train_all_pixels()
    per_pixel_acc = (pred_all == Y.astype(int)).sum(axis=0)
    for j in range(15):
        print(f"out neuron {j:2d}: {per_pixel_acc[j]}/{N_PAIRS}")
    exact_letters = int((pred_all == Y.astype(int)).all(axis=1).sum())
    mean_bit = float((pred_all == Y.astype(int)).mean())
    print(f"\n전체: exact letters={exact_letters}/{N_PAIRS}, bit_acc={mean_bit:.3f}")

    # recall 비트맵 그리드 (input -> target -> pred)
    plot_recall_grid(pred_all)

    # 픽셀별 정확도 5x3 히트맵
    fig, ax = plt.subplots(figsize=(4.5, 5.2))
    grid = per_pixel_acc.reshape(5, 3)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=N_PAIRS)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{grid[r,c]}/{N_PAIRS}", ha="center", va="center", fontsize=10,
                    color="black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Per-pixel accuracy (single-layer temporal STDP, {N_PAIRS} pairs)\n"
                 f"exact letters {exact_letters}/{N_PAIRS}, bit {mean_bit:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(RESULTS / "sweep_per_pixel_acc.png", dpi=120)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if SWEEP_ALL:
        run_sweep_all()                              # recall 그리드 포함
    else:
        run_single(OUT_NEURON)
        if RECALL_GRID:                              # 단일 뉴런 실행에도 같이 첨부
            print("\nrecall 그리드용으로 15픽셀 전부 학습 중...")
            _, pred_all = train_all_pixels()
            n_exact = int((pred_all == Y.astype(int)).all(axis=1).sum())
            bit = float((pred_all == Y.astype(int)).mean())
            print(f"전체: exact letters={n_exact}/{N_PAIRS}, bit_acc={bit:.3f}")
            plot_recall_grid(pred_all)
    print(f"\n그림 저장: {RESULTS}")
    plt.show()
