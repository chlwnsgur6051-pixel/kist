"""가중치 = 입력 픽셀 스파이크의 '발화 시각(간격)' 인 temporal 실험.

아이디어 (요청안)
----------------
- 스파이크는 '1'만 남긴다: 활성 픽셀(비트=1)만 발화. 0 픽셀은 발화 없음.
- 가중치를 진폭이 아니라 **각 입력 픽셀의 발화 시각 tau_i** 로 둔다 (=스파이크 간격).
- 학습: 출력 타깃=1 이면 그 패턴 활성 픽셀들의 간격을 **좁힌다(모은다)**,
        타깃=0 이면 **벌린다**.
- 판정: **출력 뉴런의 발화 여부**. 활성 픽셀만 자기 tau_i 에 스파이크를 쏘고,
        각 스파이크는 AMP/활성수 만큼 막전위에 실린다(페어 간 스케일 정규화).
        OUT_TAU 로 누설적분한 봉우리가 OUT_THR 을 넘으면 발화 -> 1, 침묵 -> 0.
        스파이크가 뭉칠수록 봉우리가 높아지므로 coincidence 검출이 된다.
        (spread(max-min)는 더 이상 판정에 쓰지 않고 진단용으로만 남김)

학습 규칙(패턴 p, 활성 집합 A_p, 평균 mu):
    타깃=1:  tau_i += eta * (mu - tau_i)   (평균으로 당김 -> 간격↓)
    타깃=0:  tau_i -= eta * (mu - tau_i)   (평균에서 밀어냄 -> 간격↑)

주의: tau_i(가중치)는 픽셀마다 하나. 여러 패턴에 공유되므로, 어떤 픽셀이
      타깃1·타깃0 패턴 모두에 쓰이면 요구가 충돌한다(=이 방식의 한계 실험 포인트).

실행: python "템퍼럴_stdp/temporal_interval.py"
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

# 학습 로직에서 평균과의 거리를 spread 로 정규화할지 (False = 이전버전: step = eta*dev)
# 주의: 이건 '학습 로직'에만 영향. 페어별 시뮬(output_membrane 활성수 나눗셈)은 그대로.
NORMALIZE = False
TAG = "_normalize" if NORMALIZE else ""

EVO = RESULTS / f"evolution_interval{TAG}"
EVO.mkdir(parents=True, exist_ok=True)
PER_OUT = RESULTS / f"per_output_interval{TAG}"   # 15 출력별 상세 그림
PER_OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(0)

# ===== 설정 =====
ALL_OUTPUTS = False     # True: 출력 15픽셀 전부 학습 + 전체 시각화 + recall 그리드
N_PAIRS = 10            # 사용할 페어 수 (앞에서부터 N개 숫자). 최대 10.
OUT_NEURON = 7          # ALL_OUTPUTS=False 일 때만 쓰는 단일 출력. (evolution 프레임도 이 뉴런만)
T_MAX = 10.0
T_CENTER = 5.0          # 발화 시각 중심
INIT_SPREAD = 3.0       # 초기 tau 랜덤 분산 폭
# 3-factor(오차 게이팅) on/off. False = HS 기준 그대로 (타깃1 가까워 / 타깃0 멀어져)
THREE_FACTOR = False
ETA_3F = 0.05
ETA_CLUSTER = 0.005      # (THREE_FACTOR=False 일 때) 타깃=1 모으는 속도
ETA_SPREAD = 0.005       # (THREE_FACTOR=False 일 때) 타깃=0 벌리는 속도
EPOCHS = 300
EVO_EVERY = 5
SPREAD_THR = 2.0        # (판정에는 미사용) spread 진단/시각화용 참고선
# 출력 coincidence 뉴런 — 여기가 실제 판정기: 짧은 tau -> 뭉치면 높은 봉우리
DT_SIM = 0.05
OUT_TAU = 0.5
NORM_BY_ACTIVE = True  # 스파이크 크기를 활성 픽셀 수로 나눌지 (False = 끔, raw 합산)
# OUT_THR: NORM_BY_ACTIVE=True 면 ~[0,AMP] 이므로 0.5,  False(raw) 면 3.0 정도(개수 기준)
OUT_THR = 0.4
AMP = 1.0
# ================

digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()
X01 = digit_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]   # 0/1 입력 (앞 N개)
Y01 = letter_images.reshape(10, 15).astype(np.float64)[:N_PAIRS]  # 0/1 타깃
digit_names = [str(i) for i in range(N_PAIRS)]


def _grid_fig(n, figsize=(15, 6.4)):
    """페어 수 n 에 맞춘 유연한 서브플롯 그리드. (fig, 사용할 axes 리스트) 반환."""
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for k in range(n, len(flat)):
        flat[k].axis("off")                       # 남는 칸 숨김
    return fig, flat[:n]


def spread_of(tau, active):
    """활성 픽셀 발화 시각의 퍼짐(max-min)."""
    if len(active) <= 1:
        return 0.0
    t = tau[active]
    return float(t.max() - t.min())


def predict(tau):
    """판정 = 출력 뉴런의 발화 여부.

    활성 픽셀(=1)만 자기 tau_i 에 스파이크를 쏘고, 각 스파이크는
    AMP/활성수 만큼 막전위에 실린다(페어 간 스케일 정규화).
    OUT_TAU 로 누설적분한 봉우리가 OUT_THR 을 넘으면 발화 -> 1, 못 넘으면 침묵 -> 0.
    반환: (예측, 봉우리 높이).  spread(max-min)는 진단용으로 spread_of() 를 따로 씀."""
    preds, peaks = [], []
    for i in range(N_PAIRS):
        A = np.where(X01[i] > 0.5)[0]
        pk = float(output_membrane(tau, A).max()) if len(A) > 0 else 0.0
        peaks.append(pk)
        preds.append(1 if pk >= OUT_THR else 0)
    return np.array(preds), np.array(peaks)


def output_membrane(tau, active):
    """coincidence 출력 뉴런의 시간별 막전위. 활성 스파이크가 tau_i 에 도착,
    짧은 OUT_TAU 로 적분 -> 뭉치면 높은 봉우리.
    스파이크 크기를 활성 픽셀 수로 나눔 -> 페어 간 스케일 정규화
    (전부 뭉치면 봉우리≈AMP, 흩어지면 ≈AMP/n_active)."""
    n = int(round(T_MAX / DT_SIM))
    decay = np.exp(-DT_SIM / OUT_TAU)
    amp = AMP / max(len(active), 1) if NORM_BY_ACTIVE else AMP   # 활성수 나눗셈 on/off
    I = np.zeros(n)
    for i in active:
        s = int(round(tau[i] / DT_SIM))
        if 0 <= s < n:
            I[s] += amp
    v, trace = 0.0, np.zeros(n)
    for step in range(n):
        v = v * decay + I[step]
        trace[step] = v
    return trace


def peak_of(tau, active):
    """coincidence 막전위의 (봉우리 높이, 봉우리 시각). 판정은 높이로만 함."""
    if len(active) == 0:
        return 0.0, 0.0
    tr = output_membrane(tau, active)
    s = int(np.argmax(tr))
    return float(tr[s]), s * DT_SIM


def plot_frame(tau, tgt, epoch):
    """페어별 활성 픽셀 발화 시각(간격) + 판정 스냅샷을 EVO 에 저장."""
    pred, peaks = predict(tau)
    acc = int((pred == tgt).sum())
    fig, axs = _grid_fig(N_PAIRS)
    for i, axp in enumerate(axs):
        A = np.where(X01[i] > 0.5)[0]
        axp.scatter(tau[A], A, s=26, color="tab:blue", zorder=3)   # 활성 픽셀 스파이크(1만)
        if len(A) > 0:                                             # spread 구간 표시
            axp.axvspan(tau[A].min(), tau[A].max(), color="tab:orange", alpha=0.12)
            axp.axvline(tau[A].mean(), color="tab:orange", ls="-", lw=0.8, alpha=0.6)
        pk, pt = peak_of(tau, A)
        fired = pk >= OUT_THR
        axp.axvline(pt, color=("tab:red" if fired else "tab:gray"),   # 막전위 봉우리 시각
                    ls="-", lw=1.4, alpha=0.75, zorder=2)
        ok = "O" if pred[i] == tgt[i] else "X"
        axp.set_title(f"digit {i} tgt={tgt[i]} pred={pred[i]} {ok}\n"
                      f"peak={peaks[i]:.2f} {'FIRE' if fired else 'sil'}",
                      fontsize=8, color=("green" if pred[i] == tgt[i] else "red"))
        axp.set_xlim(0, T_MAX); axp.set_ylim(-1, 15); axp.set_yticks([0, 5, 10, 14])
    fig.suptitle(f"epoch {epoch:04d}  acc {acc}/{N_PAIRS}  (out {OUT_NEURON})  "
                 f"weight=spike time, peak>={OUT_THR}->fire->1", fontsize=12)
    fig.supxlabel("spike time (= weight / interval)"); fig.supylabel("input pixel (0-14)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(EVO / f"interval{TAG}_n{OUT_NEURON}_ep{epoch:04d}.png", dpi=110)
    plt.close(fig)


def train(out_neuron, save_evo=False):
    tgt = Y01[:, out_neuron].astype(int)
    tau = T_CENTER + (np.random.rand(15) - 0.5) * 2 * INIT_SPREAD   # 초기 발화시각
    tau = np.clip(tau, 0, T_MAX)
    acc_hist = []
    tau_hist = np.zeros((EPOCHS + 1, 15)); tau_hist[0] = tau
    if save_evo:
        plot_frame(tau, tgt, 0)
    for ep in range(EPOCHS):
        for p in np.random.permutation(N_PAIRS):
            A = np.where(X01[p] > 0.5)[0]
            if len(A) == 0:
                continue
            mu = tau[A].mean()
            dev = tau[A] - mu                       # 평균과의 거리 (=eligibility)
            if THREE_FACTOR:
                # 3-factor: M(오차) 로 게이팅. 맞으면 skip, 틀리면 HS 방향으로 교정.
                # 판정 기준은 predict() 와 동일하게 '발화 여부'.
                pred_p = 1 if output_membrane(tau, A).max() >= OUT_THR else 0
                M = tgt[p] - pred_p                 # +1(1인데 0), -1(0인데 1), 0(맞음)
                if M == 0:
                    continue                        # 정답 -> 업데이트 안 함
                if NORMALIZE:
                    scale = max(float(dev.max() - dev.min()) if len(A) > 1 else 1.0, 1e-6)
                    tau[A] -= ETA_3F * M * dev / scale
                else:
                    tau[A] -= ETA_3F * M * dev      # M=+1 가까워 / M=-1 멀어져
                tau = np.clip(tau, 0, T_MAX)
                continue
            eta = ETA_CLUSTER if tgt[p] == 1 else ETA_SPREAD   # (구버전) 타깃 기반
            if NORMALIZE:
                scale = float(dev.max() - dev.min()) if len(A) > 1 else 1.0
                scale = max(scale, 1e-6)
                step = eta * dev / scale
            else:
                step = eta * dev
            if tgt[p] == 1:
                tau[A] -= step
            else:
                tau[A] += step
            tau = np.clip(tau, 0, T_MAX)
        tau_hist[ep + 1] = tau
        pred, _ = predict(tau)
        acc_hist.append(int((pred == tgt).sum()))
        if save_evo and (ep + 1) % EVO_EVERY == 0:
            plot_frame(tau, tgt, ep + 1)
    return tau, tgt, acc_hist, tau_hist


def plot_output_figs(j, tau, tgt, pred, peaks, tau_hist):
    """출력 픽셀 j 하나에 대한 상세 그림 4장을 PER_OUT 에 저장(닫음)."""
    acc = int((pred == tgt).sum())
    # 1) 입력 스파이크 타이밍 + 판정 근거(봉우리 시각/높이)
    fig1, axs = _grid_fig(N_PAIRS, figsize=(15, 6.2))
    for i, axp in enumerate(axs):
        A = np.where(X01[i] > 0.5)[0]
        axp.scatter(tau[A], A, s=26, color="tab:blue", zorder=3)   # 활성 픽셀만 발화
        if len(A) > 0:
            axp.axvspan(tau[A].min(), tau[A].max(), color="tab:orange", alpha=0.12)
        pk, pt = peak_of(tau, A)
        fired = pk >= OUT_THR
        axp.axvline(pt, color=("tab:red" if fired else "tab:gray"),   # 막전위 봉우리 시각
                    ls="-", lw=1.4, alpha=0.75, zorder=2)
        ok = "O" if pred[i] == tgt[i] else "X"
        axp.set_title(f"digit {i} tgt={tgt[i]} pred={pred[i]} {ok}\n"
                      f"peak={pk:.2f}@t={pt:.1f} {'FIRE' if fired else 'sil'}", fontsize=8,
                      color=("green" if pred[i] == tgt[i] else "red"))
        axp.set_xlim(0, T_MAX); axp.set_ylim(-1, 15); axp.set_yticks([0, 5, 10, 14])
    fig1.suptitle(f"out {j}: input spike timing (weight=spike time)   "
                  f"vertical line = membrane peak,  peak>={OUT_THR} -> fire -> 1", fontsize=12)
    fig1.supxlabel("spike time"); fig1.supylabel("input pixel")
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(PER_OUT / f"interval{TAG}_n{j:02d}_1_timing.png", dpi=110)
    fig1.savefig(RESULTS / f"interval{TAG}_n{j:02d}_timing.png", dpi=120)   # 메인에도 저장
    plt.close(fig1)
    # 2) 멤브레인 봉우리 막대 (= 판정 근거). OUT_THR 넘으면 발화 -> 1
    fig2, ax = plt.subplots(figsize=(9, 3.4))
    colors = ["tab:red" if tgt[i] == 1 else "tab:blue" for i in range(N_PAIRS)]
    ax.bar(np.arange(N_PAIRS), peaks, color=colors, alpha=0.85)
    ax.axhline(OUT_THR, color="k", ls="--", lw=1, label=f"OUT_THR={OUT_THR}")
    for i in range(N_PAIRS):
        ok = pred[i] == tgt[i]
        ax.text(i, peaks[i] + 0.02, "O" if ok else "X", ha="center",
                color=("green" if ok else "red"), fontweight="bold")
    ax.set_xticks(np.arange(N_PAIRS)); ax.set_xticklabels(digit_names)
    ax.set_ylim(0, max(AMP * 1.15, float(np.max(peaks)) * 1.15))
    ax.set_ylabel("membrane peak"); ax.legend(fontsize=8)
    ax.set_title(f"out {j}: output membrane peak per digit (acc {acc}/{N_PAIRS})")
    fig2.tight_layout(); fig2.savefig(PER_OUT / f"interval{TAG}_n{j:02d}_2_peak.png", dpi=110)
    plt.close(fig2)
    # 3) tau 궤적
    fig3, ax = plt.subplots(figsize=(9, 4.5))
    eps = np.arange(EPOCHS + 1); c2 = plt.cm.viridis(np.linspace(0, 1, 15))
    for k in range(15):
        ax.plot(eps, tau_hist[:, k], lw=1.0, color=c2[k])
    ax.set_xlabel("epoch"); ax.set_ylabel("tau"); ax.set_title(f"out {j}: 15 tau over epochs")
    fig3.tight_layout(); fig3.savefig(PER_OUT / f"interval{TAG}_n{j:02d}_3_tau_time.png", dpi=110)
    plt.close(fig3)
    # 4) 출력 멤브레인
    t_axis = np.arange(int(round(T_MAX / DT_SIM))) * DT_SIM
    fig4, axs = _grid_fig(N_PAIRS)
    for i, axp in enumerate(axs):
        A = np.where(X01[i] > 0.5)[0]
        tr = output_membrane(tau, A)
        axp.plot(t_axis, tr, lw=1.4, color=("tab:red" if tgt[i] == 1 else "tab:blue"))
        axp.axhline(OUT_THR, color="k", ls=":", lw=1)
        fired = tr.max() >= OUT_THR
        axp.set_title(f"d{i} tgt={tgt[i]} {'FIRE' if fired else 'sil'}", fontsize=8,
                      color=("green" if int(fired) == tgt[i] else "red"))
        axp.set_xlim(0, T_MAX); axp.set_ylim(-0.1, max(AMP * 1.1, float(tr.max()) * 1.15))
    fig4.suptitle(f"out {j}: output membrane (coincidence)", fontsize=12)
    fig4.supxlabel("time"); fig4.supylabel("v")
    fig4.tight_layout(); fig4.savefig(PER_OUT / f"interval{TAG}_n{j:02d}_4_out_membrane.png", dpi=110)
    plt.close(fig4)


def plot_recall_grid(pred_all):
    """input(숫자) / target(글자) / pred(글자) 5x3 비트맵 그리드."""
    fig, axes = plt.subplots(3, N_PAIRS, figsize=(1.45 * N_PAIRS, 4.6), squeeze=False)
    for k in range(N_PAIRS):
        inp = X01[k].reshape(5, 3); tg = Y01[k].reshape(5, 3); pr = pred_all[k].reshape(5, 3)
        axes[0][k].imshow(inp, cmap="gray_r", vmin=0, vmax=1)
        axes[1][k].imshow(tg, cmap="gray_r", vmin=0, vmax=1)
        axes[2][k].imshow(pr, cmap="gray_r", vmin=0, vmax=1)
        exact = np.array_equal(pr, tg); bit = float((pr == tg).mean())
        axes[0][k].set_title(f"{k}->{chr(65 + k)}", fontsize=9,
                             color=("green" if exact else "red"))
        axes[2][k].set_xlabel(f"{bit:.2f}", fontsize=8, color=("green" if exact else "red"))
        for r in range(3):
            axes[r][k].set_xticks([]); axes[r][k].set_yticks([])
    axes[0][0].set_ylabel("input"); axes[1][0].set_ylabel("target"); axes[2][0].set_ylabel("pred")
    n_exact = sum(np.array_equal(pred_all[k].reshape(5, 3), Y01[k].reshape(5, 3))
                  for k in range(N_PAIRS))
    fig.suptitle(f"Recall: input -> target -> pred   (exact {n_exact}/{N_PAIRS})", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / f"interval{TAG}_recall_grid.png", dpi=120)


def plot_per_pixel_acc(per_pixel, n_exact, bit):
    """출력 픽셀별 정확도 5x3 히트맵."""
    fig, ax = plt.subplots(figsize=(4.5, 5.2))
    grid = per_pixel.reshape(5, 3)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=N_PAIRS)
    for r in range(5):
        for c in range(3):
            ax.text(c, r, f"{grid[r, c]}/{N_PAIRS}", ha="center", va="center", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Per-pixel accuracy (interval)\nexact letters {n_exact}/{N_PAIRS}, bit {bit:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / f"interval{TAG}_per_pixel_acc.png", dpi=120)


def plot_membrane_show(tau, tgt, pred, peaks, out_neuron):
    """페어별 출력 멤브레인 (coincidence). 닫지 않음 -> plt.show 로 화면 표시."""
    t_axis = np.arange(int(round(T_MAX / DT_SIM))) * DT_SIM
    fig, axs = _grid_fig(N_PAIRS)
    for i, axp in enumerate(axs):
        A = np.where(X01[i] > 0.5)[0]
        tr = output_membrane(tau, A)
        axp.plot(t_axis, tr, lw=1.5, color=("tab:red" if tgt[i] == 1 else "tab:blue"))
        axp.axhline(OUT_THR, color="k", ls=":", lw=1)
        axp.scatter(tau[A], np.full_like(tau[A], -0.05), s=12, marker="|", color="gray", zorder=3)
        fired = tr.max() >= OUT_THR
        axp.set_title(f"d{i} tgt={tgt[i]} pred={pred[i]} {'FIRE' if fired else 'sil'}", fontsize=8,
                      color=("green" if pred[i] == tgt[i] else "red"))
        axp.set_xlim(0, T_MAX); axp.set_ylim(-0.1, max(AMP * 1.1, float(tr.max()) * 1.15))
    fig.suptitle(f"Output membrane per pair (out {out_neuron}, acc {int((pred==tgt).sum())}/{N_PAIRS})",
                 fontsize=12)
    fig.supxlabel("time"); fig.supylabel("output membrane v")
    fig.tight_layout()
    fig.savefig(RESULTS / f"interval{TAG}_n{out_neuron}_membrane.png", dpi=120)   # 안 닫음


def main():
    if not ALL_OUTPUTS:
        # 단일 출력 (기존 상세 저장 + 막전위 화면표시)
        tau, tgt, _, tau_hist = train(OUT_NEURON, save_evo=True)
        pred, peaks = predict(tau)
        print(f"out {OUT_NEURON}: acc {int((pred==tgt).sum())}/{N_PAIRS}")
        plot_output_figs(OUT_NEURON, tau, tgt, pred, peaks, tau_hist)     # 저장(닫음)
        plot_membrane_show(tau, tgt, pred, peaks, OUT_NEURON)              # 화면표시용
        plt.show()
        return

    # 15 출력 전부 학습
    pred_all = np.zeros((N_PAIRS, 15), dtype=int)
    per_pixel = np.zeros(15, dtype=int)
    rep = None                                             # OUT_NEURON 대표(막전위 표시용)
    print("=== interval rule, 출력 15픽셀 전부 학습 ===")
    for j in range(15):
        tau, tgt, _, tau_hist = train(j, save_evo=(j == OUT_NEURON))   # evo는 OUT_NEURON만
        pred, peaks = predict(tau)
        pred_all[:, j] = pred
        per_pixel[j] = int((pred == tgt).sum())
        plot_output_figs(j, tau, tgt, pred, peaks, tau_hist)
        if j == OUT_NEURON:
            rep = (tau.copy(), tgt.copy(), pred.copy(), peaks.copy())
        print(f" out {j:2d}: {per_pixel[j]}/{N_PAIRS}")

    n_exact = sum(np.array_equal(pred_all[k], Y01[k].astype(int)) for k in range(N_PAIRS))
    bit = float((pred_all == Y01.astype(int)).mean())
    print(f"\n전체: exact letters={n_exact}/{N_PAIRS}, bit_acc={bit:.3f}")

    plot_recall_grid(pred_all)
    plot_per_pixel_acc(per_pixel, n_exact, bit)
    if rep is not None:                                   # OUT_NEURON 출력 막전위 화면표시
        plot_membrane_show(rep[0], rep[1], rep[2], rep[3], OUT_NEURON)
    print(f"그림: {RESULTS}  (출력별 상세: {PER_OUT})")
    plt.show()


if __name__ == "__main__":
    main()
