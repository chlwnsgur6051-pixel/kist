"""STDP T2 실행 결과(stdp_t2_inhibition.py 출력)를 시각화.

읽는 것:
  results/stdp_t2_inhibition/<run>/metrics.json   (쌍별 pred/target, 정확도)
  results/stdp_t2_inhibition/<run>/weights.npy    (학습된 15x15 가중치 W)

만드는 것 (visualization/results/):
  stdp_weights.png   학습된 W 히트맵
  stdp_recall.png    쌍별 target vs pred 5x3 그리드 + 정확도

사용법:
  python visualization/viz_stdp_results.py            # 최신 run 자동 선택
  python visualization/viz_stdp_results.py <run_name> # 특정 run 지정

이 스크립트는 visualization/ 안에 있으므로 프로젝트 루트를 import 경로에 추가한다.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt  # 백엔드 자동 선택: GUI 있으면 창 표시, 없으면 저장만
import matplotlib.colors as mcolors

import binary_character as data

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def parse_cli():
    p = argparse.ArgumentParser(description="Visualize an STDP T2 run.")
    p.add_argument("run_name", nargs="?", default=None, help="특정 run 이름 (없으면 최신)")
    p.add_argument("--runs-dir", default="results/stdp_t2_inhibition",
                   help="run 폴더들이 있는 경로 (예: results/stdp_t2_pm)")
    return p.parse_args()


_cli = parse_cli()
RUNS_DIR = Path(_cli.runs_dir)
if not RUNS_DIR.is_absolute():
    RUNS_DIR = ROOT / RUNS_DIR


def pick_run_dir():
    if _cli.run_name:
        d = RUNS_DIR / _cli.run_name
        if not d.is_dir():
            sys.exit(f"run not found: {d}")
        return d
    if not RUNS_DIR.is_dir():
        sys.exit(f"no results dir yet: {RUNS_DIR}\n먼저 학습 스크립트를 실행하세요.")
    subdirs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    if not subdirs:
        sys.exit(f"no runs under {RUNS_DIR}\n먼저 학습 스크립트를 실행하세요.")
    return max(subdirs, key=lambda p: p.stat().st_mtime)   # 최신 run


run_dir = pick_run_dir()
metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
weights = np.load(run_dir / "weights.npy")   # (15, 15)
pair_rows = metrics["pair_rows"]
run_name = metrics.get("run_name", run_dir.name)

# ---------------------------------------------------------------------------
# 1) 학습된 가중치 히트맵
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(weights, cmap="viridis", aspect="equal")
ax.set_title(f"Learned CS->hidden weights  (run: {run_name})\n"
             f"min={weights.min():.1f}  max={weights.max():.1f}  mean={weights.mean():.1f}")
ax.set_xlabel("hidden neuron (letter pixel 0-14)")
ax.set_ylabel("CS input (digit pixel 0-14)")
fig.colorbar(im, ax=ax, label="weight (conductance)")
fig.tight_layout()
out1 = RESULTS / "stdp_weights.png"
fig.savefig(out1, dpi=120, bbox_inches="tight")
print(f"saved: {out1}")

# ---------------------------------------------------------------------------
# 2) 쌍별 회상 결과: input(숫자) / target(알파벳) / pred 5x3 그리드
# ---------------------------------------------------------------------------
n = len(pair_rows)
digit_images, _ = data.make_digit_dataset()   # 입력 숫자 패턴 (10, 5, 3)
fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 5.5))
if n == 1:
    axes = axes.reshape(3, 1)

for k, row in enumerate(pair_rows):
    dig_idx = int(row["pair"].split("->")[0])   # "3->D" -> 3
    inp = digit_images[dig_idx]
    tgt = np.array(row["target"])
    prd = np.array(row["pred"])
    axes[0, k].imshow(inp, cmap="gray_r", vmin=0, vmax=1)
    axes[1, k].imshow(tgt, cmap="gray_r", vmin=0, vmax=1)
    axes[2, k].imshow(prd, cmap="gray_r", vmin=0, vmax=1)
    ok = row["exact"]
    axes[0, k].set_title(row["pair"], fontsize=10,
                         color=("green" if ok else "red"))
    axes[2, k].set_xlabel(f"acc={row['bit_acc']:.2f}", fontsize=9,
                          color=("green" if ok else "red"))
    for r in (0, 1, 2):
        axes[r, k].set_xticks([]); axes[r, k].set_yticks([])

axes[0, 0].set_ylabel("input", fontsize=11)
axes[1, 0].set_ylabel("target", fontsize=11)
axes[2, 0].set_ylabel("pred", fontsize=11)
fig.suptitle(f"STDP T2 recall  —  exact={metrics['exact_count']}/{n}  "
             f"mean_bit_acc={metrics['mean_bit_acc']:.3f}\n"
             f"(green=exact match, red=mismatch)", fontsize=12)
fig.tight_layout()
out2 = RESULTS / "stdp_recall.png"
fig.savefig(out2, dpi=120, bbox_inches="tight")
print(f"saved: {out2}")

# ---------------------------------------------------------------------------
# 3) 막전위 매트릭스: 행=쌍, 열=15 픽셀(은닉 뉴런), 색=peak 막전위
#    이진 픽셀 대신 "발화 판정 직전의 연속값"을 매트릭스로 본다.
#    red=임계 위(발화), blue=임계 아래(침묵), 초록 테두리=target ON(발화해야 함)
# ---------------------------------------------------------------------------
if all("mem_peak" in r for r in pair_rows):
    threshold = float(metrics.get("neuron_threshold", 1.0))
    mem = np.array([np.array(r["mem_peak"]).reshape(-1) for r in pair_rows])   # (n, 15)
    tgt_mat = np.array([np.array(r["target"]).reshape(-1) for r in pair_rows])  # (n, 15)

    # margin = mean(target-on 막전위) - mean(target-off 막전위). 클수록 잘 갈림.
    margins = []
    for i in range(n):
        on = mem[i][tgt_mat[i] >= 0.5]
        off = mem[i][tgt_mat[i] < 0.5]
        margins.append(float(on.mean() - off.mean()) if on.size and off.size else 0.0)
    mean_margin = float(np.mean(margins))

    vmin = min(float(mem.min()), threshold - 1e-6)
    vmax = max(float(mem.max()), threshold + 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=threshold, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11, 0.55 * n + 2))
    im = ax.imshow(mem, cmap="RdBu_r", norm=norm, aspect="auto")
    im.get_cursor_data = lambda event: None  # TwoSlopeNorm 위 마우스 호버 시 matplotlib 버그 회피
    ax.set_xticks(range(15)); ax.set_xticklabels(range(15), fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels([r["pair"] for r in pair_rows], fontsize=9)
    ax.set_xlabel("hidden neuron (letter pixel 0-14)")
    ax.set_ylabel("pair (digit -> letter)")

    for i in range(n):
        for j in range(15):
            nv = norm(mem[i, j])
            tcol = "white" if (nv < 0.22 or nv > 0.78) else "black"
            ax.text(j, i, f"{mem[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color=tcol)
            if tgt_mat[i, j] >= 0.5:   # target=1 인 픽셀(발화해야 함) 강조
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="lime", linewidth=1.6))

    # 행별 margin 을 오른쪽에 표기 (양수=제대로 갈림, 음수=순서 뒤집힘)
    for i in range(n):
        ax.text(15.3, i, f"{margins[i]:+.3f}", ha="left", va="center", fontsize=8,
                color=("green" if margins[i] > 0 else "red"))
    ax.text(15.3, -0.9, "margin", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, 18.0)

    fig.colorbar(im, ax=ax, label="peak membrane potential")
    ax.set_title(f"Peak membrane potential matrix  (run: {run_name}, threshold={threshold:.2f})\n"
                 f"mean margin = {mean_margin:+.4f}   "
                 "(margin = mean[target-ON mem] - mean[target-OFF mem]; 클수록 좋음)\n"
                 "red=above threshold (fired)   blue=below   green box = target ON (should fire)",
                 fontsize=10)
    fig.tight_layout()
    out3 = RESULTS / "stdp_mem_matrix.png"
    fig.savefig(out3, dpi=120, bbox_inches="tight")
    print(f"saved: {out3}")
else:
    print("mem_peak 없음: stdp_t2_inhibition.py 를 다시 실행해 막전위를 기록하세요.")

print(f"\nrun_dir : {run_dir}")
print(f"exact   : {metrics['exact_count']}/{n}")
print(f"bit_acc : {metrics['mean_bit_acc']:.3f}")
print(f"failed  : {metrics.get('failed_pairs', [])}")

plt.show()  # 저장된 두 figure(가중치 히트맵 + 회상 결과)를 창으로 표시
