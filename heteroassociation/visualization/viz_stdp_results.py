"""STDP 결과 시각화 — 재사용 가능한 플롯 함수 모음 + standalone 실행.

플롯 함수(plot_recall / plot_weights / plot_mem_matrix / plot_mem_bitmap)는
sweep_stdp_t2.py 등에서 import 해서 케이스별로 호출한다.

standalone 실행 시: 지정(또는 최신) run 하나를 그려서 그 run 폴더에 저장.
    python visualization/viz_stdp_results.py                        # 최신(results/stdp_t2_inhibition)
    python visualization/viz_stdp_results.py <run_name> --runs-dir results/stdp_t2_pm
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt          # 백엔드: import 하는 쪽에서 필요시 Agg 설정
import matplotlib.colors as mcolors

import binary_character as data

DIGIT_IMAGES, _ = data.make_digit_dataset()


# ---------------------------------------------------------------------------
# 재사용 플롯 함수 (out_path 로 저장 위치를 받음)
# ---------------------------------------------------------------------------
def plot_recall(metrics, out_path, run_name, show=False):
    """input(숫자) / target(알파벳) / pred 5x3 그리드."""
    rows = metrics["pair_rows"]
    n = len(rows)
    fig, axes = plt.subplots(3, n, figsize=(1.4 * n, 4.6))
    if n == 1:
        axes = axes.reshape(3, 1)
    for k, row in enumerate(rows):
        dig = int(row["pair"].split("->")[0])
        axes[0, k].imshow(DIGIT_IMAGES[dig], cmap="gray_r", vmin=0, vmax=1)
        axes[1, k].imshow(np.array(row["target"]), cmap="gray_r", vmin=0, vmax=1)
        axes[2, k].imshow(np.array(row["pred"]), cmap="gray_r", vmin=0, vmax=1)
        ok = row["exact"]
        axes[0, k].set_title(row["pair"], fontsize=9, color=("green" if ok else "red"))
        axes[2, k].set_xlabel(f"{row['bit_acc']:.2f}", fontsize=8, color=("green" if ok else "red"))
        for r in range(3):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=10)
    axes[1, 0].set_ylabel("target", fontsize=10)
    axes[2, 0].set_ylabel("pred", fontsize=10)
    fig.suptitle(f"{run_name}\nexact={metrics['exact_count']}/{n}  "
                 f"bit_acc={metrics['mean_bit_acc']:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    if not show:
        plt.close(fig)


def plot_weights(weights, out_path, run_name, show=False, fmt=None):
    """학습된 가중치 히트맵. 셀이 적으면(<=225) 수치도 표기.

    fmt: 셀 값 포맷 문자열(예: "{:.3f}"). None이면 값 크기에 따라 자동 선택
         (|w|max >= 25 → 정수 "{:.0f}", 아니면 소수 3째자리 "{:.3f}").
    """
    w = np.asarray(weights)
    n_pre, n_post = w.shape
    annotate = n_pre * n_post <= 15 * 15
    if fmt is None:
        fmt = "{:.0f}" if np.abs(w).max() >= 25 else "{:.3f}"
    # 0=흰색, 양수=빨강, 음수=파랑: 0 중심 대칭 정규화(-M~+M) + RdBu_r
    m = float(np.abs(w).max()) or 1e-9
    norm = mcolors.Normalize(vmin=-m, vmax=m)
    fig, ax = plt.subplots(figsize=((6.5, 6) if annotate else (5.5, 5)))
    im = ax.imshow(w, cmap="RdBu_r", norm=norm, aspect="auto")
    if annotate:
        for i in range(n_pre):
            for j in range(n_post):
                nv = norm(w[i, j])   # 0..1, 0.5=흰색(=0)
                tc = "white" if abs(nv - 0.5) > 0.32 else "black"   # 짙은 셀만 흰 글씨
                ax.text(j, i, fmt.format(w[i, j]), ha="center", va="center", fontsize=6.5,
                        color=tc)
        ax.set_xticks(range(n_post)); ax.set_yticks(range(n_pre)); ax.tick_params(labelsize=7)
    ax.set_title(f"{run_name}\nweights min={w.min():.1f} max={w.max():.1f} mean={w.mean():.1f}",
                 fontsize=10)
    ax.set_xlabel("hidden neuron (letter pixel)")
    ax.set_ylabel("CS input (digit pixel)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    if not show:
        plt.close(fig)


def _mem_norm(metrics):
    rows = metrics["pair_rows"]
    threshold = float(metrics.get("neuron_threshold", 1.0))
    mem = np.array([np.array(r["mem_peak"]).reshape(-1) for r in rows])   # (n,15)
    vmin = min(float(mem.min()), threshold - 1e-6)
    vmax = max(float(mem.max()), threshold + 1e-6)
    return threshold, mcolors.TwoSlopeNorm(vmin=vmin, vcenter=threshold, vmax=vmax)


def plot_mem_matrix(metrics, out_path, run_name, show=False):
    """막전위 매트릭스: 행=쌍, 열=15픽셀. mem_peak 있을 때만."""
    rows = metrics["pair_rows"]
    if not all("mem_peak" in r for r in rows):
        return False
    n = len(rows)
    threshold, norm = _mem_norm(metrics)
    mem = np.array([np.array(r["mem_peak"]).reshape(-1) for r in rows])
    tgt = np.array([np.array(r["target"]).reshape(-1) for r in rows])

    fig, (ax, axt) = plt.subplots(2, 1, figsize=(11, 1.0 * n + 2.5),
                                  gridspec_kw={"height_ratios": [1, 1]})
    # 윗줄: 막전위 매트릭스 + 값
    im = ax.imshow(mem, cmap="RdBu_r", norm=norm, aspect="auto")
    im.get_cursor_data = lambda event: None
    ax.set_xticks(range(15)); ax.set_xticklabels(range(15), fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels([r["pair"] for r in rows], fontsize=9)
    ax.set_ylabel("membrane\n(pair)")
    for i in range(n):
        for j in range(15):
            nv = norm(mem[i, j]); tc = "white" if (nv < .22 or nv > .78) else "black"
            ax.text(j, i, f"{mem[i, j]:.2f}", ha="center", va="center", fontsize=6, color=tc)

    # 아랫줄: 정답 target 매트릭스 (흑백) — 위와 폭 동일
    axt.imshow(tgt, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
    axt.set_xticks(range(15)); axt.set_xticklabels(range(15), fontsize=8)
    axt.set_yticks(range(n)); axt.set_yticklabels([r["pair"] for r in rows], fontsize=9)
    axt.set_xlabel("hidden neuron (letter pixel 0-14)")
    axt.set_ylabel("target\n(answer)")

    ax.set_title(f"{run_name}  (threshold={threshold:.2f})\n"
                 "top: membrane (red=fire, blue=silent)   bottom: target (answer)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return True


def plot_mem_bitmap(metrics, out_path, run_name, show=False):
    """페어별 출력층 peak 막전위(윗줄, 값 표기)와 정답 target(아랫줄) 5x3 비트맵."""
    rows = metrics["pair_rows"]
    if not all("mem_peak" in r for r in rows):
        return False
    n = len(rows)
    threshold, norm = _mem_norm(metrics)
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 4.2))
    if n == 1:
        axes = axes.reshape(2, 1)
    for k, r in enumerate(rows):
        mem = np.array(r["mem_peak"]); tgt = np.array(r["target"])
        # 윗줄: 막전위 히트맵 + 값
        ax = axes[0, k]
        im = ax.imshow(mem, cmap="RdBu_r", norm=norm)
        im.get_cursor_data = lambda event: None
        for i in range(5):
            for j in range(3):
                nv = norm(mem[i, j]); tc = "white" if (nv < .22 or nv > .78) else "black"
                ax.text(j, i, f"{mem[i, j]:.2f}", ha="center", va="center", fontsize=6, color=tc)
        ax.set_title(r["pair"], fontsize=9, color=("green" if r["exact"] else "red"))
        ax.set_xticks([]); ax.set_yticks([])
        # 아랫줄: 정답 target 비트맵
        axes[1, k].imshow(tgt, cmap="gray_r", vmin=0, vmax=1)
        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    axes[0, 0].set_ylabel("membrane", fontsize=10)
    axes[1, 0].set_ylabel("target", fontsize=10)
    fig.suptitle(f"{run_name}  —  output-layer peak membrane (threshold={threshold:.2f})\n"
                 "top: red=above thr (fire), blue=below   |   bottom: target (answer)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return True


def visualize_run(run_dir: Path, out_dir: Path = None, prefix: str = ""):
    """run_dir 의 metrics.json/weights.npy 를 읽어 out_dir 에 모든 그림 저장."""
    out_dir = out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    name = metrics.get("run_name", run_dir.name)
    plot_recall(metrics, out_dir / f"{prefix}recall.png", name)
    if (run_dir / "weights.npy").exists():
        plot_weights(np.load(run_dir / "weights.npy"), out_dir / f"{prefix}weights.png", name)
    plot_mem_matrix(metrics, out_dir / f"{prefix}mem_matrix.png", name)
    plot_mem_bitmap(metrics, out_dir / f"{prefix}mem_bitmap.png", name)
    return metrics


# ---------------------------------------------------------------------------
# standalone
# ---------------------------------------------------------------------------
def _pick_run(runs_dir: Path, run_name):
    if run_name:
        d = runs_dir / run_name
        if not d.is_dir():
            sys.exit(f"run not found: {d}")
        return d
    subs = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    if not subs:
        sys.exit(f"no runs under {runs_dir}")
    return max(subs, key=lambda p: p.stat().st_mtime)


def main():
    p = argparse.ArgumentParser(description="Visualize one STDP run.")
    p.add_argument("run_name", nargs="?", default=None)
    p.add_argument("--runs-dir", default="results/stdp_t2_inhibition")
    cli = p.parse_args()
    runs_dir = Path(cli.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = ROOT / runs_dir
    run_dir = _pick_run(runs_dir, cli.run_name)
    m = visualize_run(run_dir)
    print(f"run: {run_dir.name}  exact={m['exact_count']}  bit_acc={m['mean_bit_acc']:.3f}")
    print(f"saved into: {run_dir}")


if __name__ == "__main__":
    main()
