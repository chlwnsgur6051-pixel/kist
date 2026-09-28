"""
Contrastive optimal weights — REMOVE-BIAS 버전 (편향 b 제거).

동기: STDP 모델은 시냅스 가중치만 학습하고 '뉴런별 편향'이 없다
      (발화 ⟺ W·x ≥ threshold, b 항 없음).
      공정 비교를 위해 contrastive 에서도 편향을 제거한다.

원본(contrastive_optimal_weight.py)과의 차이:
  - phi_aug = [Φ | 1]  →  Φ  (편향 열 제거)
  - weight = pinv(Φ) @ teacher   (b 없음)
  - 회상: mem = Φ @ W  (+ b 제거)

나머지(교사 ±1, 특징 확장, 클리핑, LIF 임계, 지표, 시각화)는 원본과 동일.
결과/시각화는 STDP 결과와 같은 형식(run 폴더 + viz 함수)으로 저장.

pair 수 실험: 상단 N_PAIRS_SWEEP 리스트를 편집하면 각 pair 수로 순차 실행하고
마지막에 요약표를 출력한다. CLI 로 하나만 돌리려면 --n-pairs N.
Humans: edit PARAMETER_SET / N_PAIRS_SWEEP. Agents: use argparse.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import binary_character as data

# 시각화 백엔드: --no-show 면 저장만(Agg), 아니면 창도 표시
SHOW = "--no-show" not in sys.argv
import matplotlib
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "visualization"))
import viz_stdp_results as viz

# ===== 여기서 pair 수 실험 조절 =====
# 여러 값을 넣으면 각 pair 수로 순차 실행하고 마지막에 요약표를 출력한다.
# 예: [1, 2, 3, 5, 10]  /  단일 실험이면 [3] 처럼 하나만.
N_PAIRS_SWEEP = [10]
# ===================================

# ---------------------------------------------------------------------------
# PARAMETER SET — human-editable
# ---------------------------------------------------------------------------
PARAMETER_SET = {
    "threshold": 0.8,
    "tau": 20e-3,
    "dt": 1e-3,
    "basis": "linear",  # "linear" | "quadratic"
    "target_one_current": 1.0,
    "target_zero_current": -1.0,
    "weight_clip": 20.0,
    "recall_steps": 1,
    "count_threshold": 1,
    "device": "cpu",
    "output_dir": "results/contrastive_optimal_weight",
    "run_name": "contrastive_removeB",
}


def parse_args():
    p = argparse.ArgumentParser(description="Contrastive optimal weight, NO bias (STDP-fair).")
    p.add_argument("--basis", choices=["linear", "quadratic"], default=PARAMETER_SET["basis"])
    p.add_argument("--n-pairs", type=int, default=None,
                   help="이 값을 주면 N_PAIRS_SWEEP 대신 이 pair 수 하나만 실행")
    p.add_argument("--target-one-current", type=float, default=PARAMETER_SET["target_one_current"])
    p.add_argument("--target-zero-current", type=float, default=PARAMETER_SET["target_zero_current"])
    p.add_argument("--weight-clip", type=float, default=PARAMETER_SET["weight_clip"])
    p.add_argument("--recall-steps", type=int, default=PARAMETER_SET["recall_steps"])
    p.add_argument("--count-threshold", type=int, default=PARAMETER_SET["count_threshold"])
    p.add_argument("--device", default=PARAMETER_SET["device"])
    p.add_argument("--output-dir", default=PARAMETER_SET["output_dir"])
    p.add_argument("--run-name", default=PARAMETER_SET["run_name"])
    p.add_argument("--no-show", action="store_true", help="창 표시 없이 저장만")
    return p.parse_args()


def load_patterns(device, n_pairs=10):
    """앞에서부터 n_pairs 개만 사용 (STDP 쪽 load_patterns 와 동일한 슬라이싱)."""
    digit_images, _ = data.make_digit_dataset()
    letter_images, _ = data.make_letter_dataset()
    cs = torch.tensor(digit_images.reshape(10, -1)[:n_pairs], dtype=torch.float32, device=device)
    target = torch.tensor(letter_images.reshape(10, -1)[:n_pairs], dtype=torch.float32, device=device)
    return cs, target


def expand_features(cs, basis):
    if basis == "linear":
        return cs
    features = [cs]
    pair_features = []
    n_features = cs.shape[1]
    for i in range(n_features):
        for j in range(i, n_features):
            pair_features.append((cs[:, i] * cs[:, j]).unsqueeze(1))
    features.append(torch.cat(pair_features, dim=1))
    return torch.cat(features, dim=1)


def train_contrastive_association(cs, target, basis, target_one_current, target_zero_current):
    """편향 없이: W = pinv(Φ) @ teacher  (Φ 에 1 열을 붙이지 않음)."""
    phi = expand_features(cs, basis)
    teacher_current = target * target_one_current + (1.0 - target) * target_zero_current
    weight = torch.linalg.pinv(phi) @ teacher_current      # ← [Φ|1] 아니라 Φ 만
    return phi, weight


def lif_recall(phi, weight, threshold, tau, dt, recall_steps, count_threshold):
    """편향 없이: mem = β·mem + Φ @ W  (+ b 제거)."""
    beta = math.exp(-dt / tau)
    mem = torch.zeros(phi.shape[0], weight.shape[1], dtype=phi.dtype, device=phi.device)
    counts = torch.zeros_like(mem)
    for _ in range(recall_steps):
        mem = beta * mem + phi @ weight
        spikes = (mem >= threshold).to(phi.dtype)
        counts += spikes
        mem = mem * (1.0 - spikes)
    return (counts >= count_threshold).to(phi.dtype)


def compute_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    target_np = target.detach().cpu().numpy().astype(np.float32)
    pair_rows = []
    bit_accs, mses, cosines = [], [], []
    for idx in range(pred_np.shape[0]):
        p, t = pred_np[idx], target_np[idx]
        bit_acc = float((p == t).mean())
        mse = float(np.mean((p - t) ** 2))
        denom = float(np.linalg.norm(p) * np.linalg.norm(t) + 1e-8)
        cosine = float(np.dot(p, t) / denom)
        exact = bool(np.array_equal(p, t))
        bit_accs.append(bit_acc)
        mses.append(mse)
        cosines.append(cosine)
        pair_rows.append(
            {
                "pair": f"{idx}->{chr(ord('A') + idx)}",
                "exact": exact,
                "bit_acc": bit_acc,
                "mse": mse,
                "cosine": cosine,
                "pred_ones": int(p.sum()),
                "target_ones": int(t.sum()),
                "pred": p.reshape(5, 3).astype(int).tolist(),
                "target": t.reshape(5, 3).astype(int).tolist(),
            }
        )
    return {
        "exact_count": int(sum(row["exact"] for row in pair_rows)),
        "mean_bit_acc": float(np.mean(bit_accs)),
        "mean_mse": float(np.mean(mses)),
        "mean_cosine": float(np.mean(cosines)),
        "pair_rows": pair_rows,
    }


def run_one(args, device, n_pairs):
    """pair 수 하나로 학습→회상→저장→시각화. 요약용 dict 반환. (bias 없음)"""
    cs, target = load_patterns(device, n_pairs)
    run_name = f"{args.run_name}_pair={n_pairs}"

    phi, weight = train_contrastive_association(
        cs, target, args.basis, args.target_one_current, args.target_zero_current
    )
    weight = torch.clamp(weight, -args.weight_clip, args.weight_clip)   # bias 없음

    pred = lif_recall(
        phi, weight,
        PARAMETER_SET["threshold"], PARAMETER_SET["tau"], PARAMETER_SET["dt"],
        args.recall_steps, args.count_threshold,
    )
    metrics = compute_metrics(pred, target)
    rank = int(torch.linalg.matrix_rank(phi).item())    # Φ 만의 랭크 (편향 열 없음)

    result = {
        "run_name": run_name,
        "variant": "remove_bias",
        "n_pairs": n_pairs,
        "fixed_neuron": {
            "threshold": PARAMETER_SET["threshold"],
            "tau": PARAMETER_SET["tau"],
            "dt": PARAMETER_SET["dt"],
        },
        "config": {
            "basis": args.basis,
            "bias": "removed",
            "n_pairs": n_pairs,
            "target_one_current": args.target_one_current,
            "target_zero_current": args.target_zero_current,
            "weight_clip": args.weight_clip,
            "feature_dim": int(phi.shape[1]),
            "feature_rank": rank,
        },
        "weight_stats": {
            "min": float(weight.min().item()),
            "max": float(weight.max().item()),
            "mean": float(weight.mean().item()),
        },
        **metrics,
    }

    # STDP 결과와 동일한 run 폴더 + 시각화
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)
    w_np = weight.detach().cpu().numpy()
    np.save(run_dir / "weights.npy", w_np.astype(np.float32))

    viz.plot_recall(result, run_dir / "recall.png", run_name, show=SHOW)
    viz.plot_weights(w_np, run_dir / "weights.png", run_name, show=SHOW)
    if SHOW:
        plt.show()
        plt.close("all")

    print(
        f"[pair={n_pairs:2d}] feature_dim={phi.shape[1]}, rank={rank}, "
        f"exact={metrics['exact_count']}/{n_pairs}, bit_acc={metrics['mean_bit_acc']:.3f}  "
        f"-> {run_dir}"
    )
    return {
        "n_pairs": n_pairs,
        "exact": metrics["exact_count"],
        "bit_acc": round(metrics["mean_bit_acc"], 4),
        "rank": rank,
        "feature_dim": int(phi.shape[1]),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --n-pairs 를 주면 그 하나만, 아니면 상단 N_PAIRS_SWEEP 전부 실행
    pair_list = [args.n_pairs] if args.n_pairs is not None else list(N_PAIRS_SWEEP)

    print("=== Contrastive optimal weight (REMOVE BIAS) — pair sweep ===")
    summary = [run_one(args, device, n) for n in pair_list]

    # 요약 CSV + 표 출력
    sweep_dir = Path(args.output_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    cols = ["n_pairs", "feature_dim", "rank", "exact", "bit_acc"]
    lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in summary]
    (sweep_dir / f"{args.run_name}_pair_sweep_summary.csv").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n--- summary ---")
    print(f"{'pairs':>5} {'dim':>4} {'rank':>5} {'exact':>7} {'bit_acc':>8}")
    for r in summary:
        print(f"{r['n_pairs']:>5} {r['feature_dim']:>4} {r['rank']:>5} "
              f"{str(r['exact'])+'/'+str(r['n_pairs']):>7} {r['bit_acc']:>8.3f}")
    print(f"\nsummary csv: {sweep_dir / (args.run_name + '_pair_sweep_summary.csv')}")


if __name__ == "__main__":
    main()
