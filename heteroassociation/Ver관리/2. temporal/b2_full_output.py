from pathlib import Path
import sys
import json
import numpy as np

HERE = Path(__file__).resolve().parent

# heteroassociation root
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stdp_temporal_symmetric_teacher_only as base


RESULTS = HERE / "results_full_B2_teacher_only"
RESULTS.mkdir(parents=True, exist_ok=True)


def train_one_output(out_idx: int):
    """
    기존 stdp_temporal.py의 learning rule을 그대로 사용해서
    output pixel 하나만 학습한다.
    """
    base.OUT_NEURON = out_idx

    # output마다 동일한 random permutation 조건 사용
    np.random.seed(0)

    w, tgt, acc_hist, w_hist = base.train()
    pred, _ = base.predict(w)

    return {
        "output": out_idx,
        "weights": w,
        "target": tgt,
        "pred": pred,
        "acc": int((pred == tgt).sum()),
        "acc_hist": acc_hist,
    }


def main():
    output_results = []

    for out_idx in range(15):
        result = train_one_output(out_idx)
        output_results.append(result)

        print(
            f"output {out_idx:2d}: "
            f"{result['acc']}/{base.N_PAIRS}"
        )

    # shape:
    # rows = digit 0~9
    # cols = output pixel 0~14
    pred_matrix = np.stack(
        [r["pred"] for r in output_results],
        axis=1
    )

    target_matrix = base.Y01.astype(int)

    # 각 15-bit letter를 전부 맞췄는지
    exact_each = np.all(
        pred_matrix == target_matrix,
        axis=1
    )

    exact_count = int(exact_each.sum())

    # 전체 150 bit 중 몇 개 맞췄는지
    bit_accuracy = float(
        np.mean(pred_matrix == target_matrix)
    )

    print()
    print("=== Full-output baseline ===")
    print(f"Exact association: {exact_count}/{base.N_PAIRS}")
    print(f"Bit accuracy: {bit_accuracy:.4f}")

    print("\nPrediction matrix:")
    print(pred_matrix)

    print("\nTarget matrix:")
    print(target_matrix)

    print("\nExact per digit:")
    print(exact_each.astype(int))

    # 결과 저장
    np.save(RESULTS / "predictions.npy", pred_matrix)
    np.save(RESULTS / "targets.npy", target_matrix)

    weight_matrix = np.stack(
        [r["weights"] for r in output_results],
        axis=1
    )
    np.save(RESULTS / "weights.npy", weight_matrix)

    metrics = {
        "n_pairs": int(base.N_PAIRS),
        "epochs": int(base.EPOCHS),
        "exact_count": exact_count,
        "exact_rate": exact_count / base.N_PAIRS,
        "bit_accuracy": bit_accuracy,
        "per_output_accuracy": [
            int(r["acc"]) for r in output_results
        ],
        "exact_per_digit": exact_each.astype(int).tolist(),
    }

    with open(
        RESULTS / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to: {RESULTS}")


if __name__ == "__main__":
    main()