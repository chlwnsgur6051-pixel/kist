from itertools import combinations
import numpy as np
from pathlib import Path
import json
import csv

import stdp_temporal_symmetric as base

RESULT_DIR = Path(__file__).resolve().parent / "results_pair_count_sweep"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULT_DIR / "subset_results2.csv"
JSON_PATH = RESULT_DIR / "summary2.json"

PAIR_COUNTS = [1, 3, 5, 7, 10]

# 원본 데이터 보존
FULL_X = base.X01.copy()
FULL_Y = base.Y01.copy()


def train_one_output(out_idx):
    base.OUT_NEURON = out_idx

    # 모든 output에서 같은 학습 순서
    np.random.seed(0)

    w, tgt, _, _ = base.train()
    pred, _ = base.predict(w)

    return pred


def evaluate_subset(indices):
    """
    지정한 association subset에 대해서
    15개 output neuron을 각각 학습하고
    full-pattern exact / bit accuracy 계산
    """
    indices = list(indices)

    base.X01 = FULL_X[indices].copy()
    base.Y01 = FULL_Y[indices].copy()
    base.N_PAIRS = len(indices)

    preds = []

    for out_idx in range(15):
        pred = train_one_output(out_idx)
        preds.append(pred)

    # (15, n_pairs) -> (n_pairs, 15)
    pred_matrix = np.stack(preds, axis=1)
    target_matrix = base.Y01.astype(int)

    exact_each = np.all(
        pred_matrix == target_matrix,
        axis=1
    )

    exact_count = int(exact_each.sum())
    exact_rate = exact_count / len(indices)

    bit_accuracy = float(
        np.mean(pred_matrix == target_matrix)
    )

    return exact_count, exact_rate, bit_accuracy


def main():
    print("=== Pair-count sweep ===")

    summary = {}

    # 각 subset 결과를 모두 저장
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "n_pairs",
            "subset",
            "exact_count",
            "exact_rate",
            "bit_accuracy"
        ])

        for n_pairs in PAIR_COUNTS:

            subset_exact_rates = []
            subset_bit_accs = []
            perfect_subsets = 0

            all_subsets = list(
                combinations(range(10), n_pairs)
            )

            for indices in all_subsets:
                exact_count, exact_rate, bit_acc = \
                    evaluate_subset(indices)

                subset_exact_rates.append(exact_rate)
                subset_bit_accs.append(bit_acc)

                if exact_count == n_pairs:
                    perfect_subsets += 1

                # subset 하나 끝날 때마다 바로 파일 저장
                writer.writerow([
                    n_pairs,
                    "-".join(map(str, indices)),
                    exact_count,
                    exact_rate,
                    bit_acc
                ])
                csvfile.flush()

            exact_rates = np.array(subset_exact_rates)
            bit_accs = np.array(subset_bit_accs)

            result = {
                "n_pairs": n_pairs,
                "subsets_tested": len(all_subsets),
                "mean_exact_rate": float(exact_rates.mean()),
                "median_exact_rate": float(np.median(exact_rates)),
                "best_exact_rate": float(exact_rates.max()),
                "worst_exact_rate": float(exact_rates.min()),
                "perfect_subsets": int(perfect_subsets),
                "mean_bit_accuracy": float(bit_accs.mean()),
            }

            summary[str(n_pairs)] = result

            print()
            print(f"N_PAIRS = {n_pairs}")
            print(f"subsets tested      : {result['subsets_tested']}")
            print(f"mean exact rate     : {result['mean_exact_rate']:.4f}")
            print(f"median exact rate   : {result['median_exact_rate']:.4f}")
            print(f"best exact rate     : {result['best_exact_rate']:.4f}")
            print(f"worst exact rate    : {result['worst_exact_rate']:.4f}")
            print(
                f"perfect subsets     : "
                f"{perfect_subsets}/{len(all_subsets)}"
            )
            print(
                f"mean bit accuracy   : "
                f"{result['mean_bit_accuracy']:.4f}"
            )

            # pair count 하나 끝날 때마다 summary 저장
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    # 원본 복원
    base.X01 = FULL_X
    base.Y01 = FULL_Y
    base.N_PAIRS = 10

    print()
    print(f"CSV saved  : {CSV_PATH}")
    print(f"JSON saved : {JSON_PATH}")
    
if __name__ == "__main__":
    main()