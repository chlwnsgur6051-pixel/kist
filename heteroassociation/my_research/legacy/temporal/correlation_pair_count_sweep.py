from __future__ import annotations

import sys
import json
import csv
from itertools import combinations
from pathlib import Path

import numpy as np


# =========================================================
# 경로 / 데이터
# =========================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import binary_character as data


digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()

X01 = digit_images.reshape(10, 15).astype(np.int64)
Y01 = letter_images.reshape(10, 15).astype(np.int64)


# =========================================================
# 설정
# =========================================================

PAIR_COUNTS = [1, 3, 5, 7, 10]

RESULT_DIR = HERE / "results_correlation_pair_count"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULT_DIR / "subset_results.csv"
SUMMARY_PATH = RESULT_DIR / "summary.json"


# =========================================================
# Correlation memory
# =========================================================

def evaluate_subset(indices):
    """
    선택한 association subset에 대해
    pure bipolar correlation memory를 구성하고 recall한다.

    X : input digit
    Y : output letter

    W = X^T Y
    score = X W
    pred = sign(score)
    """

    indices = list(indices)

    x01 = X01[indices]
    y01 = Y01[indices]

    # 0/1 -> -1/+1
    xb = 2 * x01 - 1
    yb = 2 * y01 - 1

    # -----------------------------------------------------
    # 학습
    # -----------------------------------------------------
    # shape:
    # xb : (n_pairs, 15)
    # yb : (n_pairs, 15)
    # W  : (15, 15)
    # -----------------------------------------------------

    W = xb.T @ yb

    # -----------------------------------------------------
    # recall
    # -----------------------------------------------------

    scores = xb @ W

    # score > 0 -> 1
    # score <= 0 -> 0
    pred = (scores > 0).astype(np.int64)

    # -----------------------------------------------------
    # accuracy
    # -----------------------------------------------------

    correct_bits = pred == y01

    bit_accuracy = float(correct_bits.mean())

    exact_each = np.all(correct_bits, axis=1)

    exact_count = int(exact_each.sum())
    exact_rate = exact_count / len(indices)

    # -----------------------------------------------------
    # margin
    #
    # target bipolar sign과 score가 같은 방향이면 positive
    # margin < 0이면 해당 bit가 잘못 recall된 것
    # -----------------------------------------------------

    margins = yb * scores

    mean_margin = float(margins.mean())
    min_margin = float(margins.min())

    wrong_bits = int((margins <= 0).sum())

    return {
        "indices": indices,
        "exact_count": exact_count,
        "exact_rate": exact_rate,
        "bit_accuracy": bit_accuracy,
        "mean_margin": mean_margin,
        "min_margin": min_margin,
        "wrong_bits": wrong_bits,
    }


# =========================================================
# Sweep
# =========================================================

def main():

    print("=== Pure Correlation Memory Pair-count Sweep ===")
    print()
    print("X bipolar : 0 -> -1, 1 -> +1")
    print("Y bipolar : 0 -> -1, 1 -> +1")
    print("W = X^T @ Y")
    print("score = X @ W")
    print("pred = score > 0")
    print()

    summary = {}

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "n_pairs",
            "subset",
            "exact_count",
            "exact_rate",
            "bit_accuracy",
            "mean_margin",
            "min_margin",
            "wrong_bits",
        ])

        for n_pairs in PAIR_COUNTS:

            all_subsets = list(
                combinations(range(10), n_pairs)
            )

            exact_rates = []
            bit_accuracies = []
            mean_margins = []
            min_margins = []

            perfect_subsets = 0

            for indices in all_subsets:

                result = evaluate_subset(indices)

                exact_rates.append(
                    result["exact_rate"]
                )

                bit_accuracies.append(
                    result["bit_accuracy"]
                )

                mean_margins.append(
                    result["mean_margin"]
                )

                min_margins.append(
                    result["min_margin"]
                )

                if result["exact_count"] == n_pairs:
                    perfect_subsets += 1

                writer.writerow([
                    n_pairs,
                    "-".join(map(str, indices)),
                    result["exact_count"],
                    result["exact_rate"],
                    result["bit_accuracy"],
                    result["mean_margin"],
                    result["min_margin"],
                    result["wrong_bits"],
                ])

                f.flush()

            exact_rates = np.array(exact_rates)
            bit_accuracies = np.array(bit_accuracies)
            mean_margins = np.array(mean_margins)
            min_margins = np.array(min_margins)

            result_summary = {
                "n_pairs": n_pairs,
                "subsets_tested": len(all_subsets),

                "mean_exact_rate":
                    float(exact_rates.mean()),

                "median_exact_rate":
                    float(np.median(exact_rates)),

                "best_exact_rate":
                    float(exact_rates.max()),

                "worst_exact_rate":
                    float(exact_rates.min()),

                "perfect_subsets":
                    int(perfect_subsets),

                "mean_bit_accuracy":
                    float(bit_accuracies.mean()),

                "mean_margin":
                    float(mean_margins.mean()),

                "worst_min_margin":
                    float(min_margins.min()),
            }

            summary[str(n_pairs)] = result_summary

            # 중간 결과도 계속 저장
            with open(
                SUMMARY_PATH,
                "w",
                encoding="utf-8"
            ) as sf:
                json.dump(
                    summary,
                    sf,
                    indent=2
                )

            print(
                f"N_PAIRS = {n_pairs}"
            )

            print(
                f"subsets tested      : "
                f"{len(all_subsets)}"
            )

            print(
                f"mean exact rate     : "
                f"{exact_rates.mean():.4f}"
            )

            print(
                f"median exact rate   : "
                f"{np.median(exact_rates):.4f}"
            )

            print(
                f"best exact rate     : "
                f"{exact_rates.max():.4f}"
            )

            print(
                f"worst exact rate    : "
                f"{exact_rates.min():.4f}"
            )

            print(
                f"perfect subsets     : "
                f"{perfect_subsets}/"
                f"{len(all_subsets)}"
            )

            print(
                f"mean bit accuracy   : "
                f"{bit_accuracies.mean():.4f}"
            )

            print(
                f"mean margin         : "
                f"{mean_margins.mean():.4f}"
            )

            print(
                f"worst min margin    : "
                f"{min_margins.min():.4f}"
            )

            print()

    print("=== Saved ===")
    print(f"CSV     : {CSV_PATH}")
    print(f"Summary : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()