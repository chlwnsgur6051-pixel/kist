import numpy as np
import stdp_temporal_symmetric_teacher_only as base


def main():
    X01 = base.X01[:base.N_PAIRS].astype(int)
    Y01 = base.Y01[:base.N_PAIRS].astype(int)

    # 0/1 -> -1/+1
    X = 2 * X01 - 1
    Y = 2 * Y01 - 1

    # correlation matrix
    C = X.T @ Y

    # correlation-memory recall score
    score = X @ C

    # self signal:
    # x_p dot x_p = 15
    desired = 15 * Y

    # contribution from all other stored patterns
    crosstalk = score - desired

    pred = (score > 0).astype(int)

    bit_acc = np.mean(pred == Y01)
    exact = np.all(pred == Y01, axis=1)

    print("=== Correlation / Crosstalk Diagnostic ===")
    print(f"bit accuracy : {bit_acc:.4f}")
    print(f"exact        : {exact.sum()}/{base.N_PAIRS}")

    print("\n=== Per-digit ===")

    for p in range(base.N_PAIRS):
        errors = np.sum(pred[p] != Y01[p])

        print(
            f"digit {p}: "
            f"errors={errors:2d}  "
            f"exact={int(exact[p])}  "
            f"mean|cross|={np.mean(np.abs(crosstalk[p])):.2f}  "
            f"max|cross|={np.max(np.abs(crosstalk[p])):.2f}"
        )

    # out7을 자세히 보기
    j = 7

    print("\n=== Output 7 detail ===")

    for p in range(base.N_PAIRS):
        print(
            f"digit {p}: "
            f"target={Y01[p, j]}  "
            f"desired={desired[p, j]:+3d}  "
            f"cross={crosstalk[p, j]:+4d}  "
            f"total={score[p, j]:+4d}  "
            f"pred={pred[p, j]}"
        )
        
    margin = Y * score
    cross_margin = Y * crosstalk

    print("\n=== Margin diagnostic ===")

    for p in range(base.N_PAIRS):
        wrong = int(np.sum(margin[p] <= 0))

        print(
            f"digit {p}: "
            f"wrong={wrong:2d}  "
            f"min_margin={margin[p].min():+4d}  "
            f"mean_margin={margin[p].mean():6.2f}  "
            f"worst_cross={cross_margin[p].min():+4d}"
        )


if __name__ == "__main__":
    main()