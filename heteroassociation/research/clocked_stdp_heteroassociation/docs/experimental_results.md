# 검증된 결과

아래 값은 ascending 순서, 500 epoch, teacher-free Clocked LIF readout이다. Exact는 15개 출력 bit가 전부 맞는 association 비율, bit 정확도는 전체 출력 bit의 정답률이다.

| 데이터 | K | 최초 100% epoch | Best exact (epoch) | Final exact | Final bit |
|---|---:|---:|---:|---:|---:|
| Original | 10 | 65 | 100% (65) | 100% | 100% |
| Modified | 10 | 10 | 100% (10) | 100% | 100% |
| Original Custom v2 | 15 | 117 | 100% (117) | 100% | 100% |
| Modified Custom v2 | 15 | 69 | 100% (69) | 100% | 100% |
| Original Custom final | 20 | 없음 | 60% (206) | 55% | 95.0% |
| Modified Custom final | 20 | 없음 | 75% (33) | 70% | 97.67% |

K=10/15/20의 원본 epoch별 weight는 `results/checkpoints/`에 있고 `results/regression_verification.json`에 재실행 대조가 있다. 전체 K=10–20 curve는 `results/csv/k20_capacity_accuracy_summary.csv`, K15 학습 순서는 `k15_capacity_accuracy_summary.csv`를 참조한다. K20은 최고 epoch 이후에도 성능이 떨어져 best와 final을 구분해야 한다.

K=10 고정 가중치에서 mixed 1-bit 입력 flip의 exact는 Original 14%, Modified 34%였다. 이는 **복원 시점 입력 변형** 결과다. Jitter 학습에서 combined σ=0.1 ms, seed 10개의 평균 최종 exact는 Original 90%, Modified 96%다. 두 값은 서로 다른 조건이며, 각각 `results/csv/input_noise_*`와 `jitter_*`의 원본 요약 및 `artifact_manifest.csv`의 원본 경로를 참조한다.
