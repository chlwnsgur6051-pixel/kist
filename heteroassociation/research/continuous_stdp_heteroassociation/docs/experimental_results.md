# 학습 결과와 기존 모델 대조

Original·Modified K=10, 보관된 seed 0의 1,000개 epoch별 제시 순서를 사용했다. 각 조건은 영행렬에서 독립 학습했다. W ≥ 0 조건만 presentation 종료 후 원소별 0 하한을 적용했다.

| Dataset | 조건 | 최초 10/10 | 마지막까지 유지 시작 | 최종 sequential | 최종 independent |
|---|---|---:|---:|---:|---:|
| Original | Baseline | 72 | 72 | 10/10 | 10/10 |
| Original | W ≥ 0 | 없음 | 없음 | 4/10 | 4/10 |
| Modified | Baseline | 10 | 10 | 10/10 | 10/10 |
| Modified | W ≥ 0 | 16 | 16 | 10/10 | 10/10 |

이전 λ=1 Basic Correction 기록과 새 sequential/independent recall의 epoch별 Exact Recall 비교는 각각 **4,000/4,000 일치**했다. 자연 출력의 4 ms 이외 발화는 네 조건 모두 0건이었다. 최종 가중치의 이전 모델 대비 최대 절댓값 차이는 Original Baseline `2.48×10⁻¹⁵`, Original W ≥ 0 `0`, Modified Baseline `1.25×10⁻¹⁵`, Modified W ≥ 0 `0`이다. λ=0.1 결과는 `experiments/lambda_0p1/`에 보존했다.

Association 경계를 가로지르는 spike pair의 합산 절댓값은 조건별 약 `6.62×10⁻¹³`–`9.86×10⁻¹³`이었다. 이 수치가 작은 것은 5 ms 무입력 간격과 해당 STDP 창에 한정된 관측이다. Sequential probe는 학습 상태의 복사본에서 평가하므로 학습 궤적에 영향을 주지 않는다.

원자료: [`epoch_history.csv`](../results/csv/epoch_history.csv), [`epoch_comparison.csv`](../results/csv/epoch_comparison.csv), [`summary.csv`](../results/csv/summary.csv), [`final_weights.npz`](../results/checkpoints/final_weights.npz). 시각화: [`accuracy_by_epoch.png`](../results/figures/accuracy_by_epoch.png), [`final_weights.png`](../results/figures/final_weights.png).
