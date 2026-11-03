# 학습 결과와 기존 모델 대조

Original·Modified K=10, 보관된 seed 0의 1,000개 epoch별 제시 순서를 사용했다. 각 조건은 영행렬에서 독립 학습했다. W ≥ 0 조건만 presentation 종료 후 원소별 0 하한을 적용했다.

| Dataset | 조건 | 최초 10/10 | 마지막까지 유지 시작 | 최종 sequential | 최종 independent |
|---|---|---:|---:|---:|---:|
| Original | Baseline | 295 | 295 | 10/10 | 10/10 |
| Original | W ≥ 0 | 없음 | 없음 | 5/10 | 5/10 |
| Modified | Baseline | 48 | 48 | 10/10 | 10/10 |
| Modified | W ≥ 0 | 52 | 52 | 10/10 | 10/10 |

기존 Basic Correction 기록과 새 sequential/independent recall의 epoch별 Exact Recall 비교는 각각 **4,000/4,000 일치**했다. 자연 출력의 4 ms 이외 발화는 네 조건 모두 0건이었다. 최종 가중치의 이전 모델 대비 최대 절댓값 차이는 Original Baseline `8.34×10⁻¹⁷`, Original W ≥ 0 `0`, Modified Baseline `1.23×10⁻¹⁶`, Modified W ≥ 0 `0`이다.

기존의 trial-local 이벤트 이력과 달리 이 모델은 association 경계 간 pair를 계산했다. 해당 pair는 조건에 따라 약 835만–1242만 개 계수로 기록됐지만, 합산 절댓값은 `6.56×10⁻¹⁴`에서 `9.76×10⁻¹⁴` 사이였다. **쌍의 개수만으로 학습 영향의 크기를 판단할 수 없다.** Sequential probe는 학습 상태의 복사본에서 평가하므로 학습 궤적에 영향을 주지 않는다.

원자료: [`epoch_history.csv`](../results/csv/epoch_history.csv), [`epoch_comparison.csv`](../results/csv/epoch_comparison.csv), [`summary.csv`](../results/csv/summary.csv), [`final_weights.npz`](../results/checkpoints/final_weights.npz). 시각화: [`accuracy_by_epoch.png`](../results/figures/accuracy_by_epoch.png), [`final_weights.png`](../results/figures/final_weights.png).
