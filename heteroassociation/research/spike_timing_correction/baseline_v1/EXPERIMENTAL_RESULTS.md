# 검증된 실험 결과

결과의 원자료는 `results/csv/*_epoch_history.csv`, `results/checkpoints/*_training.npz`, `configs/*_archived_condition.json`이다. 그래프는 `figures/`에 있다. 이 페이지의 수치는 2026-11-05에 보관본 코드로 1000 epoch를 **두 데이터셋 각각 다시 실행**하여 원본과 비교했다. 매 epoch exact·bit accuracy 1000/1000, 저장된 전체 post-spike event와 최종 W가 일치했고 최종 W 최대 절댓값 차이는 두 조건 모두 `0.0`이었다.

| Dataset | 첫 10/10 epoch | 마지막까지 연속 10/10 시작 | Epoch 1000 exact | Epoch 1000 bit | 최종 W 범위 |
|---|---:|---:|---:|---:|---:|
| Original | 113 | 226 | 10/10 | 100% | −0.27883984283576996 ~ +0.6911356834624628 |
| Modified | 21 | 37 | 10/10 | 100% | −0.12174276998301942 ~ +0.41390660988723293 |

첫 10/10 뒤 일부 epoch에서 정확도가 떨어질 수 있어 두 epoch 값을 구분한다. “지속”은 **관측한 마지막 epoch 1000까지**를 의미한다.

## 실제 spike 기록

각 데이터셋에서 1000 epoch × 10 association 동안 예상한 target=1 teacher spike **100,000개가 모두** 실제 LIF에서 관측됐다. Target=0에는 teacher spike가 없으며, 추가 natural output spike는 보관된 두 조건에서 **0개**였다. 학습 중 자연 spike는 Original **98,991개**, Modified **99,764개**였다. 가장 이른 natural spike는 Original 2.5 ms, Modified 3.0 ms, 가장 늦은 것은 두 데이터셋 모두 4.5 ms다. 이는 STDP timing correction이 실제 event 시각을 사용했다는 관측 근거이며, 모든 조건의 보편적 속성으로 확장할 수 없다.

원본 이벤트 NPZ에는 presentation별 15개 출력 뉴런의 post-spike tick과 개수가 있다. [그림](figures/stdp_early_late.png)은 Original 0→A output 0의 보관된 실측 event를 표시한다. Epoch 1의 자연 spike 3.4 ms에서는 그 trial 내 한 활성 input의 teacher/natural pair 순기여가 약 −0.0122, epoch 6의 4.5 ms에서는 약 +0.0115였다. 전체 학습 업데이트에는 이전 association에서 유지된 spike history의 pair도 들어간다. 따라서 이 그림의 두 숫자는 전체 association `ΔW`의 합계로 제시하면 안 된다.

## Recall 및 그림

공식 학습 정확도는 각 epoch 직후 **teacher-free 연속 cue probe**로 산출했다. 학습 state를 복사한 뒤 association 0–9를 순서대로 제시하며, probe가 실제 학습 상태와 W를 변경하지 않는다. 보관된 최종 W를 NPZ에서 다시 읽어 별도 초기화된 teacher-free recall로 시험했을 때도 Original·Modified 각각 10/10이었다. 이 별도 시험은 공식 연속 probe 지표와 구분해 기록한다.

- [Original 학습 정오답](figures/original_learning.png), [Modified 학습 정오답](figures/modified_learning.png): 초록=15비트 exact correct, 빨강=오답.
- [정확도 곡선](figures/accuracy_by_epoch.png): 두 데이터셋의 epoch별 exact recall.
- [최종 signed W](figures/final_weights.png): 색 범위 통일. **정확한 수치는 NPZ**에 있다.
- [Original의 실제 학습/recall 장면](figures/original_training_vs_recall.png), [선택한 spike·막전위 trace](figures/original_epoch6_spike_membrane.png): teacher는 학습에만 존재한다.

## 측정과 해석의 구분

K=10에서 두 데이터셋이 10/10에 도달하고 유지된 것은 측정 결과다. Modified가 더 빨리 지속 10/10에 도달한 원인이 단일 요인이라는 결론은 이 비교만으로는 나오지 않는다. 이 자료는 noisy membrane, 지수형 PSC, K>10 capacity, on-chip 구현을 검증하지 않는다. 이전 Basic Correction 또는 nonnegative 조건의 성능은 이 baseline 결과와 혼용하지 않는다.
