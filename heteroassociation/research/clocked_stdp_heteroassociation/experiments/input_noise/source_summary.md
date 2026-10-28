# 입력 bit-flip에 대한 teacher-free Clocked LIF 복원 강건성

최신 검증된 Centered Inverted Mexican-hat 모델의 clean 학습 최종 weight를 고정해 사용했다. 동일 설정의 clean 학습을 재실행해 **500개 epoch checkpoint 모두 보관된 weight와 정확히 일치**함을 확인했다. Original과 Modified 각각 clean 입력 10개를 100% 복원했고 기존 LIF simulator와 출력이 일치했다. Noisy input 재학습, Dataset A, spike jitter는 사용하지 않았다.

## 실제 측정: 전체 exact / bit 정확도

| Dataset | Noise type | Flips | Cues | Exact (%) | Bit (%) | Mean output errors |
|---|---|---:|---:|---:|---:|---:|
| Original | mixed | 0 | 10 | 100.00 | 100.00 | 0.000 |
| Original | mixed | 1 | 150 | 14.00 | 81.60 | 2.760 |
| Original | mixed | 2 | 1050 | 9.71 | 77.61 | 3.359 |
| Original | mixed | 3 | 4550 | 6.37 | 73.72 | 3.942 |
| Original | mixed | 4 | 13650 | 4.08 | 71.03 | 4.345 |
| Original | remove_active | 0 | 10 | 100.00 | 100.00 | 0.000 |
| Original | remove_active | 1 | 104 | 13.46 | 81.15 | 2.827 |
| Original | remove_active | 2 | 505 | 9.70 | 76.90 | 3.465 |
| Original | remove_active | 3 | 1516 | 5.80 | 73.23 | 4.016 |
| Original | remove_active | 4 | 3136 | 3.51 | 70.81 | 4.379 |
| Original | add_inactive | 0 | 10 | 100.00 | 100.00 | 0.000 |
| Original | add_inactive | 1 | 46 | 15.22 | 82.61 | 2.609 |
| Original | add_inactive | 2 | 99 | 10.10 | 81.68 | 2.747 |
| Original | add_inactive | 3 | 135 | 5.19 | 79.01 | 3.148 |
| Original | add_inactive | 4 | 128 | 5.47 | 76.88 | 3.469 |
| Modified | mixed | 0 | 10 | 100.00 | 100.00 | 0.000 |
| Modified | mixed | 1 | 150 | 34.00 | 90.49 | 1.427 |
| Modified | mixed | 2 | 1050 | 18.76 | 85.41 | 2.189 |
| Modified | mixed | 3 | 4550 | 12.84 | 81.96 | 2.706 |
| Modified | mixed | 4 | 13650 | 8.70 | 79.28 | 3.108 |
| Modified | remove_active | 0 | 10 | 100.00 | 100.00 | 0.000 |
| Modified | remove_active | 1 | 70 | 34.29 | 92.19 | 1.171 |
| Modified | remove_active | 2 | 214 | 21.96 | 87.48 | 1.879 |
| Modified | remove_active | 3 | 373 | 15.01 | 83.77 | 2.434 |
| Modified | remove_active | 4 | 405 | 9.88 | 79.93 | 3.010 |
| Modified | add_inactive | 0 | 10 | 100.00 | 100.00 | 0.000 |
| Modified | add_inactive | 1 | 80 | 33.75 | 89.00 | 1.650 |
| Modified | add_inactive | 2 | 284 | 15.49 | 83.76 | 2.437 |
| Modified | add_inactive | 3 | 589 | 9.34 | 81.41 | 2.788 |
| Modified | add_inactive | 4 | 791 | 5.31 | 79.78 | 3.033 |

## Association별 50% exact recall 허용 flip 수

각 noise type에 대해 측정된 k=0..4 중 exact recall이 **50% 초과**인 최대 k를 표시한다. 중간 k가 비단조적이어도 최대값만 취하는 정의이며, 평균은 10개 digit의 산술 평균이다.

| Dataset | Noise type | Mean maximum flips | Per-digit maximum (0..9) |
|---|---|---:|---|
| Original | mixed | 0.00 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| Original | remove_active | 0.00 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| Original | add_inactive | 0.00 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| Modified | mixed | 0.30 | [1, 0, 1, 0, 0, 0, 1, 0, 0, 0] |
| Modified | remove_active | 0.30 | [0, 0, 1, 0, 0, 1, 1, 0, 0, 0] |
| Modified | add_inactive | 0.20 | [1, 0, 0, 0, 0, 0, 1, 0, 0, 0] |

## 해석

### Original
- 1 flips: mixed 14.00%, 1→0 13.46%, 0→1 15.22%.
- 2 flips: mixed 9.71%, 1→0 9.70%, 0→1 10.10%.
- 3 flips: mixed 6.37%, 1→0 5.80%, 0→1 5.19%.
- 4 flips: mixed 4.08%, 1→0 3.51%, 0→1 5.47%.
- Mixed 2-flip에서 특히 취약한 digit: 9 (3.8%), 2 (4.8%), 5 (4.8%).
- Mixed 2-flip에서 가장 취약한 output neuron: 11 (64.8%), 8 (65.5%), 3 (69.0%).
- Mixed 2-flip의 exact 9.71%와 bit 77.61% 차이는 일부 output bit 오류만 있어도 association 전체가 실패로 분류되는 특성을 보여준다.
### Modified
- 1 flips: mixed 34.00%, 1→0 34.29%, 0→1 33.75%.
- 2 flips: mixed 18.76%, 1→0 21.96%, 0→1 15.49%.
- 3 flips: mixed 12.84%, 1→0 15.01%, 0→1 9.34%.
- 4 flips: mixed 8.70%, 1→0 9.88%, 0→1 5.31%.
- Mixed 2-flip에서 특히 취약한 digit: 9 (1.9%), 8 (3.8%), 1 (5.7%).
- Mixed 2-flip에서 가장 취약한 output neuron: 13 (67.5%), 7 (75.2%), 8 (76.2%).
- Mixed 2-flip의 exact 18.76%와 bit 85.41% 차이는 일부 output bit 오류만 있어도 association 전체가 실패로 분류되는 특성을 보여준다.

Original/Modified 비교와 입력 overlap/crosstalk 설명은 이 두 고정 dataset에 한정된다. Noise robustness를 새로운 학습 규칙의 개선 증거로 해석하지 않는다. 상세 생성 방식은 `methods.md`, 전수 결과는 `csv/noisy_recall_per_cue.csv`에 있다.

동일한 mixed-flip 조건에서 Modified는 1–4비트 모두 Original보다 높은 exact 및 bit accuracy를 보였다. 1→0이 일관되게 더 해로운 것은 아니다. Modified에서는 2–4비트에서 0→1이 더 해로웠고, Original은 수준에 따라 순서가 바뀌었다. Directional 조건마다 가능한 원본 bit의 수가 달라 표본 집합도 다르다는 점을 함께 봐야 한다.

오류는 출력 15비트가 한꺼번에 무너지는 형태만은 아니었다. Mixed 2-flip에서 Original의 bit accuracy는 77.61%이지만 exact는 9.71%; Modified는 각각 85.41%, 18.76%였다. 완전 복원 판정은 몇 개 output bit 오류에도 급감한다.

Original의 digit별 가장 가까운 다른 clean input까지의 Hamming 거리는 평균 2.3비트, Modified는 3.3비트였다. 이는 Modified의 높은 강건성과 방향이 맞지만, 개별 digit의 1-flip 성능을 거리 하나로 설명하지는 못한다. 입력 overlap과 학습된 출력별 weight margin이 함께 영향을 줄 수 있다는 해석이며, 인과적 증명은 아니다.
