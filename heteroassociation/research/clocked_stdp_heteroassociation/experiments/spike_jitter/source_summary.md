# Spike timing jitter robustness

검증 범위: full: 500 epochs and 10 independent seeds per condition. Part A에서는 1 ms까지 STDP event-pair 민감도를 측정했다. 최종 recall은 모든 조건에서 clean input, teacher-free, jitter-free Clocked LIF로 평가했다.

## STDP cancellation

Nominal F(2)=-0.066666666666666666, F(6)=0.066666666666666666; 합=0.

| Jitter source | σ (ms) | Correct-11 residual mean | Correct-11 residual SD |
|---|---:|---:|---:|
| input | 0.0 | 0 | 0 |
| input | 0.1 | -5.432e-05 | 0.0088616 |
| input | 0.2 | -0.00018463 | 0.017531 |
| input | 0.5 | -0.00085861 | 0.041381 |
| input | 1.0 | -0.0027159 | 0.069916 |
| prediction | 0.0 | 0 | 0 |
| prediction | 0.1 | 1.9495e-05 | 0.012119 |
| prediction | 0.2 | -0.0001704 | 0.02404 |
| prediction | 0.5 | -0.00090904 | 0.0567 |
| prediction | 1.0 | -0.00025943 | 0.09448 |
| teacher | 0.0 | 0 | 0 |
| teacher | 0.1 | 1.0178e-05 | 0.0033083 |
| teacher | 0.2 | -1.8877e-05 | 0.0065462 |
| teacher | 0.5 | -0.00013976 | 0.015333 |
| teacher | 1.0 | -0.0018884 | 0.025177 |
| combined | 0.0 | 0 | 0 |
| combined | 0.1 | -8.2559e-05 | 0.015313 |
| combined | 0.2 | 4.6598e-05 | 0.030117 |
| combined | 0.5 | -0.0015186 | 0.067606 |
| combined | 1.0 | -0.0054703 | 0.10296 |

## Full learning

| Dataset | Jitter source | σ (ms) | Mean final exact (%) | SD | Final 100% runs | Lost after earlier 100% |
|---|---|---:|---:|---:|---:|---:|
| Original | input | 0.0 | 100.00 | 0.00 | 10/10 | 0 |
| Original | input | 0.1 | 89.00 | 15.78 | 5/10 | 10 |
| Original | input | 0.2 | 53.00 | 19.00 | 0/10 | 10 |
| Original | input | 0.5 | 26.00 | 10.20 | 0/10 | 0 |
| Original | prediction | 0.0 | 100.00 | 0.00 | 10/10 | 0 |
| Original | prediction | 0.1 | 89.00 | 12.21 | 5/10 | 10 |
| Original | prediction | 0.2 | 79.00 | 15.13 | 2/10 | 10 |
| Original | prediction | 0.5 | 63.00 | 25.32 | 1/10 | 10 |
| Original | teacher | 0.0 | 100.00 | 0.00 | 10/10 | 0 |
| Original | teacher | 0.1 | 93.00 | 15.52 | 8/10 | 10 |
| Original | teacher | 0.2 | 93.00 | 11.00 | 7/10 | 10 |
| Original | teacher | 0.5 | 94.00 | 6.63 | 5/10 | 10 |
| Original | combined | 0.0 | 100.00 | 0.00 | 10/10 | 0 |
| Original | combined | 0.1 | 90.00 | 13.42 | 5/10 | 10 |
| Original | combined | 0.2 | 73.00 | 27.22 | 3/10 | 10 |
| Original | combined | 0.5 | 33.00 | 6.40 | 0/10 | 0 |
| Modified | input | 0.0 | 100.00 | 0.00 | 10/10 | 0 |
| Modified | input | 0.1 | 95.00 | 6.71 | 6/10 | 10 |
| Modified | input | 0.2 | 97.00 | 4.58 | 7/10 | 10 |
| Modified | input | 0.5 | 86.00 | 9.17 | 2/10 | 10 |
| Modified | prediction | 0.0 | 100.00 | 0.00 | 10/10 | 0 |
| Modified | prediction | 0.1 | 96.00 | 9.17 | 8/10 | 10 |
| Modified | prediction | 0.2 | 85.00 | 12.85 | 3/10 | 10 |
| Modified | prediction | 0.5 | 84.00 | 20.10 | 5/10 | 10 |
| Modified | teacher | 0.0 | 100.00 | 0.00 | 10/10 | 0 |
| Modified | teacher | 0.1 | 99.00 | 3.00 | 9/10 | 10 |
| Modified | teacher | 0.2 | 97.00 | 6.40 | 8/10 | 10 |
| Modified | teacher | 0.5 | 94.00 | 9.17 | 6/10 | 10 |
| Modified | combined | 0.0 | 100.00 | 0.00 | 10/10 | 0 |
| Modified | combined | 0.1 | 96.00 | 4.90 | 6/10 | 10 |
| Modified | combined | 0.2 | 92.00 | 16.61 | 8/10 | 10 |
| Modified | combined | 0.5 | 75.00 | 13.60 | 0/10 | 10 |

## 기전 분리

이 결과는 수학적 STDP pair 창의 변화와 실제 LIF event 발생 변화를 구분해서 해석해야 한다. 아래 값은 jittered prediction spike와 같은 시점의 weight에서 계산한 nominal 4-ms Clocked LIF prediction의 bit disagreement 비율이다. Teacher spike 신뢰도 및 correct-11 residual과 함께 보아야 한다.

| Dataset | Source | σ (ms) | Prediction disagreement (%) | Teacher reliability (%) | Mean correct-11 residual magnitude per run |
|---|---|---:|---:|---:|---:|
| Original | input | 0.0 | 0.000 | 100.00 | 0.000 |
| Original | input | 0.1 | 1.003 | 100.00 | 3379.297 |
| Original | input | 0.2 | 2.479 | 100.00 | 6719.582 |
| Original | input | 0.5 | 5.460 | 100.00 | 14736.500 |
| Original | prediction | 0.0 | 0.000 | 100.00 | 0.000 |
| Original | prediction | 0.1 | 0.021 | 100.00 | 4626.531 |
| Original | prediction | 0.2 | 0.038 | 100.00 | 9341.453 |
| Original | prediction | 0.5 | 0.065 | 100.00 | 21895.453 |
| Original | teacher | 0.0 | 0.000 | 100.00 | 0.000 |
| Original | teacher | 0.1 | 0.000 | 100.00 | 1273.934 |
| Original | teacher | 0.2 | 0.000 | 100.00 | 2618.017 |
| Original | teacher | 0.5 | 0.000 | 100.00 | 6195.613 |
| Original | combined | 0.0 | 0.000 | 100.00 | 0.000 |
| Original | combined | 0.1 | 1.076 | 100.00 | 6261.996 |
| Original | combined | 0.2 | 2.322 | 100.00 | 11637.295 |
| Original | combined | 0.5 | 5.214 | 100.00 | 24656.879 |
| Modified | input | 0.0 | 0.000 | 100.00 | 0.000 |
| Modified | input | 0.1 | 0.177 | 100.00 | 2348.142 |
| Modified | input | 0.2 | 0.564 | 100.00 | 4799.767 |
| Modified | input | 0.5 | 2.261 | 100.00 | 11328.436 |
| Modified | prediction | 0.0 | 0.000 | 100.00 | 0.000 |
| Modified | prediction | 0.1 | 0.011 | 100.00 | 3231.352 |
| Modified | prediction | 0.2 | 0.022 | 100.00 | 6601.244 |
| Modified | prediction | 0.5 | 0.046 | 100.00 | 15586.327 |
| Modified | teacher | 0.0 | 0.000 | 100.00 | 0.000 |
| Modified | teacher | 0.1 | 0.000 | 100.00 | 885.418 |
| Modified | teacher | 0.2 | 0.000 | 100.00 | 1805.379 |
| Modified | teacher | 0.5 | 0.000 | 100.00 | 4354.117 |
| Modified | combined | 0.0 | 0.000 | 100.00 | 0.000 |
| Modified | combined | 0.1 | 0.225 | 100.00 | 4391.876 |
| Modified | combined | 0.2 | 0.657 | 100.00 | 8376.082 |
| Modified | combined | 0.5 | 2.545 | 100.00 | 18403.761 |

Teacher 전류와 clocked readout은 감독 및 동기화 가정이다. Natural LIF 학습을 검증한 결과가 아니다.

## 결론과 검증 범위

- **기준 재현:** σ=0에서 Original·Modified 모두 500 epoch 최종 정확도 100%이다. Original은 첫 100%가 epoch 65, Modified는 epoch 10이었다. 보관된 매 epoch 정확도와 일치했고, 보관 가중치와의 최대 원소 차이는 1e-12 검증 허용치 안이었다. 명목 Δt=2 ms에서 LTD −1/15, Δt=6 ms에서 LTP +1/15이며 올바른 1/1 상태의 합은 0이다.
- **작은 지터도 지속 학습을 흔든다:** σ=0.1 ms에서 500 epoch 최종 exact accuracy 평균은 Original 89–93%, Modified 95–99%이다. 이는 seed 10개의 평균이며 단일 실행의 보장이 아니다. 대부분의 비영(非零) 지터 조건에서 일시적으로 100%를 달성한 후 한 번 이상 다시 잃었다. `lost_100_later`는 최종 실패만이 아니라 *첫 100% 이후 어느 epoch에서든* 정확도가 내려간 실행을 센다.
- **지터 원인별 영향:** σ=0.5 ms에서 Original의 최종 exact accuracy는 입력 26%, prediction clock 63%, teacher 94%, 결합 33%이다. Modified는 각각 86%, 84%, 94%, 75%이다. 이 설정에서는 입력 지터가 Original의 Clocked LIF prediction 자체를 가장 자주 바꾸고(명목 clock prediction과 5.46% bit disagreement), prediction-clock 지터는 prediction 존재 변화가 0.065%에 그쳐도 STDP 상대시각과 1/1 상쇄를 흐트러뜨렸다. Teacher 단독 지터의 최종 정확도 영향은 더 작았다.
- **상쇄의 수치적 취약성:** 직접 STDP pair 시험에서 σ=0.5 ms의 올바른 1/1 잔여 업데이트 표준편차는 입력 0.0414, prediction 0.0567, teacher 0.0153, 결합 0.0676이었다(한 활성 시냅스의 weight 단위). 평균 잔여값도 정확히 0은 아니었다. 이 값은 막전위 재시뮬레이션이 없는 Part A 결과이며, Part B는 실제 grid event와 LIF firing을 사용했다.
- **부호 변화:** σ=0.5 ms의 Part B에서 명목상 LTD인 prediction pair가 양의 업데이트를 만든 경우가 Original 입력 지터에서 691,178건, 결합 지터에서 1,097,298건이었다(각각 모든 양·음 pair 사건의 약 6.71%, 10.72%). 이 수치는 10개 seed와 500 epoch의 합이며, correct-state 잔여 업데이트 외에도 타이밍이 learning 방향 자체를 바꿀 수 있음을 보여준다. Teacher 단독 지터에서는 이러한 prediction pair 부호 변화가 없었다.
- **Teacher 사건:** 대부분의 실행에서 target=1 teacher spike가 모두 생성되었다. 전체 실행에서 최저 신뢰도도 약 99.976%였으므로 이 범위의 성능 저하는 주로 teacher spike 소실 하나로 설명되지 않는다. 입력 도착시각 변화가 prediction을 바꾸는 효과와, 실제 spike pair의 타이밍 변화가 correct-state에서도 가중치를 움직이는 효과가 함께 나타난다.
- **데이터셋 비교:** 시험한 모든 σ=0.5 조건에서 Modified의 평균 최종 정확도가 Original보다 높았다. 이것은 두 고정 데이터셋에서의 관찰이며 새로운 데이터셋 전반에 대한 일반화는 아니다.

Part B는 clean input으로만 학습하고 clean, teacher-free, jitter-free Clocked LIF로 평가했다. 입력 bit flip, capacity, STDP 파라미터 탐색은 수행하지 않았다. 다음 단계의 핵심 질문은 correct 1/1 출력에서 지터성 잔여 plasticity가 누적되는 문제를 실제 회로가 어떻게 억제할 수 있는가이다. 이 보고서는 그 억제 기전을 제안하거나 검증하지 않는다.
