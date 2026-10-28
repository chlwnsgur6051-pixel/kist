# 대표 그림

| 파일 | 읽는 법 | 용도 |
|---|---|---|
| `results/figures/validated_stdp_window.png` | 2 ms 예측 LTD, 6 ms teacher LTP | 원리 설명 |
| `results/figures/four_case_spike_events.png` | 실제 네 가지 target/prediction event | 메커니즘 설명 |
| `results/figures/k15_capacity.png` | 사용자 정의 v2 K≤15 결과 | 기준 성능 |
| `results/figures/k20_capacity.png` | 최종 사용자 지정 K≤20 결과, best와 final 구분 | 용량 분석 |
| `results/figures/k20_rank_capacity.png` | 입력 rank와 정확도/최적 readout | 수학적 가능성과 학습 차이 |
| `results/figures/original_k20_learning.png`, `modified_k20_learning.png` | epoch별 실제 STDP 정확도 | 간섭·순서 효과 |
| `results/figures/original_noise.png`, `modified_noise.png` | clean weight를 고정한 recall 입력 flip | 강건성 |
| `results/figures/jitter_accuracy.png` | 학습 시 spike timing jitter, 최종 clean recall | 시각 민감도 |

그림은 원본 결과에서 그대로 복사했다. 원본 경로와 SHA-256은 `artifact_manifest.csv`에 있다. Noise와 jitter의 조건·seed는 `experiments/` 아래 보관된 원본 methods 문서로 확인한다. K20의 최적 readout은 STDP로 학습된 가중치가 아니다.
