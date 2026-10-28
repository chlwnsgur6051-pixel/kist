# Methods

검증 패키지 `20261027_122711`의 clean 500-epoch 최종 weight를 읽어 freeze했다. 별도의 검증 스크립트로 동일 설정의 clean 학습을 다시 실행하고 **모든 500개 epoch checkpoint와 최종 weight가 두 데이터셋 모두 최대 원소 차이 0**임을 확인했다(`csv/clean_training_replay_validation.csv`). 따라서 noisy recall에는 검증된 보관 weight를 사용했다. Noisy input으로 재학습하지 않았다. Dataset A는 사용하지 않았다.

입력은 15-bit binary이다. 각 digit, 각 k=0..4에 대해 mixed는 15개 자리 중 k개, 1→0은 활성 자리 중 k개, 0→1은 비활성 자리 중 k개를 **모두 조합**했다. 조합별 teacher-free Clocked LIF의 원본 Euler 적분 순서와 threshold 0.01을 사용했다. Prediction은 4 ms 시각의 막전위 > threshold로 결정한다. 깨끗한 10개 입력의 출력과 원본 `simulate_trial` 10회 출력이 일치함을 확인했다. No random sampling; seed 없음.

추가로 무작위로 선택한 40개 mixed-noise cue에 대해 vectorized readout 결과와 원본 `simulate_trial()`의 teacher-free 출력이 모두 일치했다. 이 검증에는 재현 가능한 선택 seed 19를 사용했다. 전체 recall 집합의 선택에는 난수를 사용하지 않았다.

Metric 단위는 noisy cue 1개이다. Exact=15개 output bit 전부 일치; bit accuracy=15개 중 일치 비율. 각 type/k의 전체 비율은 가능한 모든 조합을 합친 micro-average이다. 각 digit의 조합 수가 noise type별로 다르므로 조건 간 차이는 가능 cue 집합의 차이도 반영한다. 가능한 비활성 bit가 k개보다 적은 경우의 0→1 조건은 미측정(NA)이며 heatmap에서 회색이다.

Bitmap 예시는 매 digit/type/k에서 조합 나열의 중간 인덱스를 사전 결정해 선택했다. Recall 성공 여부로 예시를 선별하지 않았다. 5×3 bitmap, row-major 비트 순서를 사용한다. 원본 데이터는 변경하지 않았다.
