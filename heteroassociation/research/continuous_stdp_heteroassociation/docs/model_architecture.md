# 모델 구조와 시뮬레이션 시각

- 입력 15개, Natural LIF 출력 15개, 가소성 가중치 행렬 15×15. 추가 학습층은 없다.
- 입력 bit 1만 1 ms에 pre-spike를 낸다. bit 0은 silent다.
- 학습 Target bit 1은 `[-0.1, 0)` ms 한 timestep 동안 teacher current 30을 받아 실제 LIF post-spike를 0 ms에 낸다. Target bit 0에는 teacher current가 없다.
- CS step current는 `[1, 4)` ms 동안 각 출력의 `min(Σ_i x_i W_ij, 1.282590743781507)`이다. 음수 전류는 signed weight의 결과로 그대로 유지한다.
- dt 0.1 ms, 막 시정수 2 ms, threshold 1, reset 0, refractory 1 ms. Euler 적분 후 역치 통과를 구간 종료 시각의 spike로 기록한다. Clock으로 출력 spike를 생성하지 않는다.
- 각 trial은 `[-0.1, 10.0]` ms의 101개 적분 구간이다. 추가 5 ms 무입력 간격 후 다음 association이 시작된다. 막전위와 refractory 만료 시각은 다음 trial로 전달한다.
- Teacher 없는 recall은 저장된 원래 입력만 사용한다. Sequential probe는 학습 시점 상태의 **복사본**에서 10개 cue를 순차적으로 시험한다. 이 probe는 학습 timeline과 STDP 이력에 삽입되지 않는다. Independent probe는 cue마다 초기 정지 상태로 시작하는 기존 평가 방식이다.

실제 관측에서 모든 자연 출력 spike는 명목상 4 ms였다. 이 결과는 현재 current cap·입력 시간·데이터에 한정된다.
