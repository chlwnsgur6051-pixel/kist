# 확인된 한계와 후속 질문

Clocked LIF 예측 시각과 teacher 자극 시각은 외부에서 제어된다. 자연 threshold-crossing LIF의 예측 latency·다중 발화를 그대로 구현한 자율 시스템은 아니다. Teacher 입력은 목표 정보를 사용한다. Signed weight는 효과적 흥분·억제 전류 추상화다.

K가 증가하면 일부 association의 순차 학습이 다른 association을 다시 손상시키며 best epoch와 final epoch가 달라질 수 있다. K20 Original은 best 60%에서 final 55%, Modified는 75%에서 70%로 감소했다. 두 경우 모두 LP 최적 readout보다 낮으므로 일부 차이는 학습 제한이다.

입력 noise 결과는 clean 학습 후 recall 입력 bit flip에 대한 취약성을 보여준다. Timing jitter는 정답 11 case의 cancellation 잔차와 LIF event 변화가 누적될 수 있음을 보여준다. Hardware 후보에서 clock, teacher-phase 분리, signed synaptic efficacy, 맞춤형 STDP 창, 정밀한 상쇄와 동기화가 가능해야 한다. 이 소프트웨어 결과만으로 특정 칩의 API·정밀도·스케줄링 지원을 입증하지 않는다.

후속 연구는 현재 모델의 clock·teacher event가 실제 하드웨어에서 재현되는지, weight precision 및 timing error가 어떤 제한을 만드는지 검증하는 것이다. 새로운 학습 규칙이나 자연 LIF 성공을 이 아카이브의 검증 성과로 기록하지 않는다.
