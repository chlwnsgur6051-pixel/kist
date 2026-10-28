# 모델 구조

입력 15개, 출력 current-based LIF 15개, 학습 가능한 15×15 feedforward 가중치 한 층이다. 입력 bit 1만 2 ms에 spike를 만든다. 입력 bit 0은 발화하지 않는다. Signed effective weight는 흥분·억제 전류를 허용하는 모델 추상화다. 초기 effective weight는 모두 0이다.

막전위는 `dt=0.1 ms`, `τ_m=2 ms`, synaptic current `τ_s=1 ms`, rest/reset 0, refractory 1 ms로 적분한다. Input gain=1, threshold=0.01이다. 예측 발화는 4 ms에서만 막전위가 threshold를 **초과**할 때 발생한다. Target=1에는 8 ms에 teacher current 30을 주고, 실제 threshold spike가 발생해야 STDP에 들어간다. 예측은 teacher보다 먼저 결정된다. 자세한 실행 순서는 `src/model.py`와 `configs/validated_model.json`에 고정되어 있다.

Recall은 teacher 인자 없이 `src/recall.py`에서 시행한다. 4 ms의 Clocked LIF 발화가 출력이다. `xW>0`의 정적 판정과는 threshold가 달라 같은 결과라고 가정하지 않는다. 이 모델은 외부 prediction clock 및 target-dependent teacher 신호를 사용하며 자연 연속시간 발화형 자율 LIF 학습이 아니다.
