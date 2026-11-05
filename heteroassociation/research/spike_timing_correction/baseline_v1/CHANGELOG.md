# 변경 기록

## baseline_v1 — 2026-11-05

- 공식 모델 명칭: **Spike-Timing Correction**.
- 검증된 signed-weight Natural LIF, 실제 0 ms teacher spike, 실제 natural output spike, Generic All-to-All Pair-Based Mexican-hat STDP를 원본 소스 그대로 보관했다.
- 조건: Original·Modified K=10, CS `[1.0,4.5)` ms, current cap 없음, `λ=0.5`, 초기 W=0, 1000 epoch, association 사이 5 ms 휴지, 연속 LIF·STDP state.
- 관측 결과: Original 첫 10/10 epoch 113, 지속 시작 226; Modified 첫 21, 지속 시작 37; 두 조건 모두 epoch 1000에서 10/10.
- 기존 Basic Correction과 달리 STDP에 natural output의 **실제** spike time을 사용한다. 고정 4 ms 가상 post-spike를 학습에 쓰지 않는다. 이 비교는 모델 정의의 차이이며, 이 문서에서 새 비교 실험을 수행한 것은 아니다.
- 기존 `research/continuous_stdp_heteroassociation` 및 일반 작업 디렉토리의 모델·결과를 수정하지 않았다. 이후 변화는 이 폴더를 덮어쓰지 않고 별도 버전으로 보관한다.
