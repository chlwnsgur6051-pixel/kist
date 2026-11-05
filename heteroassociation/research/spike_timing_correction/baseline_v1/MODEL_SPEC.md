# 모델 명세 — Spike-Timing Correction baseline_v1

아래는 [원본 코드 스냅샷](src/sweep_model.py)과 지정 조건 `CS_END_TICK=45`, `LAMBDA=0.5`, `CAP=None`의 실제 동작이다. 원본 파일의 전역 기본값 `CS_END_TICK=40`, `LAMBDA=1.0`은 sweep에서 조건별로 설정되던 초기값이며, **공식 baseline 조건이 아니다**. [재현 wrapper](scripts/reproduce.py)가 검증된 4.5 ms·0.5 조건을 명시적으로 설정한다.

## 구조와 입력

`X,Y∈{0,1}^{10×15}`, `W∈R^{15×15}`. 출력별 원시 CS 전류는 `I_j=Σ_i x_i W_ij`이며 별도 bias·gain·cap이 없다(gain=1). `x_i=1`이면 1.0 ms에 pre-spike 하나를 기록하고, `0`이면 기록하지 않는다. CS step current는 `1.0≤t<4.5 ms`에 `I_j`, 그 밖에 0이다. 음수 `I_j`는 그대로 적분에 들어간다. 가중치는 모두 0에서 시작하며 하한·상한·정규화가 없다.

## Natural LIF와 teacher

시간 간격은 `dt=0.1 ms`, 막전위 시정수는 `τ_m=2 ms`, 임계값은 `1`, reset은 `0`, 불응기는 `1 ms`다. 활동 가능한 적분 구간의 Forward Euler 식은

`V_j(t+dt)=V_j(t)+(dt/τ_m)[−V_j(t)+I_CS,j(t)+I_teacher,j(t)]`.

Target `y_j=1`인 뉴런에만 `[-0.1,0)` ms 한 적분 구간 동안 teacher current `30`을 넣는다. 초기 0 V에서는 해당 구간 종료 시 V=1.5>1로 실제 LIF spike가 0 ms에 발생한다. Target=0에는 teacher current가 없다. 활동 가능한 뉴런의 적분 후 전압이 **임계값을 초과할 때(`>`, `≥`가 아님)** spike를 기록하고 즉시 0으로 reset한다. Spike timestamp는 적분 구간의 종료 시각이며, `k=0`이 0 ms, `k=10`이 1 ms, `k=45`가 4.5 ms다. 불응기 동안 전압은 0에 고정되고 CS 전류원은 그대로 존재하지만 막전위 적분에는 반영하지 않는다. Teacher spike도 같은 출력 뉴런의 실제 post-spike이므로 reset과 refractory가 적용된다. 학습 분석에서 `k>0`인 spike가 natural prediction에 해당하며 별도 강제 4 ms spike를 만들지 않는다. 추가 natural spike도 코드상 허용된다.

## STDP

`Δt=t_post−t_pre`, `u=Δt/s`, `s=1.286893592 ms`, `C=0.170690168`에 대해

`F(Δt)=C(1−u²)exp(−u²/2)`.

한 association의 기본 pair update는 `Δw_ij=λΣ_preΣ_post F(t_post−t_pre)`, `λ=0.5`다. **동일한 F**를 teacher-induced spike와 natural spike 모두에 적용한다. 예를 들어 0 ms teacher, 1 ms pre, 4 ms natural spike라면 `λF(−1)≈+0.025`와 `λF(3)≈−0.025`가 상쇄된다. Natural spike가 3.4 ms면 `λF(2.4)≈−0.03716`으로 순 LTD가 남고, 4.5 ms면 `λF(3.5)≈−0.01352`로 순 LTP가 남는다. 이는 **그 trial 내 pair만**의 수치다.

Mexican-hat 커널은 `F(Δt)=F(−Δt)`인 짝함수다. LTD/LTP 부호는 `|Δt|`가 `s`보다 작은지 큰지에 따라 바뀌며, 일반적인 시간 순서 비대칭 STDP로 설명하면 잘못된다. 큰 `|Δt|`에서는 기여가 다시 0으로 간다. 따라서 실제 발화 시각에 따라 어떤 방향의 보정이 남을지는 이 커널의 위치를 확인해야 한다.

구현은 이벤트 도착 시 그전에 발생한 반대 종류 spike와 pair를 한 번씩 계산하는 `ContinuousAllPairs`다. 같은 시각의 pre/post 이벤트는 **pre를 먼저** 처리한다. LIF 상태와 spike history는 association 경계에서 리셋되지 않는다. 각 association의 누적 `ΔW`는 그 association의 LIF 발화가 끝난 **뒤** W에 한 번 반영된다. 따라서 같은 trial의 teacher-induced LTP가 그 trial의 natural spike 발생을 즉시 바꾸지 않는다. `WINDOW_HISTORY_TICKS=500`(50 ms)보다 오래된 spike 기록은 만료한다. 원본 코드의 설명대로 이 범위 밖 커널 값은 float64에서 0으로 언더플로한다. Association 처리 101 tick 뒤 추가로 50 tick(5 ms) 동안 외부 전류 없이 막전위를 계속 진화시킨다. 이 기간과 epoch 경계에서도 상태를 강제로 초기화하지 않는다.

## 학습 순서와 recall

각 epoch의 10개 association 제시 순서는 [NPZ](datasets/original_K10.npz)와 참조 ZIP의 seed-0 순서다. Original과 Modified는 각각 저장된 순서를 사용하며, 동일한 패턴·순서가 1000 epoch에 걸쳐 적용된다. 학습 뒤 epoch별 readout은 **학습 당시 막전위·불응기 상태의 복사본**으로 수행한다. Teacher를 제거한 채 association 0–9를 순차적으로 제시하고 각 cue 사이에도 5 ms gap을 둔다. 이 probe는 W·STDP history·실제 학습 state를 바꾸지 않는다. 각 뉴런에서 자연 spike가 한 번 이상 발생하면 bit 1, 없으면 0이다. 15 bit 모두 목표와 같아야 해당 association을 exact correct로 센다. Bit accuracy는 150개 bit 중 일치 비율이다.

## 해석의 경계

이 소프트웨어는 양·음의 가중치, float64, 동기화된 teacher pulse, step current, spike timestamp grid, presentation 종료 후 지연 업데이트를 사용한다. 하드웨어에서 동일한 전류·정밀도·지연 적용을 지원한다는 검증은 없다. Correct target/prediction이라도 실제 natural timing이 4 ms와 다르거나 이전 association spike pair가 남으면 총 update가 정확히 0이라는 보장은 없다.
