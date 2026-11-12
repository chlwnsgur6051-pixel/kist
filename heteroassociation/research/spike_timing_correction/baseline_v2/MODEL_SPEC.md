# Baseline v2 모델 명세

## 시간 축

- dt = 0.1 ms, Forward Euler. association 시작 tick을 `base`라 할 때 local tick `k`는 구간 `[base+k-1, base+k)`를 적분한다.
- `k = 0`: teacher 구간 [-0.1, 0) ms. target=1 출력에만 전류 30을 넣어 실제 LIF spike를 0 ms에 만든다.
- `k = 1..100`: 응답 구간 0.1–10.0 ms. 입력 spike는 1.0 ms(local tick 10)에 있다.
- `k = 101..interval-1`: gap. Step은 49 tick(4.9 ms), Exponential은 899 tick(89.9 ms)이다.
- 시작 간격은 Step 150 tick = **15.0 ms**, Exponential 1000 tick = 100 ms이다.
- 연속된 teacher spike 시작 간격은 정확히 150 tick이다(검증됨).

## LIF (모든 tick 동일)

```
active = (base + k - 1) >= refractory_until
if active:  v += 0.05 * (-v + I_cs + I_teacher)
else:       v = 0
if active and v > 1:  spike at k;  v = 0;  refractory_until = base + k + 10   (1 ms)
```

- gap tick도 위 식을 그대로 쓴다. v1은 gap에서 threshold 검사를 하지 않았고, refractory를 `base+k`로 판정했다(1 tick 차이).
- **Step:** `I_cs = Σ_i x_i W_ij`, k = 11..45 ([1.0, 4.5) ms), 그 외 0. gap 전류가 0이라 gap에서 threshold에 도달할 수 없다.
- **Exponential:** k = 11에 `tail += Σ_i x_i W_ij`를 더하고, 매 tick 끝에 `tail *= exp(-0.1/8)`을 적용한다. tail은 association 경계를 넘어 이어진다.

## STDP와 weight 적용

- `F(Δt) = C (1 - (Δt/s)^2) exp(-Δt^2 / (2 s^2))`, s = 1.286893592 ms, C = 0.170690168. pair 기여는 `λ F(t_post - t_pre)`이다.
- **처리 단위:** association interval(trial + gap)의 모든 실제 post spike(teacher, 응답 구간, gap)와 입력 pre spike를 시간순으로 처리한다. 같은 시각이면 pre가 먼저다.
- **pair 구성:** 각 새 event는 50 ms history 안의 모든 반대 종류 event와 pair를 만든다. 이는 v1 `ContinuousAllPairs.process`와 같은 순서·규칙이다.
- **적용 시점:** interval이 끝날 때 `W = clip(W + ΔW, -1, 1)`을 한 번 적용한다.
  - gap 동안 W는 쓰이지 않는다. Step 전류는 trial 안에서만 흐르고, Exponential 진폭은 k = 11에 이미 정해지기 때문이다.
  - 그래서 v1처럼 trial 끝(gap 전)에 적용해도 동역학은 같다.
- **연속 상태:** LIF 막전위, refractory, 전류 tail, STDP history는 association과 epoch 사이에 초기화하지 않는다.

## Recall

- 매 epoch 학습이 끝난 뒤, 학습 상태의 **복사본**에서 teacher 없이 10개 cue를 순서대로 제시한다. 학습 상태와 W는 바뀌지 않는다.
- 출력 bit는 응답 구간(k = 1..100)에 spike가 있으면 1이다.
- gap spike는 상태(reset, refractory)에 반영하고 별도로 집계한다.
- Exact Recall은 15 bit가 모두 맞은 association의 수, Bit Accuracy는 맞은 bit의 비율이다.

## Dataset과 제시 순서

- 입력·목표는 `datasets/{original,modified}_K10.npz`의 `input_binary`, `target_binary`(각 10×15, 0/1)다. v1 보관본과 같은 파일이다.
- 학습 순서는 같은 파일의 `presentation_orders`(1000×10)다. **epoch마다 다른 순열**이며 `numpy.random.default_rng(0).permutation(10)`을 1000번 뽑아 저장한 값이다.
- Original과 Modified, v1과 v2가 모두 같은 순서 배열을 쓴다. epoch `e`는 `orders[(e-1) % 1000]`을 쓴다.
- Recall의 cue 순서는 항상 `0,1,…,9`다.

## 수치 구현

- C(`gcc -O2 -fno-fast-math -ffp-contract=off`)의 연산 순서는 v1 Python 구현과 같다.
- **legacy 모드:** v1의 spike, weight, 정확도를 1000 epoch 동안 비트 단위로 재현한다(Step v1, Exponential 100 ms 5000 epoch).
- **v2 모드:** 독립 Python 참조 구현과 association마다 spike, ΔW, W, recall이 비트 단위로 같다.
