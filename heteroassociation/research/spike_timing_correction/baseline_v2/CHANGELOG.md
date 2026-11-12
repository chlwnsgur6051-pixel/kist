# CHANGELOG

## baseline_v2 — 2026-11-12

v1([../baseline_v1](../baseline_v1/README.md))에서 세 가지를 바꿨다. v1의 코드, 데이터, 결과는 그대로 보존했다.

1. **Gap 구간 LIF 수정**
   - v1과 이전 Exponential 코드는 association 사이 gap에서 막전위를 적분만 하고, threshold 검사·spike·reset·STDP를 하지 않았다.
   - 또 gap의 refractory를 trial보다 1 tick 늦은 시점으로 판정했다.
   - v2는 gap tick에도 trial tick과 같은 LIF 규칙을 적용하고, gap spike를 STDP post event로 처리한다.
   - Step은 gap 전류가 0이라 영향이 없다. Exponential 100 ms의 Original/Modified 학습도 gap spike가 0이라 영향이 없다.
2. **Step 시작 간격 15.1 → 15.0 ms**
   - gap을 50에서 49 tick으로 줄였다. trial(101 tick)과 teacher 구간은 그대로다.
   - Original의 첫 10/10이 113 → 119로 바뀌었다. 연속 10/10 시작(226)과 최종 결과는 같다. Modified는 같다.
3. **Weight bound [-1, 1]** (Step, Exponential 공통)
   - association interval이 끝날 때 `W = clip(W + ΔW, -1, 1)`을 적용한다.
   - Original/Modified 정상 학습에서는 bound에 닿지 않는다(최대 |W| 0.75).

### 바꾸지 않은 것

- STDP window, 학습률, teacher·입력 시각, 전류 크기.
- Original/Modified 입력·목표와 제시 순서(epoch마다 다른 seed-0 순열). `datasets/`는 v1과 바이트 단위로 같다.

### 이전 결과의 Baseline 표기

- 2026-11-12 이전 `results/`의 모든 Spike-Timing Correction 실험은 v1 규약(15.1 ms, gap spike 없음, 제한 없음)이다.
- Exponential 실험 중 gap 처리의 영향을 받은 조건은 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)에 정리했다.
