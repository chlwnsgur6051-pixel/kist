# Spike-Timing Correction — 공식 Baseline v2 (2026-11-12)

[baseline_v1](../baseline_v1/README.md)을 대체하는 현재 공식 기준 모델이다. Step과 Exponential 두 전류 모델을 함께 정의한다. v1은 이전 실험의 재현을 위해 그대로 보존했다. 이 폴더의 코드는 legacy 모드로 v1을 비트 단위로 재현할 수도 있다.

## 1. 연구 목적과 구조

- **목적:** 15개 입력과 15개 출력을 잇는 단일 feedforward 가중치 행렬에, STDP만으로 입력 패턴 → 목표 패턴 연상(heteroassociation)을 저장한다.
- **구조:** 15 input × 15 output, 입력과 목표는 0/1이다. 가중치는 0에서 시작하고 부호를 허용한다.
- **뉴런:** 출력은 Natural LIF다. 막전위가 threshold를 실제로 넘을 때만 발화한다(dt 0.1 ms, 막 시정수 2 ms, threshold 1, reset 0, 불응기 1 ms).
- **감독 신호:** 학습 때만 target=1 출력에 teacher 전류를 넣어 0 ms에 실제 LIF spike를 만든다. Recall에는 teacher가 없다.
- **입력:** 값이 1인 입력만 1.0 ms에 발화한다.
- **연속 상태:** 막전위, 불응기, 전류, STDP spike history를 association과 epoch 사이에 초기화하지 않는다.

## 2. Step 모델과 Exponential 모델

| | Step | Exponential |
|---|---|---|
| CS 전류 | `I0 = Σ x_i W_ij`, [1.0, 4.5) ms 동안 일정 | `I0·exp(-(t-1)/8 ms)`, 1.0 ms부터 자연 감쇠, cutoff 없음 |
| 학습률 λ | 0.5 | 0.25 |
| Association 시작 간격 | 15.0 ms (trial 10.1 ms + gap 4.9 ms) | 100 ms (trial 10.1 ms + gap 89.9 ms) |
| Weight 범위 | [-1, 1] | [-1, 1] |
| 설정 | [configs/step_v2.json](configs/step_v2.json) | [configs/exponential_v2.json](configs/exponential_v2.json) |

그 외 LIF, teacher, 입력 시각, STDP window는 두 모델이 같다.

## 3. Spike-Timing Correction 학습 규칙

- **STDP window:** Mexican hat `F(Δt) = C (1 − (Δt/s)²) exp(−Δt² / (2s²))`, `s = 1.286893592 ms`, `C = 0.170690168`, `Δt = t_post − t_pre`.
- **Pair:** 실제로 발생한 모든 pre/post spike pair에 `λ·F(Δt)`를 더한다(all-to-all, 50 ms history).
- **동작 원리:**
  - teacher spike(0 ms)와 입력(1.0 ms)의 pair는 window의 양(+) 구간에 들어가 가중치를 키운다.
  - 가중치가 커지면 출력이 teacher 없이도 자연 발화한다. 이 자연 spike가 입력과 만드는 pair는 발화 시각에 따라 부호가 달라진다. 입력 직후(약 1.3 ms 이내)면 양(+), 그보다 늦으면 음(−)이다.
  - 자연 spike 시각이 약 4 ms가 되면 양과 음의 기여가 상쇄되어 갱신이 0에 가까워진다. 발화가 이보다 빠르거나 늦으면 그 차이만큼 가중치가 보정된다.
- **적용 시점:** association interval(trial + gap)이 끝날 때 `W = clip(W + ΔW, −1, 1)`을 한 번 적용한다.

자세한 시간 규약은 [MODEL_SPEC.md](MODEL_SPEC.md)에 있다.

## 4. v1에서 v2로 바뀐 것

| 항목 | v1 | v2 |
|---|---|---|
| Step association 시작 간격 | 15.1 ms (trial 101 + gap 50 tick) | **15.0 ms** (trial 101 + gap 49 tick) |
| Weight 범위 | 제한 없음 | **W ∈ [-1, 1]** |
| Gap 구간 LIF | 막전위 적분만 하고 threshold/spike/reset 없음. 불응기를 1 tick 늦게 판정 | **trial과 같은 규칙.** threshold 통과 시 natural spike, reset, 불응기. gap spike도 STDP post event |

STDP window, 학습률, teacher·입력 시각, 전류 크기, dataset, 제시 순서는 바꾸지 않았다. 변경 근거는 [CHANGELOG.md](CHANGELOG.md)에 있다.

## 5. Dataset과 제시 순서

- `datasets/original_K10.npz`, `datasets/modified_K10.npz`는 v1 보관본과 바이트 단위로 같은 파일이다. 새로 생성하지 않았다.
- 각 파일에는 `input_binary`(10×15), `target_binary`(10×15), `presentation_orders`(1000×10)가 있다. `.csv`는 사람이 읽기 위한 입력/목표 사본이다.
- **제시 순서는 epoch마다 다른 순열이다.** 고정 순서가 아니다.
  - `numpy.random.default_rng(0)`으로 `permutation(10)`을 1000번 뽑아 미리 저장한 것이다. 1000개 행이 모두 다르고, `0,1,…,9` 순서인 행은 없다.
  - Original과 Modified가 같은 순서 배열을 쓴다. v1도 같은 배열을 썼다.
  - 1000 epoch을 넘으면 같은 1000개 행을 처음부터 다시 쓴다(epoch 1001은 epoch 1의 순서).
  - 첫 다섯 epoch: `4 6 2 7 3 5 9 0 8 1` / `2 9 3 6 0 4 8 7 5 1` / `5 4 9 0 8 2 1 6 7 3` / `4 5 7 6 2 3 1 9 0 8` / `1 2 7 8 0 4 9 5 6 3`
- 이 순서는 검증된 baseline 조건의 일부이므로 그대로 둔다. 다른 순서(예: 모든 epoch에서 `0,1,…,K-1`)는 `train()`에 순서 배열을 넘겨 실험 설정으로 지정한다(7절).

## 6. 검증 결과 (Original / Modified, K=10, noise-free)

| Model | Dataset | 첫 10/10 | 연속 10/10 시작 | Epoch 1000 | Epoch 5000 | Clipping | Gap spike |
|---|---|---:|---:|---|---|---:|---:|
| Step v2 | Original | 119 (v1: 113) | 226 (v1: 226) | 10/10 | 10/10 | 0 | 0 |
| Step v2 | Modified | 21 | 37 | 10/10 | 10/10 | 0 | 0 |
| Exponential v2 | Original | 176 | 443 | 10/10 | 10/10 | 0 | 0 |
| Exponential v2 | Modified | 32 | 69 | 10/10 | 10/10 | 0 | 0 |

- 5000 epoch 동안 연속 10/10 시작 이후 하락은 없었다.
- 가중치는 [-1, 1]에 닿지 않는다(최대 |W|: Step 0.69 / 0.42, Exponential 0.75 / 0.50). 따라서 bound는 이 조건의 학습에 영향이 없다.
- Exponential v2의 결과는 이전 100 ms Exponential 모델과 epoch별로 같다.
- Step v2와 v1의 차이는 시작 간격 0.1 ms에서만 온다. Original의 첫 10/10이 6 epoch 늦어졌다.
- Gap spike 처리와 clipping이 실제로 일어나는 조건의 검증은 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)에 있다.

## 7. 실행

필요한 것은 Python, numpy, C 컴파일러(gcc)다. 버전은 [requirements.txt](requirements.txt)에 있다. C 소스(`src/stc_core.c`)는 첫 실행 때 `gcc -O2 -fno-fast-math -ffp-contract=off`로 `build/`에 컴파일된다. `build/`와 `runs/`는 Git에서 제외한다.

이 폴더에서 실행한다.

```bash
# 저장된 결과와 대조 (1000 epoch 4개 조건, 약 7분)
python scripts/reproduce.py
python scripts/reproduce.py --epochs 5000          # 5000 epoch 전체 대조
python scripts/reproduce.py --legacy-v1            # 같은 코드로 v1 재현 (../baseline_v1 필요)

# 학습 + 매 epoch recall
python scripts/train.py --model step --dataset Original --epochs 1000
python scripts/train.py --model exponential --dataset Modified --epochs 1000 --save runs/exp_modified.npz

# 저장된 가중치로 recall만
python scripts/recall_checkpoint.py

# 빠른 자체 점검, 약 2–3분 (dataset, 상수, 첫 60 epoch, 저장 가중치, gap spike·clipping)
python tests/test_baseline_v2.py
```

코드에서 직접 쓸 때:

```python
import sys; sys.path.insert(0, "research/spike_timing_correction/baseline_v2/src")
import stc_model as M
x, y, orders = M.load_reference_dataset("Original")
rows, w, state, extra = M.train(M.STEP_V2, x, y, orders, 1000)   # rows: epoch별 recall·weight·clipping·gap 지표
result = M.recall(M.STEP_V2, x, y, w, state)                     # teacher 없는 recall
```

`orders`는 `(epoch 수, K)` 배열이며 epoch `e`는 `orders[(e-1) % len(orders)]`를 쓴다. 한 행만 넘기면 모든 epoch에서 그 순서를 반복한다.

**의존 관계:** 모델(`stc_core.c`, `stc_model.py`), `reproduce.py`, `train.py`, `recall_checkpoint.py`는 이 폴더만으로 실행된다. `src/reference_model.py`와 `--legacy-v1`, 테스트의 다섯 번째 항목만 `../baseline_v1`을 읽는다.

## 8. 폴더

- `src/stc_core.c`: tick 단위 LIF 루프와 all-pairs STDP 이벤트 루프.
- `src/stc_model.py`: 학습·recall 일정, weight bound, 지표. `STEP_V2`, `EXP_V2`, `STEP_V1`, `EXP100_V1` 정의.
- `src/reference_model.py`: v2의 독립 순수 Python 참조 구현. STDP는 v1 보관본의 `ContinuousAllPairs`를 그대로 사용한다.
- `configs/`: 두 모델의 전체 파라미터.
- `datasets/`: Original/Modified K=10의 입력, 목표, 제시 순서.
- `results/csv`, `results/checkpoints`: 5000 epoch 기록과 epoch 1000·5000 가중치.
- `results/validation`: 채택 검증의 요약 CSV 사본.
- `scripts/`, `tests/`: 재현, 학습, recall, 자체 점검.
- `artifact_manifest.csv`: 보관 파일의 SHA-256.

## 9. 이전 결과가 사용한 Baseline

- `research/spike_timing_correction/baseline_v1`과 2026-11-12 이전의 `results/` 실험(지수 전류 실험, noisy training, capacity 포함)은 모두 **v1 규약**이다. 즉 15.1 ms, gap spike 없음, weight 제한 없음을 따른다.
- Exponential 실험은 v1의 gap 루프를 물려받았다. 어떤 과거 조건이 이 영향을 받았는지는 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)의 표에 있다.

## 10. Capacity 실험에서 아직 해결되지 않은 문제

v2는 Original/Modified K=10에서만 검증됐다. 무작위 dataset의 저장 용량은 아직 측정하지 않았고, 아래 문제가 남아 있다.

- **무작위 dataset K=10에서 학습 실패가 남는다.** 5개 dataset 중 Step v2는 3개만 10/10이다(나머지 2/10, 4/10). Exponential v2는 0–3/10이다. v1과 같은 결과다.
  - 오류는 모두 false positive다. 입력 직후의 이른 natural spike가 STDP window의 양(+) 구간에 들어가 가중치를 계속 키운다.
  - Weight bound는 발산을 숫자상 막을 뿐이다. 실패한 출력의 가중치는 +1에 붙은 채 매 epoch 잘린다.
  - 이 dataset들은 선형 분리 가능하므로, 실패는 표현 한계가 아니라 학습 규칙의 문제다.
- **제시 순서의 영향이 분리되지 않았다.** 지금까지의 모든 capacity 실행은 epoch마다 다른 순열을 썼다. 모든 epoch에서 `0,1,…,K-1`을 쓰는 실행은 아직 없다.
- **이전 capacity 결과는 v1 규약이다.** `results/exponential_synaptic_current/capacity`의 K=10–100 결과에는 gap 수정과 weight bound가 적용되지 않았다. v2로 다시 돌린 것은 K=10, 20의 예비 확인뿐이다.
- **Capacity 코드와 dataset이 Exponential 전용 폴더에 있다.** 새 capacity 실험은 별도 위치에서 v2 코드를 불러 쓰도록 구성해야 한다.

## 11. 한계

- 검증 범위는 Original/Modified K=10 결정론적 학습, Step membrane noisy training, random K=10 5개 dataset이다.
- 관측한 5000 epoch 안의 결과이며 무한한 학습 안정성을 뜻하지 않는다.
- noisy jitter Exponential 3개 실행은 수정된 gap으로 다시 돌리지 않았다.
- Teacher는 감독 신호이며, 소프트웨어 결과는 하드웨어 구현 가능성을 입증하지 않는다.
