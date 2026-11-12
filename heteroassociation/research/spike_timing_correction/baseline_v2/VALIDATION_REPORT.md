# Baseline v2 채택 검증 요약

이 문서는 채택 판단에 쓴 사실만 요약한다.

- **근거 파일:** 요약 CSV는 [`results/validation/`](results/validation)에 사본이 있다. 실행별 원자료와 그림을 포함한 전체 검증 기록은 `results/spike_timing_correction_baseline_v2_validation/`(저장소 최상위 기준)에 있으며, 용량 때문에 Git에는 넣지 않았다.
- **다시 확인하는 방법:** `python scripts/reproduce.py`(저장 결과 대조), `python tests/test_baseline_v2.py`(빠른 자체 점검). 2026-11-12에 다음을 확인했다.
  - 이 폴더만 따로 복사한 위치: `reproduce.py`(4개 조건, 1000 epoch), `recall_checkpoint.py`, 테스트 1–4번 항목 통과.
  - `baseline_v1`과 함께 복사한 위치: 테스트 전체(5번 항목은 gap spike 6837개, clipping 484회가 발생하는 조건에서 참조 구현과 일치)와 `reproduce.py --legacy-v1` 통과.
- **제시 순서:** 아래 모든 실행은 epoch마다 다른 순열을 썼다. Original/Modified와 random K=10은 `default_rng(0).permutation(10)` 1000행을 썼다. 고정 순서로 학습한 실행은 없다.

## 구현 동등성

- **legacy 모드:** 같은 코드로 v1을 비트 단위로 재현한다.
  - Step 1000 epoch: 모든 학습 spike 시각, epoch별 정확도, 최종 W.
  - Exponential 100 ms 5000 epoch.
  - 과거 noise-free Exponential 100개 조건, capacity 20개 실행, membrane-noise 실행.
- **v2 모드:** 독립 순수 Python 참조 구현과 association마다 spike, ΔW, W, recall이 비트 단위로 같다. gap spike와 clipping이 실제로 발생하는 실행도 포함한다.
- **Step 15.0 ms 타이밍:**
  - teacher spike는 local 0, 시작 간격은 150 tick이다.
  - 입력은 1.0 ms, CS 전류는 [1.0, 4.5) ms에만 흐른다.
  - 연속 상태는 association 사이에 유지된다.

## Original / Modified (K=10)

| Model | Dataset | 첫 10/10 | 연속 10/10 시작 | Ep 1000 | Ep 5000 | 최대 \|W\| | Clipping | Gap spike |
|---|---|---:|---:|---|---|---:|---:|---:|
| Step v1 | Original | 113 | 226 | 10/10 | 10/10 | 0.691 | — | — |
| Step v2 | Original | 119 | 226 | 10/10 | 10/10 | 0.685 | 0 | 0 |
| Step v1/v2 | Modified | 21 | 37 | 10/10 | 10/10 | 0.415 | 0 | 0 |
| Exponential 이전/v2 | Original | 176 | 443 | 10/10 | 10/10 | 0.748 | 0 | 0 |
| Exponential 이전/v2 | Modified | 32 | 69 | 10/10 | 10/10 | 0.502 | 0 | 0 |

5000 epoch 동안 연속 10/10 시작 이후 하락은 없었다. 마지막 100 epoch의 갱신 norm은 10⁻¹⁰ 수준이다.

## Membrane noisy training (Step, 3회 × 200 recall trial)

v2는 이전 결과를 유지한다.

| 조건 | 모델 | low2 recall exact / full |
|---|---|---|
| Original σ low1 | 이전 | 98.72 / 87.83% |
| Original σ low1 | v2 | 98.72 / 88.17% |
| Modified σ low2 | 이전 | 99.27 / 93.0% |
| Modified σ low2 | v2 | 99.27 / 93.0% |

Original noisy 실행 중 이전에 weight가 568로 발산하던 실행은 v2에서 output 0 열이 +1에 포화된다. 이 뉴런은 모든 target이 1이라 오류로 이어지지 않는다.

## 과거 Exponential 결과에 대한 gap 처리 영향

| 범주 | 판정 |
|---|---|
| 15.1 ms full 전류 19개 조건 (초기 실험 τ1 charge·τ2·τ4, peak_only τ2·τ4, tau_extension τ5·τ16·τ8 Original) | **영향받음.** 모두 weight 발산으로 실패한 조건이다. 수정 시 weight 경로가 크게 달라지지만 epoch 1000 정확도(0–1/10)는 같다. 동역학 수치는 재실험 대상이다. |
| truncated 전부, tau_24_64, learning_rate_duration stage1·2, long_term 40·100 ms, noisy membrane/current | 영향 없음(gap 초과 0) |
| capacity Exponential(100 ms) | gap spike는 생기지만 STDP window 밖(평균 25 ms)이다. 수정 전후 W 차이 ≤ 6×10⁻⁹, 정확도는 같다. |
| noisy jitter Exponential 3개 | gap 초과 있음. 100 ms라 영향이 작을 것으로 예상하지만 미검증이다. **재실험 대상.** |

## Random K=10과 bound의 역할

- **결과:** Step v2는 5개 중 3개 dataset에서 10/10이다. 실패한 2·4번은 v1과 같은 2/10·4/10이다. Exponential v2는 0–3/10으로 이전과 같다.
- **오류의 형태:** 모두 FP다. 실패한 output 열은 +1에 포화된 채 매 epoch 바깥쪽 갱신이 잘린다(시도 갱신 norm 0.5–1.7, 실제 0–0.06).
- **bound의 역할:** 발산을 숫자상 막을 뿐, 수렴시키거나 학습 실패를 해결하지 않는다.
- **원인:** 조기 natural spike가 Mexican-hat 양(+) 구간에 들어가 생기는 양의 STDP다. 이 dataset들은 선형 분리 가능하다.
