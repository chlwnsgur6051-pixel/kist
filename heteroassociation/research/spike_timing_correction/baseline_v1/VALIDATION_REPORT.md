# 보관본 검증 기록 — 2026-11-05

## 소스·설정

- 기준 소스: `results/timing_correction_cs_duration_lr_sweep/scripts/sweep_model.py`, `run_sweep.py`의 바이트 동일 복사본. `artifact_manifest.csv`의 SHA-256으로 확인했다.
- 적용 조건: `CAP=None`, `CS_END_TICK=45`, `LAMBDA=0.5`, Original·Modified K=10, 1000 epoch. 소스의 기본 전역값과 **재현 wrapper가 적용한 확정 조건**을 구분했다.
- 원본 reference ZIP의 input/target/order 배열과 보관 NPZ가 요소별로 일치했다. Pattern을 재작성하거나 다시 생성하지 않았다.

## 구현 단위 검증

`python -m unittest discover -s research/spike_timing_correction/baseline_v1/tests -v`: **3/3 통과**.

- `F(−1)≈+0.05`, `F(3)≈−0.05`; 동일한 pair window를 세 post-spike에 적용해 pair 3개를 각각 한 번 계산.
- 입력이 모두 0이면 pre-spike가 없어 post-spike만으로는 `ΔW=0`.
- W=0 학습 trial에서 target=1 출력은 실제 LIF teacher spike를 0 ms에 기록하며 target=0은 teacher spike가 없다.
- Teacher-free recall trial에는 teacher spike가 없고, W=0에서 natural spike도 없다.
- 보관된 final W는 두 조건 모두 음수 요소를 포함하고 크기 `(15,15)`; checkpoint 재로드 후 별도 초기 resting-state teacher-free recall은 각각 10/10.

## 전체 재현 검증

`python scripts/reproduce.py --dataset both --output /tmp/spike_timing_baseline_verify`로 보관본 소스만 사용하여 두 학습을 처음부터 수행했다.

| 항목 | Original | Modified |
|---|---:|---:|
| 보관 epoch exact/bit accuracy와 일치 | 1000/1000 | 1000/1000 |
| 보관된 전체 post-spike timestamp·count 배열과 일치 | 예 | 예 |
| 최종 W 최대 요소 차이 | 0.0 | 0.0 |
| 첫 10/10 | 113 | 21 |
| 마지막까지 연속 10/10 시작 | 226 | 37 |
| 최종 exact/bit | 10/10, 100% | 10/10, 100% |
| Teacher spike 관측/예상 | 100000/100000 | 100000/100000 |

보관본 전체를 `/tmp/spike_timing_archive_isolated`로 복사한 뒤에도 3/3 unit test와 재로드 checkpoint recall 10/10·10/10이 통과했다. 이는 wrapper와 검증 스크립트가 일반 작업 폴더의 임시 파일을 참조하지 않는다는 빠른 분리 검사다. 전체 1000-epoch 재현은 보관본이 원래 저장소 위치에 있을 때 수행했으며, 임의 위치 복사본에서는 다시 수행하지 않았다.

## 남은 범위

Modified 전체 membrane trace와 모든 epoch의 membrane trace는 대용량 중복 생성을 피하기 위해 보관하지 않았다. Spike event 전체는 Modified checkpoint에 보존했다. `numba`가 설치된 JIT 경로는 이번 환경에서 검증하지 않았다. 데이터 외부 공개 권한·라이선스와 실제 하드웨어의 timing/precision 적합성도 확인되지 않았다.
