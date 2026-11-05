# Spike-Timing Correction — 공식 Baseline v1

2026-11-05 지도교수 면담 후 채택한 **Natural LIF + 실제 teacher spike + 실제 natural output spike + all-to-all pair-based STDP** 기준 모델이다. 학습 중 출력 발화의 실제 시각이 4 ms보다 빠르거나 늦으면 STDP 기여가 달라져 timing correction이 생긴다. 이 폴더는 검증된 구현의 스냅샷과 결과를 보존한다. 기존 Basic Correction이나 Clocked LIF 모델은 이 공식 기준 모델과 다르다.

## 모델과 데이터

- 15 input × 15 output 단일 feedforward 가중치 행렬. 입력과 목표는 0/1이며 `1`인 입력만 1.0 ms에 발화한다.
- 출력은 Natural LIF의 실제 threshold crossing으로만 발화한다. Target=1 출력에는 학습 때만 `[-0.1, 0)` ms 동안 teacher current 30을 주입해 실제 0 ms spike를 만든다.
- CS 전류는 `Σ_i x_i W_ij`이며 gain 1, cap 없음, `[1.0, 4.5)` ms 동안 공급된다. 음수 가중치를 허용하고 clipping은 없다.
- Mexican-hat `F(Δt)=C[1−(Δt/s)^2]exp[−Δt²/(2s²)]`, `s=1.286893592 ms`, `C=0.170690168`, `λ=0.5`를 모든 실제 pre/post spike pair에 동일하게 적용한다.
- LIF 막전위·불응기와 STDP spike history를 association 사이에 유지한다. Association 종료 뒤 누적 `ΔW`를 한 번 적용한다. 각 association 뒤에는 전류 없는 5 ms 추가 휴지 시간을 둔다.
- 학습은 Original·Modified 각각 고정된 seed-0 제시 순서로 1000 epoch이다. Epoch 후 recall은 학습 상태의 **복사본**에서 teacher 없이 연속적으로 10개 cue를 검사한다.

자세한 수학과 시간 규약은 [MODEL_SPEC.md](MODEL_SPEC.md), 재현 명령은 [REPRODUCIBILITY.md](REPRODUCIBILITY.md), 결과는 [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md)에 있다.

## 핵심 관측 결과

| Dataset | 첫 10/10 | 최종 epoch까지 연속 10/10 시작 | Epoch 1000 |
|---|---:|---:|---:|
| Original | 113 | 226 | 10/10, bit 100% |
| Modified | 21 | 37 | 10/10, bit 100% |

이는 관측한 1000 epoch 안의 결과이며 무한한 학습 안정성을 뜻하지 않는다. 두 데이터셋의 원본과 체크포인트, 1000개 epoch 기록은 `datasets/`와 `results/`에 포함했다.

## 폴더

- `src/`: 검증된 sweep의 **바이트 동일 코드 스냅샷** `sweep_model.py`, `run_sweep.py`와 원본 입력·순서 ZIP. 두 Python 파일은 실험 당시 소스 그대로다. `sweep_model.py`의 `__main__`은 직접 실행을 막으므로 아래 wrapper를 사용한다.
- `configs/`: 확정 조건 JSON과 당시 조건별 출력 JSON.
- `datasets/`: ZIP/체크포인트에서 추출한 Original·Modified K=10의 NPZ·사람이 읽을 수 있는 CSV. 1000 epoch 제시 순서는 NPZ에 있다.
- `scripts/`: 단일 조건 재현, 그림 생성, 무결성 manifest 작성.
- `tests/`: 짧은 기전·데이터 무결성 테스트. 전체 학습 parity 검증은 재현 명령이 담당한다.
- `results/`: 원본 epoch 기록, 실제 spike event를 포함한 체크포인트, 재현 요약.
- `figures/`: 발표용 대표 그림. 생성·원본 관계는 manifest에 기록한다.

## 실행

저장소 최상위에서:

```bash
python -m unittest discover -s research/spike_timing_correction/baseline_v1/tests -v
python research/spike_timing_correction/baseline_v1/scripts/reproduce.py --dataset both
MPLCONFIGDIR=/tmp/mpl_baseline_v1 python research/spike_timing_correction/baseline_v1/scripts/plot_results.py
```

재현 결과는 기본적으로 이 폴더의 `runs/`에 저장되며 Git에서 제외한다. `--output`으로 다른 위치를 지정할 수 있다. 학습이 필요 없으면 보관된 `results/checkpoints/final_weights.npz`를 바로 사용할 수 있다. 숫자를 반올림한 그림은 checkpoint가 아니다.

## 범위와 한계

Teacher는 감독 신호이며 모델은 완전 자율 학습이 아니다. Mexican-hat 커널은 시간차에 대해 짝함수로, 일반적인 pre-before-post LTP / post-before-pre LTD 비대칭 STDP와 동일하지 않다. 현재 검증은 Original·Modified K=10의 이 결정론적 조건과 소프트웨어 시뮬레이션에 한정된다. 하드웨어 적합성, 지수형 전류, 막전위 노이즈, 대규모 capacity는 검증하지 않았다. 데이터 출처와 공유 권한은 원본 연구 기록 기준이며, **공개 라이선스나 재배포 권한은 확인되지 않았다.**

기존 보관 폴더는 변경하지 않았다. 이 `baseline_v1`은 고정 기준으로 보존하고 후속 연구는 별도 작업 폴더에서 수행한다.
