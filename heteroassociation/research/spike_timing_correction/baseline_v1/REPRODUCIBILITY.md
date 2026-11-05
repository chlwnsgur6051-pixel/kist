# 재현 방법

## 검증 환경과 설치

2026-11-05 검증 환경은 Python **3.10.21**, NumPy **2.2.6**, Matplotlib **3.10.9**였다. `numba`는 설치되지 않았고, [재현 wrapper](scripts/reproduce.py)는 원본 `@njit`을 같은 함수를 반환하는 shim으로 대체해 순수 Python으로 실행했다. 결과의 원본 코드/조건과 학습 이력/최종 W/spike event가 모두 일치했다. 다른 패키지 버전이나 numba 활성 환경에서의 동일성은 별도로 검증하지 않았다.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r research/spike_timing_correction/baseline_v1/requirements.txt
```

가상환경 설치는 사용자가 원하는 환경에서 수행한다. 이 보관본 제작 시 새 패키지를 설치하지 않았다.

## 한 명령으로 두 데이터셋 학습 재현

저장소 루트에서 다음을 실행한다. Wrapper와 그 내부 경로는 `Path(__file__)`를 기준으로 하므로 현재 작업 디렉토리나 개인 홈 경로에 의존하지 않는다.

```bash
python research/spike_timing_correction/baseline_v1/scripts/reproduce.py --dataset both
```

`--dataset Original` 또는 `Modified`로 한 조건만 실행할 수 있다. `--output /원하는/경로`를 주면 쓰기 위치를 바꿀 수 있으며 기본은 `baseline_v1/runs/`다. 실행은 두 데이터셋 각각 초기 W=0부터 1000 epoch를 진행하고, 보관된 1000개 epoch의 exact·bit accuracy, 각 출력 post-spike timestamp·count, 최종 W의 요소별 값을 비교한다. 불일치하면 assert 오류로 중단한다. 성공 시 `runs/original_final_weights.npz`, `runs/modified_final_weights.npz`, 두 epoch-history CSV, `runs/verification.csv`를 만든다. 기본 `runs/`는 Git에서 제외했다.

학습 제시 순서는 `datasets/*_K10.npz`의 `(1000,10)` `presentation_orders`다. 원본 `src/reference_20261102_105031.zip`에서 모델의 기존 `load_data()`가 같은 배열을 읽고, wrapper가 추출한 NPZ와 일치하는지 검증한다. Original·Modified input/target은 각각 `(10,15)`, 가중치는 `(15,15)`다. 새 seed를 만들지 않는다. Exact recall은 epoch 뒤 teacher-free **연속 순차 probe**에서 15개 bit 모두가 target과 같은 association의 비율이다.

## 빠른 테스트와 그림 재생성

```bash
python -m unittest discover -s research/spike_timing_correction/baseline_v1/tests -v
python research/spike_timing_correction/baseline_v1/scripts/recall_checkpoint.py
MPLCONFIGDIR=/tmp/mpl_baseline_v1 python research/spike_timing_correction/baseline_v1/scripts/plot_results.py
```

`recall_checkpoint.py`는 저장된 W를 다시 읽어 **새 resting state**에서 teacher-free recall하는 별도 진단이며, 공식 epoch별 연속 copy-state 지표와 구분한다. 그림 스크립트는 학습을 하지 않고 보관된 CSV·NPZ를 읽어 `figures/accuracy_by_epoch.*`, `figures/final_weights.*`, `figures/original_epoch6_spike_membrane.*`를 만든다. `figures/original_learning.png`, `figures/modified_learning.png`, `figures/stdp_early_late.png`, `figures/original_training_vs_recall.png`는 검증된 앞선 실험에서 복사한 대표 그림이다. 그 수치 원자료는 `results/csv/`와 `results/checkpoints/`에 보존했다. Original 선택 trace는 `selected_traces.npz`에 있다. Modified의 전체 막전위 trace는 이 baseline 보관본에 포함하지 않았다. Spike event 전체는 Modified checkpoint에 있다.

## 보관 원본과 검증 기준

- 정확한 원본 Python 스냅샷: `src/sweep_model.py`, `src/run_sweep.py`. 원본 `sweep_model.py`는 기본값으로 이전 조건을 두며 `__main__`에서 직접 실행을 금지한다. Wrapper가 검증된 `CS_END_TICK=45`, `LAMBDA=0.5`, `CAP=None`을 설정한다. 함수 본문은 바꾸지 않았다.
- 지정 조건: `configs/baseline_v1.json`; 당시 실험 출력 JSON은 `configs/*_archived_condition.json`.
- 원본 결과: `results/csv/*_epoch_history.csv`, `results/checkpoints/*_training.npz`, `results/checkpoints/final_weights.npz`.
- SHA-256 및 원본 상대 경로는 `artifact_manifest.csv`에 있다. 생성 명령은 `python scripts/build_manifest.py`다.
- 재현 결과: Original 첫 10/10=113, 지속 시작=226; Modified 첫=21, 지속 시작=37; 두 데이터셋 모두 최종 10/10, bit 100%, archived final W 차이=0, post-spike event 배열 일치.

보관된 checkpoint는 full-precision NumPy 배열이다. 그림의 색상·반올림 숫자를 checkpoint 대신 쓰지 않는다. 데이터셋 원자료의 외부 공유 권한과 공개 라이선스는 확인되지 않았으므로 이 보관본이 자동으로 공개 배포 허가를 의미하지 않는다.
