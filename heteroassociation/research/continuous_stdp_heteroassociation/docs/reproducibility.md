# 재현 방법과 파일 출처

검증 환경: Python 3.10.21, NumPy 2.2.6, Matplotlib 3.10.9. 원본은 `from numba import njit`를 사용하지만 이 환경에는 Numba가 없었다. `scripts/run_training.py`는 Numba가 없을 때 **동일 Python 함수 본문**을 실행한다. Numba가 설치된 환경에서는 원본의 JIT 경로를 사용하지만, 이번 결과는 fallback 경로를 검증한 것이다.

```bash
python research/continuous_stdp_heteroassociation/scripts/verify_results.py
python research/continuous_stdp_heteroassociation/scripts/run_training.py --output-dir /tmp/continuous_stdp_replay
MPLCONFIGDIR=/tmp/mpl_continuous python research/continuous_stdp_heteroassociation/scripts/plot_results.py
```

두 번째 명령은 출력 폴더가 비어 있을 때만 실행되고, 네 조건의 1000 epoch를 모두 계산한다. 실행 코드 `src/continuous_stdp_experiment.py`의 SHA-256은 `bd31e168fa600268aa5281b4e077c7e514771fe843083f49699b865b0e91b229`다. 기존 λ=0.1 파일은 `experiments/lambda_0p1/src/continuous_stdp_experiment.py`에 보존했으며, 실행 코드에서는 학습률 상수와 그 단위 검증값만 바꿨다. Numba fallback은 소스 코드를 바꾸지 않는다.

`src/reference_20261102_105031.zip`은 기존 `results/correction_learning_1ms4ms/20261102_105031/`의 Original·Modified K=10 Basic Correction NPZ와 seed 0 제시 순서 CSV를 원본 상대 경로대로 묶은 것이다. `src/reference_nonnegative_weight_results.zip`은 기존 `results/basic_correction_learning_rate_1/epoch_accuracy.csv`의 dataset·model·epoch·exact_count 열을 원본 코드가 기대하는 condition 표기와 ZIP 내부 경로로 **형식만 변환**했다. 결과 값은 변경하지 않았다. `datasets/`의 NPZ는 canonical Original/B 파일을 복사했으며, 참조 ZIP의 입력·target 배열과 elementwise 일치한다.

`verify_results.py`는 데이터 일치, 원본 소스 해시, 4,000개 epoch 정확도 비교, 최종 가중치 오차 `10⁻¹²` 미만 및 자연 출력의 명목 시각 이탈 0건을 검사한다. 결과 CSV의 sequential probe는 학습 상태 복사본, independent probe는 cue별 정지 상태다. 추가 seed, 데이터셋, 간격은 이번 검증에 포함하지 않았다.
