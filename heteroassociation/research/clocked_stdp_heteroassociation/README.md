# 검증된 Clocked LIF–STDP 이종연상 기억

이 폴더는 15×15 단일 feedforward 가중치 행렬에 이진 입력과 출력의 연상을 저장한 **검증 기준 버전**이다. 활성 입력 뉴런은 2 ms에 한 번 발화한다. 출력 LIF 막전위를 적분하고 4 ms clock에서 예측 spike를 판정한다. Target=1에만 8 ms teacher 전류를 주어 별도 post-spike를 유발한다. Teacher가 없는 Clocked LIF로 복원 정확도를 평가한다.

예측 spike의 STDP 기여는 LTD, teacher spike의 기여는 LTP다. 둘 다 발생하면 정해진 시각에서는 상쇄된다. 따라서 오류 출력 열에 연결된 활성 입력 synapse만 순변화한다. 코드의 실제 갱신은 `src/model.py`가 기록한 post-event와 `src/stdp.py`의 spike-pair 창으로 계산한다. 수학적 Perceptron 식을 학습 코드로 실행하지 않는다.

검증된 ascending 순서·500 epoch 결과: Original/Modified K=10은 각각 epoch 65/10에 처음 100%, K=15 v2는 117/69에 처음 100%다. 최종 K=20은 각각 55%/70%였으며, 동일 readout의 수학적 최적값 90%/95%와 다르다. 전체 수치와 조건은 [실험 결과](docs/experimental_results.md) 및 [용량 분석](docs/capacity_analysis.md)에 있다.

## 독립 실행

저장소 루트에서 Python 3.10 및 NumPy 2.2.6을 준비한 뒤 실행한다. 이 폴더 안의 설정, 데이터 및 checkpoint만으로 동작한다.

```bash
python3 research/clocked_stdp_heteroassociation/scripts/verify_baseline.py --full
python3 research/clocked_stdp_heteroassociation/scripts/run_recall.py original --k 10
python3 research/clocked_stdp_heteroassociation/scripts/run_training.py modified --k 15 --output /tmp/modified_K15_replay.npz
python3 -m unittest discover -s research/clocked_stdp_heteroassociation/tests -v
```

`run_training.py`는 K=1–20 prefix를 허용한다. K=15 v2와 최종 K=20의 앞 10/15개는 각각 이전 데이터와 일치한다. `--shuffle-seed`가 없으면 ascending 고정 순서다. 실행 출력은 명시한 경로에만 쓴다.

데이터·설정·원본 결과 경로와 SHA-256은 [artifact_manifest.csv](artifact_manifest.csv), 실험 간 연결은 [experiment_manifest.csv](experiment_manifest.csv)에 있다. [재현 방법](docs/reproducibility.md), [모델 구조](docs/model_architecture.md), [STDP 원리](docs/stdp_learning_rule.md), [한계](docs/limitations_and_future_work.md)를 함께 참조한다. 이전 실험 폴더는 변경하지 않았다.
