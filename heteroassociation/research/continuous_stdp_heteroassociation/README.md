# 검증된 Continuous-State Natural LIF + All-to-All STDP 연상기억

이 폴더는 사용자가 제공한 `continuous_stdp_experiment.py`를 **원본 그대로 보존**하고, Original·Modified K=10에서 기존 Basic Correction과 비교 검증한 기준 패키지다. 출력 뉴런 15개와 이진 입력 뉴런 15개를 15×15 단일 feedforward 가중치 행렬로 연결한다. 학습에서는 Target=1 출력에 teacher 전류를 주어 실제 0 ms 발화를 만들고, 1 ms 입력 spike 이후 자연 출력 발화를 관측한다. Recall에는 teacher가 없다.

새 코드의 차이는 association 사이에 LIF 막전위·refractory 상태와 STDP 이벤트 이력을 유지한다는 점이다. 10.1 ms trial 뒤 5 ms 무입력 간격을 둔다. 활성 pre와 실제 post의 모든 spike 쌍을 동일한 Mexican-hat 창으로 누산하고 presentation 종료 후 가중치를 갱신한다. 이 구조를 완전히 자율적인 학습이라고 설명하지 않는다. Teacher와 association 제시·간격은 외부에서 제어한다.

## 검증 결과

기존 모델과 같은 seed 0 제시 순서, 학습률 0.1, 1,000 epoch, current cap `1.282590743781507`로 네 조건을 실행했다. 두 recall 방식(상태를 이어가는 sequential probe와 각 cue를 초기 상태에서 읽는 independent probe) 모두 기존 모델의 **4,000개 epoch 정확도와 전부 일치**했다. 최종 가중치의 최대 절댓값 차이는 `1.23×10⁻¹⁶`이다.

| Dataset | 가중치 조건 | 최초 10/10 | 최종 Exact Recall |
|---|---|---:|---:|
| Original | 부호 허용 Baseline | 295 | 10/10 |
| Original | W ≥ 0 | 도달 못함 | 5/10 |
| Modified | 부호 허용 Baseline | 48 | 10/10 |
| Modified | W ≥ 0 | 52 | 10/10 |

따라서 **검증한 데이터·시각·5 ms 간격에 한해** 이 코드를 후속 Natural LIF Basic Correction 연구의 기준 버전으로 채택한다. 5 ms 이외의 간격이나 다른 데이터에 대한 동등성은 검증하지 않았다. [실험 결과](docs/experimental_results.md)와 [한계](docs/limitations.md)를 함께 읽어야 한다.

## 재현

원본 파일이 기대하는 두 참조 ZIP을 `src/`에 보관했다. 현재 환경에는 Numba가 없으므로 검증 실행은 `scripts/run_training.py`의 순수 Python fallback으로 수행했다. 이 fallback은 JIT 가속만 생략한다. 원본 학습 코드는 변경하지 않는다.

```bash
python research/continuous_stdp_heteroassociation/scripts/verify_results.py
python research/continuous_stdp_heteroassociation/scripts/run_training.py --output-dir /tmp/continuous_stdp_replay
MPLCONFIGDIR=/tmp/mpl_continuous python research/continuous_stdp_heteroassociation/scripts/plot_results.py
```

두 번째 명령은 네 조건의 1,000 epoch를 다시 실행한다. 저장된 결과는 `results/csv/`, 최종 행렬은 `results/checkpoints/`, 대표 그림은 `results/figures/`에 있다. 데이터·코드·참조의 출처와 파일 해시는 [artifact_manifest.csv](artifact_manifest.csv), 재현 절차는 [reproducibility.md](docs/reproducibility.md)에 정리했다.
