# 재현 절차와 원본 계보

기준 실행은 Python 3.10.21, NumPy 2.2.6, `dt=0.1 ms`, `epochs=500`, ascending 순서, seed 없음(결정적), no clipping이다. 설정은 `configs/validated_model.json`이다. K10 회귀 기준은 `results/inverted_mexican_hat_refined_explainer/20261027_122711`이며, K15 v2는 `results/association_capacity_experiment/20261028_141753`, K20 최종은 `results/custom_K20_capacity_experiment/20261028_160342`다. K10 canonical 데이터 SHA-256과 모든 아카이브 파일·원본 경로·해시는 `artifact_manifest.csv`에 있다. `experiment_manifest.csv`는 데이터→설정→코드→결과→그림을 연결한다.

```bash
python3 -m unittest discover -s research/clocked_stdp_heteroassociation/tests -v
python3 research/clocked_stdp_heteroassociation/scripts/verify_baseline.py --full
python3 research/clocked_stdp_heteroassociation/scripts/run_training.py original --k 20 --epochs 500 --output /tmp/original_K20_replay.npz
python3 research/clocked_stdp_heteroassociation/scripts/run_recall.py modified --k 10
```

회귀 스크립트는 여섯 결정적 실행의 **모든 501개 epoch 가중치**를 보관 checkpoint와 elementwise 비교하며 최대 오차가 1e-12 미만이어야 통과한다. 최종 예측, four-case event, inactive input 불변성, target=0 teacher 부재도 검사한다. Shuffle 재현은 `--shuffle-seed SEED`를 사용한다. 실험별 실제 순서는 CSV에 있다.

이 폴더는 외부 원본 결과 디렉터리 없이 K≤20의 학습 및 teacher-free recall이 가능하다. 전체 noise/jitter 시뮬레이션의 대용량 epoch별 원시 로그는 중복 복사하지 않았다. 해당 상세 결과가 필요하면 원본 경로·SHA-256을 manifest로 확인하여 별도로 준비한다. `scripts/archive_sources.py`는 원본 checkout에 접근할 때만 선택된 파일을 다시 복사하는 보조 도구다.
