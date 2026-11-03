# 검증 모델 안내

이 폴더는 재현 검증을 마친 두 모델을 분리해 보관한다. 새 실험의 출발점을 정할 때 모델의 뉴런 동작과 감독 신호가 서로 다르다는 점을 먼저 확인한다.

| 모델 | 주요 방식 | Original K=10 | Modified K=10 | 진입 문서 |
|---|---|---:|---:|---|
| Clocked LIF + STDP | 지정 시각의 막전위 판정과 teacher stimulation | 10/10 | 10/10 | [clocked_stdp_heteroassociation](clocked_stdp_heteroassociation/README.md) |
| Continuous-state Natural LIF + All-to-All STDP | 실제 역치 통과 spike, λ=1, 부호 허용 가중치 | 10/10 | 10/10 | [continuous_stdp_heteroassociation](continuous_stdp_heteroassociation/README.md) |

Continuous-state 모델의 이전 λ=0.1 코드와 결과는 [별도 보관](continuous_stdp_heteroassociation/experiments/lambda_0p1/README.md)했다. λ=1에서 기존 Basic Correction과 4,000개 epoch 정확도를 대조한 기록은 [검증 문서](continuous_stdp_heteroassociation/docs/experimental_results.md)에 있다. 비음수 가중치의 Original 결과는 4/10이므로 전체 모델 조건의 성공으로 해석하지 않는다.

각 모델의 실행 명령과 결과 출처는 해당 폴더의 `README.md`, `docs/reproducibility.md`, `artifact_manifest.csv`를 참조한다. 두 모델 모두 학습에 감독 신호를 사용하며, 소프트웨어 결과가 하드웨어 구현 가능성을 입증하지는 않는다.
