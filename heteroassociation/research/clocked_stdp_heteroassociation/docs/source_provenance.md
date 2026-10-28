# 원본 코드와 검증 계보

최종 기준은 이름에 `final`이 포함된 파일이 아니라, 결정적 K=10 회귀 및 사용자 확정 K=20 데이터로 실행한 코드다. 원본 checkout의 검증 시점 Git HEAD는 `2802c60c95b25660417b6f31b468ad7f1876bcd2`였다. 다음 SHA-256은 원본 파일의 바이트 해시다.

| 역할 | 원본 경로 | SHA-256 |
|---|---|---|
| 실제 LIF trial 및 spike event 생성 | `my_research/experiments/diagnostics/binary_perceptron_lif_learning.py` | `7ab0872ff774ae224edb83d877e9c8912297174f6ba6fd5a91cd96b75589df9d` |
| K-dynamic 학습과 teacher-free recall | `my_research/experiments/diagnostics/association_capacity_experiment.py` | `c73e9463f2953b857ec32ef71d437b346ebd1815d2a79791aace096e184ccecd` |
| Centered window K10 검증 실행 | `results/inverted_mexican_hat_refined_explainer/20261027_122711/scripts/inverted_mexican_hat_refined_explainer.py` | `1f3888f556a091e8303769c9b831ee9bfc0ef1c1b30440f5b914440032550e3d` |
| 최종 Custom K20 실행 | `results/custom_K20_capacity_experiment/20261028_160342/scripts/custom_K20_capacity_experiment.py` | `26c608f17bfeb413809c9301071cf3c7e634b68738fefa942bcf6da2d3b3b420` |

아카이브 `src/model.py`는 첫 번째 파일의 Clocked LIF mode만 분리했다. `src/recall.py`, `src/training.py`는 두 번째 파일의 적분·batch recall·학습 순서를 K-dynamic 형태로 유지한다. 오래된 Natural LIF mode나 다른 STDP 창은 복사하지 않았다. `scripts/verify_baseline.py --full`로 K10/K15/K20의 501개 epoch 가중치 이력을 원본 checkpoint와 비교하여 실제 동작 보존 여부를 확인한다. 원본을 교체하거나 수정하지 않았다.
