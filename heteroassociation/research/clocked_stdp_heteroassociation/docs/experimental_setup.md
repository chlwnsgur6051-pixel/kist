# 공통 실험 조건

Original과 Modified(B)의 원본 A–J target 10개를 유지한다. Custom K=15 v2는 사용자 지정 5개를 덧붙인 데이터이며, 최종 Custom K=20은 수정 확정된 P–T target을 이어 붙인다. Dataset A와 생성형 K=30 확장은 이 아카이브에 포함하지 않는다. 모든 K는 상위 데이터의 앞 K개를 사용한다.

기본 실험은 ascending association 순서, 500 epoch, 15×15 zero effective weight, clipping 없음, 외부 clock 4 ms, teacher 8 ms, teacher-free 평가다. 무작위 순서 실험의 seed/순열은 `results/csv/k15_capacity_accuracy_summary.csv` 및 `k20_presentation_order_results.csv`에 보존했다. Noise 실험은 clean K=10 가중치를 고정하고 recall 입력만 bit-flip했다. Jitter 실험은 clean 입력으로 학습하면서 spike timing을 바꾸고, 최종 jitter-free teacher-free recall로 평가했다. 두 강건성 실험은 기본 학습 코드의 파라미터를 변경한 기준 모델이 아니다.
