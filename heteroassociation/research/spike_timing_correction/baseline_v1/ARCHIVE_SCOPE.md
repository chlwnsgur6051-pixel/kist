# 보관 범위와 생략한 원본

이 디렉토리는 확정된 두 조건 **Original·Modified / CS end 4.5 ms / λ=0.5 / signed W**만 포함한다. 파일별 정확한 원본 상대 경로와 SHA-256은 `artifact_manifest.csv`에 있다. 원본 파일은 기존 위치에 그대로 남겨 두었다.

다음 대용량·중복 자료는 이 보관본에 복사하지 않았다.

| 생략 자료 | 원본 위치 | 이유 |
|---|---|---|
| 다른 CS 종료 시각과 λ 조합의 전체 sweep | `results/timing_correction_cs_duration_lr_sweep/` | 공식 baseline 조건 외의 파라미터 실험 |
| 추가 4.8·5.0 ms sweep | `results/timing_correction_cs_duration_lr_sweep/additional_cs_duration/` | 공식 baseline 조건 외의 후속 비교 |
| 모든 epoch의 full-resolution membrane/current trace | 원본 실험에도 전체 보존본 없음 | 전체 spike event·epoch 정확도·최종 W와 대표 trace를 우선 보존 |
| 발표용 그리드 전체 PDF/PNG | `results/timing_correction_final_visualization/` | 중복 그림; 대표 학습/recall 그림과 선택 trace는 포함 |
| Basic Correction·Nonnegative 결과 | `results/basic_correction_nonnegative_comparison/` 등 | 별도 모델이며 공식 baseline 성능과 혼동 방지 |

원본 ZIP에 기존 K=10 input/target 및 seed-0 제시 순서가 포함된다. 별도 `datasets/*.npz`와 CSV는 그 데이터를 읽기 쉽게 추출한 **동일 패턴**이다. 추가 random dataset, Dataset A, K>10 capacity, noise/jitter 실험은 이번 보관본에 포함하지 않았다.
