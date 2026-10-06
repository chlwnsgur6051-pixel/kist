# 템퍼럴_stdp — 단층 + temporal(latency) 연관기억 실험

우리 숫자 15bit 입력을 **단층(은닉층 없음)** 으로 받아, 출력을 **스파이크 시각(latency)**
으로 코딩하는 실험 환경. 가중치 학습 방식이 두 가지:

- **temporal_hebbian.py** — Hebbian 외적 한 방 저장 (연관기억, 우리 프로젝트 원래 프레임).
  `w = (1/K) Σ_k t_bipolar_k · x_bipolar_kᵀ`.  crosstalk 로 XNOR 픽셀(7) 천장 재현.
- **temporal_stdp.py** — supervised 오차보정 (chronotron delta 규칙, 대조군).
  `dwᵢ = lr·(latency − target_lat)·xᵢ` 반복.

## 구조 요약

- **입력**: 숫자 비트맵 15bit (`binary_character.make_digit_dataset`)
- **모델**: 단층 가중치(15-벡터) → 출력 뉴런 1개
- **출력 코딩(latency)**:
  - tonic ramp 로 중립(드라이브 0)이면 창 중앙 `T_CENTER=5` 에 발화
  - 드라이브 `D = Σ wᵢxᵢ` 가 발화 시각을 평행이동: D 크면 이른 발화(→"1"), 작으면 늦은 발화(→"0")
  - 판정: `latency < T_CENTER → 1`
- **학습(temporal STDP / chronotron 식)**:
  - 목표 스파이크 시각(1→이른 `T_TARGET_ONE`, 0→늦은 `T_TARGET_ZERO`)
  - 규칙: `dwᵢ = lr·(latency − target_lat)·xᵢ`
    (너무 늦게 발화 → 활성 입력 강화 → 다음엔 빨리 / 너무 이르면 약화)

## 실행

```bash
python "템퍼럴_stdp/temporal_stdp.py"
```

## 설정 (파일 상단)

- `N_PAIRS`    : 사용할 (숫자→글자) 페어 수, 앞에서부터 N개 (기본 10, 최대 10).
                 그림 그리드도 페어 수에 맞춰 자동으로 접힘.
- `OUT_NEURON` : 볼 출력(letter) 픽셀 (기본 0). 어려운 XNOR 픽셀은 7.
- `SWEEP_ALL`  : `True` 면 15개 출력 뉴런 전부 학습 → 픽셀별 정확도(5×3) 표시
- `T_CENTER / T_TARGET_ONE / T_TARGET_ZERO` : latency 코딩/판정 시각
- `LR / EPOCHS` : 학습률/에폭

## 출력 (results/)

단일 뉴런(`OUT_NEURON`):
- `n{J}_1_weights.png`   학습된 15 가중치 (5×3, 빨강=+, 파랑=−)
- `n{J}_2_membrane.png`  숫자별 막전위 램프 궤적 (이른 교차=1)
- `n{J}_3_latency.png`   숫자별 출력 latency + 판정
- `n{J}_4_learning.png`  학습 곡선 (정확도 / latency 오차)
- `n{J}_5_spikes.png`    페어별 스파이크 래스터 (활성 입력 @ t=0 + 출력 latency).
                         꺼진 픽셀은 모델이 안 쓰므로 그리지 않음.
- `n{J}_6_out_membrane.png`  숫자별 출력 멤브레인 v(t) 패널 (임계 교차=발화 시각)

`SWEEP_ALL=True`:
- `sweep_per_pixel_acc.png`  15픽셀 각각의 정확도 + 전체 letter/bit 정확도
`recall_grid.png` (input(숫자) → target(글자) → pred(글자) 비트맵 그리드)는
`SWEEP_ALL` 값과 무관하게 `RECALL_GRID=True` 면 항상 저장된다.

## 메모 (한계)

- 이 latency 출력은 드라이브의 **단조 함수** → 여전히 **선형 판정**. 선형분리 가능한
  픽셀은 잘 맞지만, **XNOR 픽셀(예: 7)** 은 단층으로는 한계(≤9/10 계열).
- 비선형(XNOR)까지 가려면 입력 쪽 **2-phase + coincidence** 를 추가해야 함(별도 확장).
- 여기 STDP 규칙은 spike-time-error 기반 로컬 delta 규칙(chronotron 식)으로,
  "temporal 교육 신호로 단층 가중치를 학습"하는 부분을 검증하는 용도.
