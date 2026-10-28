# Spike timing 기반 출력별 오류 수정

창은 `F(Δt)=(A/15)(u²−1)exp(−u²/2)`, `u=Δt/s`이며 `A=3.4138033513868233`, `s=2.5737871845939315 ms`, 중심 0이다. `Δt=t_post−t_pre`다. 활성 입력은 2 ms에 발화하고 예측 post-event는 4 ms, teacher post-event는 8 ms다. 이 설정에서 `F(2 ms)=−1/15`, `F(6 ms)=+1/15`(부동소수점 오차 범위)이다.

| Target | Prediction | 실제 post-event | 활성 입력 synapse의 순변화 |
|---:|---:|---|---:|
| 0 | 0 | 없음 | 0 |
| 0 | 1 | prediction | −1/15 (LTD) |
| 1 | 0 | teacher | +1/15 (LTP) |
| 1 | 1 | prediction, teacher | 약 0 (LTD+LTP) |

실행 코드는 먼저 LIF가 생성한 event 목록을 만들고, 각 활성 pre-spike와 실제 post-spike의 모든 pair에 `F`를 합산한다. 입력 0인 행은 pair가 없다. 이 고정 시각·단일 pre-event·teacher spike 신뢰성 조건에서만 순변화가 `(1/15)x_i(y_j−ŷ_j)`와 등가다. 이 식은 구현을 대체하지 않는다. 시각 오차, teacher 발화 실패, extra spike, clipping 또는 다른 pairing 규칙에서는 등가성이 깨질 수 있다. 11 case도 gross LTP/LTD가 각각 발생하며 순변화만 0이다.

`tests/test_mechanism.py`와 `scripts/verify_baseline.py`는 실제 event로 네 경우를 검사한다. 이 구조는 출력별 교정이지만 target stimulation과 clock이 필요하므로 일반적인 무감독 STDP로 해석해서는 안 된다.
