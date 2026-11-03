# 검증 범위와 한계

이번 동등성은 Original·Modified K=10, 1,000 epoch, seed 0 순서, dt 0.1 ms, 5 ms 무입력 간격, 지정 Mexican-hat 창 및 current cap에서 확인했다. 정확도 4,000개 epoch와 최종 가중치가 일치한다는 사실은 연속 상태 모델이 모든 입력·간격에서 기존 모델과 동일하다는 증명이 아니다.

원본 파일에는 현재 작업 환경에 없던 두 참조 ZIP이 하드코딩되어 있었고 Numba도 설치되어 있지 않았다. 패키지에는 기존 파일에서 **데이터·순서·정확도 수치를 변경하지 않고** 만든 참조 ZIP을 포함했다. 검증 실행은 원본 코드를 수정하지 않은 순수 Python `njit` fallback으로 수행했다. Numba로 JIT 컴파일한 경로의 수치 일치는 별도로 시험하지 않았다.

Teacher는 Target=1 출력에 외부 전류를 공급한다. Association 경계의 5 ms 대기와 제시 순서 역시 외부 프로토콜이다. Continuous-state recall은 실제 학습 뒤 10개 cue가 순차적으로 발생한 기록이 아니라 **학습 상태의 복사본을 이용한 counterfactual probe**다. 이 점 때문에 실제 하드웨어의 온라인 독립 평가까지 검증됐다고 볼 수 없다.

50 ms 이벤트 이력 절단은 이번 창에서 float64 underflow를 근거로 한 수치 구현이다. 다른 STDP 창이나 더 높은 정밀도에는 그대로 적용할 수 없다. Hardware의 이벤트 버퍼, 동시 이벤트 순서, delayed weight update 및 정밀도는 검증하지 않았다. Original의 W ≥ 0은 이번에도 10/10에 도달하지 못했으므로, 비음수 제약의 성공을 일반화할 수 없다.
