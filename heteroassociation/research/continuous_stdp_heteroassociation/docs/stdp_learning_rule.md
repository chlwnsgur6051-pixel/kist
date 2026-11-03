# Continuous All-to-All Spike-Pair STDP

원본은 모든 실제 pre/post 이벤트에 동일한 창을 사용한다.

\[
F(\Delta t)=C\left(1-(\Delta t/s)^2\right)e^{-\Delta t^2/(2s^2)},\quad
s=1.286893592\;\mathrm{ms},\quad C=0.170690168,
\]

\[
\Delta w_{ij}=\lambda\sum_{t_{\mathrm{pre},i}}\sum_{t_{\mathrm{post},j}}
F(t_{\mathrm{post},j}-t_{\mathrm{pre},i}),\qquad \lambda=1.
\]

0 ms teacher-driven spike와 1 ms pre-spike의 간격은 −1 ms이며 LTP가 약 +0.05다. 1 ms pre-spike와 4 ms 자연 출력의 간격은 +3 ms이며 LTD가 약 −0.05다. 둘 다 발생하면 명목상 서로 상쇄한다. Teacher와 자연 출력을 STDP 함수 내부에서 다른 규칙으로 처리하지 않는다. 단, target-dependent teacher current는 감독 신호다.

`ContinuousAllPairs`는 뉴런별 최근 이벤트 시각을 유지하고 새 이벤트가 들어올 때 앞선 반대편 이벤트와 한 번씩 짝짓는다. 같은 presentation 안의 모든 쌍뿐 아니라 association 경계에 걸친 쌍도 처리한다. 50 ms보다 오래된 이벤트는 버린다. 이 창의 Gaussian 항은 해당 시각에서 float64로 0에 언더플로하므로, **이 수치 구현에서는** 무시한 쌍의 기여가 0이다. 가중치는 presentation 종료 시 한 번 갱신되어 같은 presentation의 teacher LTP가 자연 prediction에 즉시 영향을 주지 않는다.

측정된 경계 간 쌍의 최대 단일 기여는 `9.47×10⁻¹⁹`, 전체 절댓값 합은 조건별 `6.62–9.86×10⁻¹³`였다. 이 작은 값과 출력 spike 시각의 일치 때문에 이번 조건의 최종 가중치가 이전 trial-local STDP와 부동소수점 오차 수준에서 같았다. 이는 다른 trial 간격·STDP 창에도 성립한다는 주장이 아니다.
