# 수학적 용량과 실제 STDP 학습

K가 증가해도 입력 차원·출력 차원·가중치 수는 15·15·225로 고정된다. 최종 Custom K=20의 input rank는 Original 14, Modified 15다. 그러나 rank만으로 exact recall을 결정할 수는 없다. 입력 간 중첩, target 관계, 각 출력 열의 threshold 선형분리 가능성, 순차 update에 따른 간섭도 영향을 준다.

같은 고정 Clocked LIF threshold를 반영한 subset LP 최적 readout은 K=15에서 두 데이터 모두 15/15를 인증했다. K=20은 Original 최대 18/20(90%), Modified 최대 19/20(95%)로 원본 결과 CSV에 `certified_subset_LP`라고 기록되었다. 이는 가중치의 **존재 가능성**을 말하며 STDP가 그 해를 학습했다는 뜻이 아니다. 실제 ascending STDP는 K=20 최종 11/20(55%), 14/20(70%)였다. 각각 수학적 한계와의 차이 35/25 percentage points에는 학습 과정의 제한이 포함된다. 반대로 LP의 100% 불가능성은 해당 고정 threshold·입력·target 조합에 관한 구조적 제약이다.

Modified의 ascending K=16–19는 모두 최종 100%; Original은 K=16부터 이미 100%에 미달했다. 그러므로 이 결과에서 '최고 100% K'는 Original 15, Modified 19이다. 이는 지정된 **prefix·순서·500 epoch** 조건의 관찰값이다. K20 다른 고정 shuffle 순서에서는 ascending과 다른 결과가 기록되어 있다. Order를 평균하지 않고 `results/csv/k20_presentation_order_results.csv`에 별도로 보존했다. Rank와 용량의 인과관계 또는 모든 순서에 대한 최대 저장량을 주장할 근거는 아니다.
