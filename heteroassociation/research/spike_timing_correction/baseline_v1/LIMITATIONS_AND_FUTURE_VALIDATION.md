# 제한 및 후속 검증

이 버전은 Original·Modified K=10, 1000 epoch, 정해진 제시 순서, step current, 결정론적 0.1 ms 격자에서 검증됐다. 두 데이터셋의 10/10은 이 범위의 관측 사실이다.

- **전류 형태:** 현재 CS는 `[1.0,4.5)` ms 직사각형 step이다. 지수 감쇠 synaptic current에서 같은 timing correction이 되는지는 아직 검증되지 않았다.
- **노이즈:** 막전위 노이즈, 시간 jitter, 회로 변동에 대한 이번 공식 baseline 조건의 강건성은 별도 검증이 필요하다. 다른 과거 모델의 noise 결과를 이 버전의 결과로 이식하지 않는다.
- **용량:** K=10 두 집합의 성공은 K=10~100 용량을 보장하지 않는다. K 증가 시 rank, 선형 분리 가능성, 간섭, 실제 STDP 학습 실패를 구분해야 한다.
- **하드웨어:** signed weight, float64 수준의 spike-pair 합산, synchronized teacher pulse, 연속 LIF/STDP state, 50 ms spike history, association 종료 후 일괄 ΔW 적용을 실제 소자·플랫폼에서 어떻게 구현할지는 확인하지 않았다. Conductance mapping과 수치 정밀도 시험이 필요하다.
- **감독 신호:** Teacher current는 목표 bit에 따라 외부에서 주입한다. 완전 자율 또는 비지도 학습으로 소개하지 않는다.
- **공유:** 원본 데이터의 라이선스·공개 권한, 외부 패키지의 재배포 조건을 공개 전에 확인해야 한다. 이번 보관은 공개 Git 저장소 생성이나 라이선스 결정을 포함하지 않는다.

후속 실험은 이 디렉토리를 수정하지 않고 일반 작업 폴더에서 진행하며, 새 모델이 별도 검증을 통과할 때만 `baseline_v2` 같은 독립된 버전으로 보관한다.
