"""STDP_CURVE_TYPES 6종 곡선을 시각화.

실제 model.STDPRule.stdp_factor(dt, weight) 를 그대로 호출해 그리므로
코드와 동일한 곡선이 나온다. 가로축은 Δt = t_post - t_pre (ms),
세로축은 곱셈 배수 factor (>1 강화, <1 약화, =1 변화없음).

시각화가 잘 보이도록 파라미터를 정규화한다:
  wmin=0, wmax=1, weight=0.5, p=1, d=1  → factor 가 대략 [0.5, 1.5] 범위.
(실제 학습에서 쓰는 p=0.001, d=0.01 은 변화가 미세해 모양이 안 보인다.)

이 스크립트는 visualization/ 안에 있으므로 프로젝트 루트를 import 경로에 추가한다.
결과는 visualization/results/stdp_curves.png 에 저장된다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt  # 백엔드 자동 선택: GUI 있으면 창 표시, 없으면 저장만

plt.rcParams["axes.unicode_minus"] = False   # keep ASCII minus sign

from model import STDPRule, STDP_CURVE_TYPES

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# 공통 시각화 파라미터 (모양이 잘 보이도록 정규화)
COMMON = dict(p=1.0, d=1.0, ptau=1e-3, dtau=5e-4, wmin=0.0, wmax=1.0)
WEIGHT = 0.5                      # 중간 가중치에서 factor 평가
DT = np.linspace(-5e-3, 5e-3, 601)  # Δt: -5ms ~ +5ms

# curve 이름 -> 추가 파라미터 (shifted 는 sh 로 실제 이동을 보여줌)
EXTRA = {"shifted_mexican_hat": dict(sh=1e-3)}

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.ravel()

for ax, (name, ctype) in zip(axes, STDP_CURVE_TYPES.items()):
    rule = STDPRule.from_curve_name(name, **COMMON, **EXTRA.get(name, {}))
    factor = np.array([rule.stdp_factor(float(dt), WEIGHT) for dt in DT])

    ax.axhline(1.0, color="gray", lw=0.8, ls="--")   # 변화 없음 기준선
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")     # Δt=0 (동시 발화)
    ax.plot(DT * 1e3, factor, color="tab:blue", lw=2)
    ax.fill_between(DT * 1e3, 1.0, factor, where=factor >= 1.0,
                    color="tab:red", alpha=0.15)       # 강화 영역
    ax.fill_between(DT * 1e3, 1.0, factor, where=factor < 1.0,
                    color="tab:blue", alpha=0.15)      # 약화 영역
    subtitle = name + (f"  (sh={EXTRA[name]['sh']*1e3:.0f}ms)" if name in EXTRA else "")
    ax.set_title(f"[{ctype}] {subtitle}", fontsize=11)
    ax.set_xlabel("dt = t_post - t_pre (ms)")
    ax.set_ylabel("factor (multiplier)")

fig.suptitle("STDP curves - stdp_factor(dt) at weight=0.5, p=d=1, wmax=1\n"
             "red = potentiation (>1), blue = depression (<1)", fontsize=13)
fig.tight_layout()
out = RESULTS / "stdp_curves.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
plt.show()  # GUI 환경이면 창 표시 (닫을 때까지 대기)
