"""STDP_CURVE_TYPES 6종 곡선을 시각화 (실제 학습 파라미터 사용).

실제 model.STDPRule.stdp_factor(dt, weight) 를 그대로 호출해 그린다.
파라미터는 stdp_t2_inhibition.PARAMETER_SET 의 실제 값을 그대로 쓴다
(정규화하지 않음). 평가 가중치는 initial_weight.

가로축 Δt = t_post - t_pre (ms), 세로축 factor (>1 강화, <1 약화, =1 변화없음).

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
import stdp_t2_inhibition as base

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# 실제 학습 파라미터 그대로
ps = base.PARAMETER_SET
COMMON = dict(p=ps["stdp_p"], d=ps["stdp_d"], ptau=ps["stdp_ptau"],
              dtau=ps["stdp_dtau"], wmin=ps["wmin"], wmax=ps["wmax"], sh=ps["stdp_sh"])
WEIGHT = float(ps["initial_weight"])    # 실제 초기 가중치에서 factor 평가
DT = np.linspace(-5e-3, 5e-3, 601)      # Δt: -5ms ~ +5ms

# shifted 만 실제 이동을 보여주려 sh 를 dtau 정도로 (다른 곡선은 실제 sh 사용)
EXTRA = {"shifted_mexican_hat": dict(sh=ps["stdp_dtau"])}

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.ravel()

for ax, (name, ctype) in zip(axes, STDP_CURVE_TYPES.items()):
    params = {**COMMON, **EXTRA.get(name, {})}   # EXTRA 가 COMMON 을 덮어씀 (sh 중복 방지)
    rule = STDPRule.from_curve_name(name, **params)
    factor = np.array([rule.stdp_factor(float(dt), WEIGHT) for dt in DT])

    ax.axhline(1.0, color="gray", lw=0.8, ls="--")   # 변화 없음 기준선
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")     # Δt=0 (동시 발화)
    ax.plot(DT * 1e3, factor, color="tab:blue", lw=2)
    ax.fill_between(DT * 1e3, 1.0, factor, where=factor >= 1.0,
                    color="tab:red", alpha=0.15)       # 강화 영역
    ax.fill_between(DT * 1e3, 1.0, factor, where=factor < 1.0,
                    color="tab:blue", alpha=0.15)      # 약화 영역
    subtitle = name + (f"  (sh={EXTRA[name]['sh']*1e3:.2f}ms)" if name in EXTRA else "")
    ax.set_title(f"[{ctype}] {subtitle}", fontsize=11)
    ax.set_xlabel("dt = t_post - t_pre (ms)")
    ax.set_ylabel("factor (multiplier)")

fig.suptitle(f"STDP curves (real params) - stdp_factor(dt) at weight={WEIGHT:.0f}\n"
             f"p={COMMON['p']}, d={COMMON['d']}, ptau={COMMON['ptau']*1e3:.1f}ms, "
             f"dtau={COMMON['dtau']*1e3:.1f}ms, wmin={COMMON['wmin']:.0f}, wmax={COMMON['wmax']:.0f}"
             "   |   red=potentiation, blue=depression", fontsize=12)
fig.tight_layout()
out = RESULTS / "stdp_curves.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
plt.show()  # GUI 환경이면 창 표시 (닫을 때까지 대기)
