"""단층(single-layer) + temporal 코딩으로 XNOR 을 푸는 프로토타입.

핵심 아이디어
-------------
비선형을 '은닉층'이 아니라 '뉴런 dynamics(coincidence)'에서 얻는다.
- 입력: 2-phase temporal (비트=1 -> 이른 스파이크 t_A, 비트=0 -> 늦은 스파이크 t_B)
- 뉴런: 짧은 tau + threshold + 리셋 = coincidence detector
        -> 같은 빈에 모인 입력만 합쳐지고, 두 빈(t_A/t_B)을 독립 평가
        -> "둘 다 같은 빈에 모이면(=값이 같으면) 발화" = XNOR
- 가중치: 단층 벡터 하나 (은닉층 없음).  single<theta<sum 밴드면 XNOR 성립.

왜 단층으로 되나: 두 빈의 합을 S1, S0 라 하면 출력은 max(S1,S0) 에 대한
threshold (both-tails) 라, 단일 선형투영의 '비단조' 함수 -> 선형 threshold가
못 하던 XNOR 가능. (XNOR 은 양수 가중치로 됨. XOR 은 억제/반전이 필요.)

구성
----
1) 미분가능한 both-tails 모델로 단층 가중치 w, threshold 학습 (XNOR)
2) 학습된 w, theta 를 '실제 스파이킹 LIF(짧은 tau+리셋)' 시뮬레이션에 넣어 확인
3) plt 시각화: 입력 래스터 / 손실 / 스파이킹 막전위 궤적 / 결과

실행: python "refer 논문/code/booltest_singlelayer.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 데이터: XNOR (값이 같으면 1)
# ---------------------------------------------------------------------------
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[1.], [0.], [0.], [1.]])          # XNOR
pat_names = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]

# temporal 인코딩 시각
T_A, T_B = 1.0, 2.5      # 비트=1 -> t_A(이른), 비트=0 -> t_B(늦은)

# ---------------------------------------------------------------------------
# 1) 미분가능 both-tails 단층 모델 학습
#    S1 = w·x (t_A 빈 합, 값=1인 입력),  S0 = w·(1-x) (t_B 빈 합, 값=0인 입력)
#    발화 점수 = max(S1, S0),  발화 <=> max >= theta
# ---------------------------------------------------------------------------
w = nn.Parameter(torch.tensor([0.5, 0.5]))
theta = nn.Parameter(torch.tensor(0.8))
opt = torch.optim.Adam([w, theta], lr=0.03)
loss_fn = nn.BCEWithLogitsLoss()

loss_hist = []
for epoch in range(2000):
    opt.zero_grad()
    S1 = X @ w                        # [4] 값=1 입력들의 가중합 (t_A 빈)
    S0 = (1.0 - X) @ w                # [4] 값=0 입력들의 가중합 (t_B 빈)
    fire_score = torch.maximum(S1, S0)
    logit = (4.0 * (fire_score - theta)).unsqueeze(1)
    loss = loss_fn(logit, Y)
    loss.backward()
    opt.step()
    loss_hist.append(float(loss))

w_np = w.detach().numpy()
theta_v = float(theta.detach())
print(f"학습된 가중치 w = {w_np},  theta = {theta_v:.3f}")
print(f"밴드 조건 확인: 단일 max(w)={w_np.max():.3f} < theta={theta_v:.3f} < 합 {w_np.sum():.3f} ?")

# ---------------------------------------------------------------------------
# 2) 실제 스파이킹 LIF(짧은 tau + 리셋) 시뮬레이션으로 확인
# ---------------------------------------------------------------------------
DT = 0.02
TAU = 0.25            # 짧게 -> coincidence
T_MAX = 4.0
DECAY = np.exp(-DT / TAU)
n_steps = int(round(T_MAX / DT))
t_axis = np.arange(n_steps) * DT
step_A, step_B = int(round(T_A / DT)), int(round(T_B / DT))


def simulate_lif(bits, w, theta):
    """단층 LIF coincidence 뉴런 1개 시뮬. bits:(2,). 막전위 궤적과 스파이크 반환."""
    v = 0.0
    v_trace, spikes = [], []
    for step in range(n_steps):
        I = 0.0
        if step == step_A:                       # 값=1 입력들이 t_A 에 발화
            I += sum(w[i] for i in range(2) if bits[i] > 0.5)
        if step == step_B:                       # 값=0 입력들이 t_B 에 발화
            I += sum(w[i] for i in range(2) if bits[i] <= 0.5)
        v = v * DECAY + I
        fired = v >= theta
        if fired:
            spikes.append(step)
            v = 0.0                              # 리셋
        v_trace.append(v if not fired else theta)  # 발화 순간은 theta 로 찍어 표시
    return np.array(v_trace), spikes


sim_traces, sim_pred = [], []
for p in range(4):
    tr, sp = simulate_lif(X[p].numpy(), w_np, theta_v)
    sim_traces.append(tr)
    sim_pred.append(1 if len(sp) >= 1 else 0)     # 한 번이라도 발화하면 1
sim_pred = np.array(sim_pred)
tgt = Y.squeeze(1).numpy().astype(int)

# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
# (a) 입력 래스터
fig1, ax = plt.subplots(figsize=(7, 3.2))
col = ["tab:blue", "tab:red"]
for p in range(4):
    for n in range(2):
        t = T_A if X[p, n] > 0.5 else T_B
        ax.scatter(t, p, s=90, color=col[n], edgecolors="k", zorder=3)
ax.axvline(T_A, color="tab:blue", ls=":", alpha=.6, label=f"bit=1 -> t_A={T_A}")
ax.axvline(T_B, color="tab:red", ls=":", alpha=.6, label=f"bit=0 -> t_B={T_B}")
ax.set_yticks(range(4)); ax.set_yticklabels(pat_names)
ax.set_xlabel("spike time"); ax.set_title("2-phase input raster (blue=n0, red=n1)")
ax.set_xlim(0, T_MAX); ax.legend(fontsize=8)
fig1.tight_layout(); fig1.savefig(RESULTS / "sl_1_input.png", dpi=120)

# (b) 손실
fig2, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(loss_hist, color="tab:red"); ax.set_title("Training loss (single-layer both-tails)")
ax.set_xlabel("epoch"); ax.set_ylabel("BCE")
fig2.tight_layout(); fig2.savefig(RESULTS / "sl_2_loss.png", dpi=120)

# (c) 스파이킹 막전위 궤적 (4 패턴) — 두 볼리 coincidence
fig3, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
for p, axp in enumerate(axes.ravel()):
    lab = tgt[p]
    axp.plot(t_axis, sim_traces[p], color=("tab:red" if lab == 1 else "tab:blue"), lw=1.5)
    axp.axhline(theta_v, color="k", ls=":", lw=1, label=f"theta={theta_v:.2f}")
    axp.axvline(T_A, color="tab:blue", ls="--", alpha=.4)
    axp.axvline(T_B, color="tab:red", ls="--", alpha=.4)
    fired = "FIRE" if sim_pred[p] == 1 else "silent"
    ok = "O" if sim_pred[p] == lab else "X"
    axp.set_title(f"{pat_names[p]}  target={lab}  pred={sim_pred[p]} [{fired}] {ok}",
                  color=("green" if sim_pred[p] == lab else "red"))
    axp.legend(fontsize=8, loc="upper right")
fig3.suptitle("Single-layer LIF (short tau + reset) membrane — two volleys coincidence", fontsize=12)
fig3.supxlabel("time"); fig3.supylabel("membrane v")
fig3.tight_layout(); fig3.savefig(RESULTS / "sl_3_membrane.png", dpi=120)

# (d) 결과
fig4, ax = plt.subplots(figsize=(6.5, 3.4))
x = np.arange(4)
ax.bar(x - 0.2, tgt, width=0.4, label="target (XNOR)", color="tab:gray")
ax.bar(x + 0.2, sim_pred, width=0.4, label="LIF pred", color="tab:orange")
for i in range(4):
    ax.text(i, 1.05, "O" if sim_pred[i] == tgt[i] else "X", ha="center",
            color=("green" if sim_pred[i] == tgt[i] else "red"), fontsize=13, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(pat_names); ax.set_ylim(0, 1.25)
ax.set_title(f"Single-layer temporal XNOR  ({int((sim_pred==tgt).sum())}/4 correct)")
ax.legend(fontsize=9, loc="upper left")
fig4.tight_layout(); fig4.savefig(RESULTS / "sl_4_result.png", dpi=120)

print("\n=== 스파이킹 LIF 확인 ===")
for p in range(4):
    print(f"{pat_names[p]}  target={tgt[p]}  pred={sim_pred[p]}  "
          f"{'O' if sim_pred[p]==tgt[p] else 'X'}")
print(f"정확도: {int((sim_pred==tgt).sum())}/4")
print(f"그림: {RESULTS}")

plt.show()
