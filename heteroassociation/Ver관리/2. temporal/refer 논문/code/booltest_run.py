"""booltest.py 의 AlphaSynapseSNN 을 실제로 XOR 로 학습·시각화하는 실행 스크립트.

- 입력: 2-phase temporal 코딩 (비트=1 -> 이른 스파이크 t_A, 비트=0 -> 늦은 스파이크 t_B)
- 모델: booltest.AlphaSynapseSNN (입력2 -> 은닉 -> 출력1, alpha 시냅스)
- 학습: 최종 누적 output_potential 를 로짓으로 -> BCEWithLogits 로 XOR 회귀 (backprop)
- 시각화(중간중간 plt):
    1) alpha 커널 h(t)=t*exp(-t/tau) 모양
    2) 4개 XOR 입력의 스파이크 시간 래스터
    3) 학습 손실 곡선
    4) 학습 후 은닉/출력 막전위의 시간 궤적 (XOR=0 vs 1 색 구분)
    5) 최종 예측 vs 정답 막대

원본 booltest.py 는 수정하지 않는다. 그림은 results/ 에 저장 + 창 표시.
실행: python "refer 논문/code/booltest_run.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from booltest import AlphaSynapseSNN   # 원본 클래스 그대로 사용

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# 하이퍼파라미터 / 데이터
# ---------------------------------------------------------------------------
T_EARLY = 1.0        # 비트=1 -> 이른 스파이크
T_LATE = 2.5         # 비트=0 -> 늦은 스파이크
T_MAX = 6.0          # 시뮬 시간축 끝
N_STEPS = 120
HIDDEN_DIM = 8
EPOCHS = 3000
LR = 0.03

time_windows = torch.linspace(0.0, T_MAX, N_STEPS)

# XOR 진리표
inputs_bits = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([0.0, 1.0, 1.0, 0.0]).unsqueeze(1)   # XOR
pat_names = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]


def bits_to_spike_times(bits: torch.Tensor) -> torch.Tensor:
    """비트 -> 스파이크 시각. 1->T_EARLY, 0->T_LATE (2-phase)."""
    return torch.where(bits > 0.5,
                       torch.full_like(bits, T_EARLY),
                       torch.full_like(bits, T_LATE))


spike_times = bits_to_spike_times(inputs_bits)   # [4, 2]


# ---------------------------------------------------------------------------
# forward 전 구간 재현 (시각화용 중간값 추출). booltest.forward 와 동일한 수식.
# ---------------------------------------------------------------------------
def forward_all(model: AlphaSynapseSNN, spk: torch.Tensor, tw: torch.Tensor):
    B, Tn = spk.size(0), tw.size(0)
    t_exp = tw.view(1, Tn, 1)
    s_exp = spk.view(B, 1, -1)
    dt = torch.clamp(t_exp - s_exp, min=0.0)
    psp = model.alpha_kernel(dt)                          # [B, T, Din]
    hidden_current = torch.matmul(psp, model.w1)          # [B, T, H]
    hidden_potential = torch.cumsum(hidden_current, dim=1)
    output_current = torch.matmul(hidden_potential, model.w2)
    output_potential = torch.cumsum(output_current, dim=1)  # [B, T, 1]
    return psp, hidden_potential, output_potential


def logits_from(output_potential: torch.Tensor) -> torch.Tensor:
    """readout: 최종 누적 막전위(정규화)를 로짓으로."""
    return output_potential[:, -1, 0:1] / N_STEPS


# ---------------------------------------------------------------------------
# 1) alpha 커널 모양 시각화 (학습 전, 개념 확인)
# ---------------------------------------------------------------------------
model = AlphaSynapseSNN(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=1)

with torch.no_grad():
    tk = torch.linspace(0, T_MAX, 200)
    hk = model.alpha_kernel(tk).cpu().numpy()
fig1, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(tk.numpy(), hk, color="tab:purple", lw=2)
ax.axvline(float(model.tau), color="gray", ls="--", lw=1, label=f"peak at t=tau={model.tau}")
ax.set_title("Alpha synapse kernel  h(t) = t * exp(-t/tau)")
ax.set_xlabel("t - t_spike"); ax.set_ylabel("PSP"); ax.legend(fontsize=9)
fig1.tight_layout(); fig1.savefig(RESULTS / "1_alpha_kernel.png", dpi=120)

# ---------------------------------------------------------------------------
# 2) 입력 스파이크 시간 래스터 (2-phase 인코딩 확인)
# ---------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(7, 3.5))
colors = ["tab:blue", "tab:red"]
for p in range(4):
    for n in range(2):
        t = float(spike_times[p, n])
        ax.scatter(t, p, s=90, color=colors[n],
                   marker="o", edgecolors="k", zorder=3)
ax.axvline(T_EARLY, color="tab:blue", ls=":", alpha=0.6, label=f"bit=1 -> t_A={T_EARLY}")
ax.axvline(T_LATE, color="tab:red", ls=":", alpha=0.6, label=f"bit=0 -> t_B={T_LATE}")
ax.set_yticks(range(4)); ax.set_yticklabels(pat_names)
ax.set_xlabel("spike time"); ax.set_ylabel("XOR input pattern")
ax.set_title("Input spike-time raster (2-phase temporal coding)\nblue=neuron0, red=neuron1")
ax.set_xlim(0, T_MAX); ax.legend(fontsize=8, loc="upper right")
fig2.tight_layout(); fig2.savefig(RESULTS / "2_input_raster.png", dpi=120)

# ---------------------------------------------------------------------------
# 3) 학습
# ---------------------------------------------------------------------------
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()
loss_hist, acc_hist = [], []

for epoch in range(EPOCHS):
    opt.zero_grad()
    out = model(spike_times, time_windows)      # 원본 forward 사용
    logits = out[:, -1, 0:1] / N_STEPS
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()

    loss_hist.append(float(loss))
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).float()
        acc_hist.append(float((pred == labels).float().mean()))
    if epoch % 300 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:4d}  loss={loss:.4f}  acc={acc_hist[-1]:.2f}")

fig3, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.5))
a0.plot(loss_hist, color="tab:red"); a0.set_title("Training loss (BCE)")
a0.set_xlabel("epoch"); a0.set_ylabel("loss")
a1.plot(acc_hist, color="tab:green"); a1.set_title("Accuracy (4 XOR cases)")
a1.set_xlabel("epoch"); a1.set_ylabel("acc"); a1.set_ylim(-0.05, 1.05)
fig3.tight_layout(); fig3.savefig(RESULTS / "3_training.png", dpi=120)

# ---------------------------------------------------------------------------
# 4) 학습 후 막전위 시간 궤적 (은닉 대표 + 출력)
# ---------------------------------------------------------------------------
with torch.no_grad():
    psp, hidden_pot, out_pot = forward_all(model, spike_times, time_windows)
    out_pot_n = (out_pot[:, :, 0] / N_STEPS).cpu().numpy()      # [4, T]
    hidden_pot_np = hidden_pot.cpu().numpy()                     # [4, T, H]
tw_np = time_windows.cpu().numpy()

fig4, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
# 출력 막전위 (정규화 로짓 스케일) — XOR 라벨 색
for p in range(4):
    lab = int(labels[p, 0])
    a0.plot(tw_np, out_pot_n[p], lw=1.8,
            color=("tab:red" if lab == 1 else "tab:blue"),
            label=f"{pat_names[p]} -> {lab}")
a0.axhline(0.0, color="gray", ls="--", lw=0.8)
a0.set_title("Output potential over time (red=XOR 1, blue=XOR 0)")
a0.set_xlabel("time"); a0.set_ylabel("output (readout scale)")
a0.legend(fontsize=8)
# 은닉 뉴런 0의 막전위 (패턴별)
for p in range(4):
    lab = int(labels[p, 0])
    a1.plot(tw_np, hidden_pot_np[p, :, 0], lw=1.4,
            color=("tab:red" if lab == 1 else "tab:blue"), alpha=0.8,
            label=f"{pat_names[p]}")
a1.set_title("Hidden neuron 0 potential over time")
a1.set_xlabel("time"); a1.set_ylabel("hidden potential"); a1.legend(fontsize=8)
fig4.tight_layout(); fig4.savefig(RESULTS / "4_membrane_traces.png", dpi=120)

# ---------------------------------------------------------------------------
# 5) 최종 예측 vs 정답
# ---------------------------------------------------------------------------
with torch.no_grad():
    probs = torch.sigmoid(logits_from(out_pot)).squeeze(1).cpu().numpy()
targets = labels.squeeze(1).cpu().numpy()
preds = (probs >= 0.5).astype(int)

fig5, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(4)
ax.bar(x - 0.2, targets, width=0.4, label="target (XOR)", color="tab:gray")
ax.bar(x + 0.2, probs, width=0.4, label="pred prob", color="tab:orange")
ax.axhline(0.5, color="k", ls=":", lw=1)
for i in range(4):
    ok = preds[i] == int(targets[i])
    ax.text(i, 1.03, "O" if ok else "X", ha="center",
            color=("green" if ok else "red"), fontsize=12, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(pat_names)
ax.set_ylim(0, 1.15); ax.set_ylabel("value")
ax.set_title(f"XOR result   ({int((preds==targets).sum())}/4 correct)")
ax.legend(fontsize=9, loc="upper left")
fig5.tight_layout(); fig5.savefig(RESULTS / "5_result.png", dpi=120)

print("\n=== 최종 결과 ===")
for i in range(4):
    print(f"{pat_names[i]} -> pred_prob={probs[i]:.3f}  pred={preds[i]}  target={int(targets[i])}"
          f"  {'O' if preds[i]==int(targets[i]) else 'X'}")
print(f"정확도: {int((preds==targets).sum())}/4")
print(f"그림 저장: {RESULTS}")

plt.show()
