"""linear basis에서 랭크가 9인 이유를 진단.

phi_aug = [숫자패턴(10x15) | 편향열(10x1)] 의 SVD 를 구해,
0 에 가까운 특이값에 대응하는 왼쪽 특이벡터(행 종속 조합 계수 c)를 찾는다.

  c^T @ phi_aug ≈ 0   ⟹   sum_i c_i * [digit_i | 1] ≈ 0

즉 어떤 숫자 패턴이 나머지 숫자들의 (가중)합으로 표현된다는 뜻.
quadratic 도 같이 돌려 랭크가 10 으로 회복되는지 비교한다.

이 스크립트는 tools/ 안에 있으므로, 프로젝트 루트를 import 경로에 추가한다.
결과는 tools/results/rank_diagnosis/ 에 저장된다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import binary_character as data
from contrastive_optimal_weight import expand_features

OUT = Path(__file__).resolve().parent / "results" / "rank_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)

digit_images, _ = data.make_digit_dataset()
cs = torch.tensor(digit_images.reshape(10, -1), dtype=torch.float32)  # (10, 15)


def build_phi_aug(basis):
    phi = expand_features(cs, basis)
    bias = torch.ones(phi.shape[0], 1, dtype=phi.dtype)
    return torch.cat([phi, bias], dim=1)


def analyze(basis, lines):
    phi_aug = build_phi_aug(basis)
    n_rows = phi_aug.shape[0]
    rank = int(torch.linalg.matrix_rank(phi_aug).item())

    # full SVD: U(10,10), S(10,), Vh(D,D)
    U, S, Vh = torch.linalg.svd(phi_aug, full_matrices=True)
    sv = S.detach().numpy()

    lines.append(f"\n================  basis = {basis}  ================")
    lines.append(f"phi_aug shape : {tuple(phi_aug.shape)}")
    lines.append(f"rank          : {rank}  (최대 가능 = {min(phi_aug.shape)})")
    lines.append("singular values (내림차순):")
    lines.append("  " + ", ".join(f"{v:.4f}" for v in sv))

    # 0 에 가까운 특이값 = 종속 방향. tol 은 numpy 기본 랭크 판정과 유사하게.
    tol = sv.max() * max(phi_aug.shape) * np.finfo(np.float32).eps
    null_idx = [k for k in range(len(sv)) if sv[k] <= tol]
    lines.append(f"tol           : {tol:.3e}")
    lines.append(f"near-zero SV  : {[f'{sv[k]:.2e}' for k in null_idx]}  (개수 ={len(null_idx)})")

    if not null_idx:
        lines.append("→ 종속 방향 없음. 모든 행이 독립 → 완전 재현 가능 (rank = 행수).")
        return

    # 각 종속 방향마다 행 조합 계수 c = U[:, k]
    for k in null_idx:
        c = U[:, k].detach().numpy()          # (10,)  행(=숫자) 결합 계수
        combo = phi_aug.t().detach().numpy() @ c   # sum_i c_i * row_i  (D,)
        lines.append(f"\n--- 종속 방향 (특이값 {sv[k]:.2e}) ---")
        lines.append("행 결합 계수 c (숫자 0~9 순):")
        lines.append("  " + ", ".join(f"{ci:+.3f}" for ci in c))
        lines.append(f"‖sum_i c_i*row_i‖ = {np.linalg.norm(combo):.3e}  (0에 가까울수록 완전 종속)")

        # 가장 큰 |c| 를 가진 숫자를 나머지의 조합으로 표현
        m = int(np.argmax(np.abs(c)))
        coeffs = -c / c[m]                     # digit_m = sum_{i!=m} coeffs_i * digit_i
        terms = [f"({coeffs[i]:+.3f})*[{i}]" for i in range(n_rows) if i != m and abs(coeffs[i]) > 1e-4]
        lines.append(f"⇒ 숫자[{m}] ≈ " + " + ".join(terms))

        # 재구성 오차 확인 (편향 제외한 15비트 패턴 기준)
        digits_np = cs.detach().numpy()
        recon = sum(coeffs[i] * digits_np[i] for i in range(n_rows) if i != m)
        err = np.linalg.norm(recon - digits_np[m])
        lines.append(f"   재구성 오차 ‖recon - 숫자[{m}]‖ = {err:.3e}")


lines = ["=== 랭크 종속성 진단 (linear vs quadratic) ==="]
analyze("linear", lines)
analyze("quadratic", lines)

text = "\n".join(lines)
(OUT / "report.txt").write_text(text, encoding="utf-8")
print(text)
print(f"\nsaved: {OUT/'report.txt'}")
