"""expand_features(루프판) vs expand_features_vector(벡터판) 결과/속도 검증.

- 실제 숫자 데이터셋(CS)을 두 함수에 통과시킨다.
- 각각의 결과를 results/expand_features_check/ 아래 텍스트 파일로 저장한다.
- 두 결과가 완전히 같은지 비교한다.
- timeit 으로 두 함수의 연산 속도(반복 평균)를 측정한다.

이 스크립트는 tools/ 안에 있으므로, 프로젝트 루트를 import 경로에 추가한다.
결과는 tools/results/expand_features_check/ 에 저장된다.
"""
import sys
import timeit
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import binary_character as data
from contrastive_optimal_weight import expand_features, expand_features_vector

OUT = Path(__file__).resolve().parent / "results" / "expand_features_check"
OUT.mkdir(parents=True, exist_ok=True)

# 입력 CS: 숫자 10개 x 15비트
digit_images, _ = data.make_digit_dataset()
cs = torch.tensor(digit_images.reshape(10, -1), dtype=torch.float32)

basis = "quadratic"
loop_out = expand_features(cs, basis)          # 기존 이중 루프
vec_out = expand_features_vector(cs, basis)     # 벡터화 버전

# 각 결과를 파일로 저장 (사람이 읽을 수 있는 정수 행렬 형태)
np.savetxt(OUT / "expand_loop.txt", loop_out.cpu().numpy(), fmt="%d",
           header=f"expand_features (loop)   shape={tuple(loop_out.shape)}")
np.savetxt(OUT / "expand_vector.txt", vec_out.cpu().numpy(), fmt="%d",
           header=f"expand_features_vector   shape={tuple(vec_out.shape)}")

# 비교
same_shape = loop_out.shape == vec_out.shape
exact_equal = bool(torch.equal(loop_out, vec_out))
max_abs_diff = float((loop_out - vec_out).abs().max()) if same_shape else float("nan")

# 속도 측정: 단일 호출은 너무 빠르므로 여러 번 반복해 평균(초/호출)을 낸다.
N_REPEAT = 100
loop_sec = timeit.timeit(lambda: expand_features(cs, basis), number=N_REPEAT) / N_REPEAT
vec_sec = timeit.timeit(lambda: expand_features_vector(cs, basis), number=N_REPEAT) / N_REPEAT
speedup = loop_sec / vec_sec if vec_sec > 0 else float("inf")

report = [
    "=== expand_features vs expand_features_vector ===",
    f"basis            : {basis}",
    f"loop   shape     : {tuple(loop_out.shape)}",
    f"vector shape     : {tuple(vec_out.shape)}",
    f"same shape       : {same_shape}",
    f"exact equal      : {exact_equal}",
    f"max abs diff     : {max_abs_diff}",
    f"saved            : {OUT/'expand_loop.txt'}",
    f"                 : {OUT/'expand_vector.txt'}",
    "RESULT           : " + ("IDENTICAL ✅" if exact_equal else "DIFFERENT ❌"),
    "",
    f"--- speed (avg over {N_REPEAT} runs, cs shape={tuple(cs.shape)}) ---",
    f"loop    : {loop_sec*1e6:10.2f} us/call",
    f"vector  : {vec_sec*1e6:10.2f} us/call",
    f"speedup : {speedup:10.2f}x  (loop / vector)",
]
text = "\n".join(report)
(OUT / "compare_report.txt").write_text(text, encoding="utf-8")
print(text)
