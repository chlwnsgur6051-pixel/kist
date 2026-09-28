"""binary_character.show_images() 원본 함수를 그대로 활용해 데이터셋을 시각화.

show_images 는 내부에서 plt.show() 를 호출한다. 여기서는 저장을 먼저 하려고
그 내부 show() 를 잠시 무력화한 뒤 figure 를 저장하고, 스크립트 끝에서 한 번에
창을 띄운다(GUI 환경). 창이 없는 환경이면 저장만 되고 표시는 건너뛴다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # 백엔드 자동 선택: GUI 있으면 창 표시, 없으면 저장만

import binary_character as data

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

_noop = lambda *a, **k: None


def save_with_show_images(images, title, filename, nrow=5):
    # show_images 내부의 plt.show() 를 잠시 no-op 으로 바꿔 figure 를 유지한 채 저장
    real_show = plt.show
    plt.show = _noop
    try:
        data.show_images(images, nrow=nrow, title=title)  # 원본 함수 그대로 사용
    finally:
        plt.show = real_show
    out = RESULTS / filename
    plt.gcf().savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved: {out}")


digit_images, _ = data.make_digit_dataset()
letter_images, _ = data.make_letter_dataset()

save_with_show_images(digit_images, "Digits (CS) 0-9", "show_images_digits.png", nrow=5)
save_with_show_images(letter_images, "Letters (US) A-J", "show_images_letters.png", nrow=5)

plt.show()  # 저장된 두 figure 를 창으로 표시 (GUI 환경에서, 닫을 때까지 대기)
