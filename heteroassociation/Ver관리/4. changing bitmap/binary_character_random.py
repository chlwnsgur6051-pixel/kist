"""
Random balanced HS (letter) bitmap design.

각 출력 픽셀마다: 1이 정확히 5개, 0이 정확히 5개 배치
→ 최대한 orthogonal한 설계로 linear separability 향상

digit 0~9 → letter A~J (각 letter는 15비트)
"""
import numpy as np
import string
import matplotlib.pyplot as plt

# 고정 시드 (재현성)
np.random.seed(0)

# 두 그룹
digits = [str(i) for i in range(10)]      # '0'~'9'
letters = list(string.ascii_uppercase[:10])  # 'A'~'J'

# Digit 패턴 (기존)
digit_patterns = {
    '0': ["111", "101", "101", "101", "111"],
    '1': ["010", "110", "010", "010", "111"],
    '2': ["111", "001", "111", "100", "111"],
    '3': ["111", "001", "111", "001", "111"],
    '4': ["101", "101", "111", "001", "001"],
    '5': ["111", "100", "111", "001", "111"],
    '6': ["111", "100", "111", "101", "111"],
    '7': ["111", "001", "010", "010", "010"],
    '8': ["111", "101", "111", "101", "111"],
    '9': ["111", "101", "111", "001", "001"],
}

# HS 패턴: 각 출력 픽셀마다 1이 5개, 0이 5개 (균형)
# letter_matrix[d, j] = digit d에서 letter 픽셀 j의 값
letter_matrix = np.zeros((10, 15), dtype=np.uint8)
for col in range(15):
    # 각 열마다 10개 중 5개를 1로 설정 (random shuffle)
    ones_idx = np.random.choice(10, 5, replace=False)
    letter_matrix[ones_idx, col] = 1

letter_patterns = {}
for d in range(10):
    letter_patterns[letters[d]] = letter_matrix[d].reshape(5, 3)

# ===== 데이터셋 생성 =====
def make_dataset(chars, pattern_dict):
    images, labels = [], []
    for i, ch in enumerate(chars):
        if isinstance(pattern_dict[ch], (list, tuple)):
            # digit: 문자열 리스트
            arr = np.array([[int(c) for c in row] for row in pattern_dict[ch]], dtype=np.uint8)
        else:
            # letter: numpy array
            arr = pattern_dict[ch].astype(np.uint8)
        images.append(arr.flatten())
        labels.append(i)
    return np.stack(images), np.array(labels)

def make_digit_dataset():
    return make_dataset(digits, digit_patterns)

def make_letter_dataset():
    return make_dataset(letters, letter_patterns)


def show_images(images, nrow=5, title="Images"):
    n = len(images)
    plt.figure(figsize=(nrow, (n + nrow - 1)//nrow * 2))
    for i in range(n):
        plt.subplot((n + nrow - 1)//nrow, nrow, i+1)
        plt.imshow(images[i].reshape(5, 3), cmap='gray_r')
        plt.title(f"{i}", fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 데이터셋 확인
    X, _ = make_digit_dataset()
    Y, _ = make_letter_dataset()
    print(f"Digit images shape: {X.shape}")
    print(f"Letter images shape: {Y.shape}")

    # HS 비트맵 통계
    print("\n=== HS (Letter) Bitmap 통계 ===")
    print(f"각 픽셀당 1의 개수:")
    for j in range(15):
        count_ones = letter_matrix[:, j].sum()
        print(f"  픽셀 {j:2d}: {count_ones}/10 (완벽한 균형: {count_ones == 5})")

    # 시각화
    print("\nDigit 비트맵:")
    show_images(X, nrow=5, title="Digits (0~9)")
    print("\nLetter 비트맵 (Random Balanced HS):")
    show_images(Y, nrow=5, title="Letters A~J (Balanced)")
