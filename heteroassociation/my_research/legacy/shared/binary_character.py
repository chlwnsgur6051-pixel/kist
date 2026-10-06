import numpy as np
import string
import matplotlib.pyplot as plt

# 두 그룹으로 나누기
digits = [str(i) for i in range(10)]      # '0'~'9'
letters = list(string.ascii_uppercase[:10])  # 'A'~'J'

# 공통 3x5 패턴
# 5x3 (세로 5, 가로 3) 패턴으로 0~9, A~J 이진 이미지
patterns = {
    '0': [
        "111",
        "101",
        "101",
        "101",
        "111"
    ],
    '1': [
        "010",
        "110",
        "010",
        "010",
        "111"
    ],
    '2': [
        "111",
        "001",
        "111",
        "100",
        "111"
    ],
    '3': [
        "111",
        "001",
        "111",
        "001",
        "111"
    ],
    '4': [
        "101",
        "101",
        "111",
        "001",
        "001"
    ],
    '5': [
        "111",
        "100",
        "111",
        "001",
        "111"
    ],
    '6': [
        "111",
        "100",
        "111",
        "101",
        "111"
    ],
    '7': [
        "111",
        "001",
        "010",
        "010",
        "010"
    ],
    '8': [
        "111",
        "101",
        "111",
        "101",
        "111"
    ],
    # [변경됨] 마지막 행 "111" -> "001": digit 행렬 rank 9 -> 10 (XNOR 종속 제거).
    #          이 변경으로 선형 단층이 10/10 가능. 원본 복원은 마지막 행을 "111"로.
    '9': [
        "111",
        "101",
        "111",
        "001",
        "001"
    ],
    'A': [
        "111",
        "101",
        "111",
        "101",
        "101"
    ],
    'B': [
        "110",
        "101",
        "110",
        "101",
        "110"
    ],
    'C': [
        "111",
        "100",
        "100",
        "100",
        "111"
    ],
    'D': [
        "110",
        "101",
        "101",
        "101",
        "110"
    ],
    'E': [
        "111",
        "100",
        "110",
        "100",
        "111"
    ],
    'F': [
        "111",
        "100",
        "110",
        "100",
        "100"
    ],
    'G': [
        "111",
        "100",
        "101",
        "101",
        "111"
    ],
    'H': [
        "101",
        "101",
        "111",
        "101",
        "101"
    ],
    'I': [
        "111",
        "010",
        "010",
        "010",
        "111"
    ],
    'J': [
        "111",
        "001",
        "001",
        "101",
        "111"
    ]
}

def make_dataset(chars):
    images, labels = [], []
    for i, ch in enumerate(chars):
        arr = np.array([[int(c) for c in row] for row in patterns[ch]], dtype=np.uint8)
        images.append(arr)
        labels.append(i)
    return np.stack(images), np.array(labels)

def make_digit_dataset():
    return make_dataset(digits)

def make_letter_dataset():
    return make_dataset(letters)

def show_images(images, nrow=5, title="Images"):
    n = len(images)
    plt.figure(figsize=(nrow, (n + nrow - 1)//nrow * 2))
    for i in range(n):
        plt.subplot((n + nrow - 1)//nrow, nrow, i+1)
        plt.imshow(images[i], cmap='gray_r')
        # plt.axis('off')
        # plt.title(f"{chars[labels[i]]}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 숫자 / 문자 데이터셋 생성
    digit_images, digit_labels = make_dataset(digits)
    letter_images, letter_labels = make_dataset(letters)

    print("Digits:", digit_images.shape, digit_labels)
    print("Letters:", letter_images.shape, letter_labels)

    # 샘플 확인
    for i, ch in enumerate(digits[:3]):
        print(f"\n{ch}:")
        print(digit_images[i])

    for i, ch in enumerate(letters[:3]):
        print(f"\n{ch}:")
        print(letter_images[i])
