import numpy as np

def visualize_class_map(class_map):
    # 예시: 클래스 0은 배경(흰색), 1은 신축(빨강), 2는 소멸(초록), 등으로 색상 지정
    color_map = np.array([[0, 0, 0],       # 배경 (black)
                          [255, 0, 0],     # 신축 (red)
                          [0, 255, 0],     # 소멸 (green)
                          [0, 0, 255],     # 갱신 (blue)
                          [255, 255, 0]])  # 색상변화 (yellow)

    # 클래스 맵을 색상 맵으로 변환
    colored_map = color_map[class_map]
    return colored_map
