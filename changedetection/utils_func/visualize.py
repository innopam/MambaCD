import numpy as np
import matplotlib.pyplot as plt

def visualize_class_map(class_map, mode):
    # 클래스 맵을 색상 맵으로 변환
    if mode:
        color_map = np.array([[0, 0, 0],       # 배경 (black)
                          [255, 0, 0],     # 신축 (red)
                          [0, 255, 0],     # 소멸 (green)
                          [0, 0, 255],     # 갱신 (blue)
                          [255, 255, 0],   # 색상변화 (yellow)
                          [255, 0, 255],   # 더미 1 (pink)
                          [0, 255, 255],   # 더미 2 (cyan)
                          [255, 255, 255]])  # 더미 3 (white)
    else:
        color_map = np.array([[0, 0, 0],       # 배경 (black)
                          [255, 255, 255]])  # 변화 (white)
    colored_map = color_map[class_map]
    return colored_map

def visualize_confidence_map(confidence_map):
    """
    신뢰도 맵을 시각화하는 함수.

    Parameters:
    confidence_map (numpy.ndarray): 신뢰도 맵, 2D 배열 (H, W) 형태로 입력.

    Returns:
    numpy.ndarray: 시각화된 컬러 맵 이미지.
    """
    # 신뢰도 맵을 0-1 범위로 정규화
    normalized_map = (confidence_map - np.min(confidence_map)) / (np.max(confidence_map) - np.min(confidence_map))

    # 컬러 맵 적용 (여기서는 'viridis' 사용)
    colormap = plt.get_cmap('viridis')
    colored_map = colormap(normalized_map)  # RGBA 형태로 변환

    # RGBA 이미지를 0-255 범위의 uint8로 변환
    colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)  # alpha 채널 제거하고 0-255로 변환

    return colored_map
