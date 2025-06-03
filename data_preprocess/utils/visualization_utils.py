# data_preprocess/utils/visualization_utils.py
import cv2
import numpy as np
from typing import List, Tuple


def display_image_with_polygons(image: np.ndarray, polygons: List[np.ndarray], window_name: str = "Preview"):
    """简单显示带有绘制多边形的图像。"""
    display_img = image.copy()
    for poly in polygons:
        if poly is not None and len(poly) > 0:
            cv2.polylines(display_img, [poly.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

    cv2.imshow(window_name, display_img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)