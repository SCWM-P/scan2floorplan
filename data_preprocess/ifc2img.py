# 目的：该文件用于将IFC构件的切片交点和IFC类型标签转换为图像

import numpy as np
import cv2
import re
from typing import List, Tuple, Dict
import os
import random

def parse_intersection_points(text: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    从文本中解析切片交点和IFC类型标签
    
    Args:
        text (str): 包含切片交点的文本
        
    Returns:
        Tuple[List[np.ndarray], List[str]]: (每个构件的所有几何体交点数组列表, 对应的IFC类型标签列表)
    """
    points_list = []
    ifc_labels = []
    # 移除最后一行
    lines = text.split('\n')
    if lines:
        text = '\n'.join(lines[:-1])
    # 分割成构件
    components = text.split("构件 ")
    for i, component in enumerate(components[1:], 1):  # 跳过第一个空字符串
        # 找到IFC类型和切片交点部分
        if "IFC类型:" not in component or "切片交点:" not in component:
            continue
        # 提取IFC类型
        ifc_type = component.split("IFC类型:")[1].split("\n")[0].strip()
        # 提取切片交点部分
        points_text = component.split("切片交点:")[1]
        # 分割成几何体
        geometry_blocks = points_text.split("几何体 No.")
        # 收集该构件的所有点
        component_points = []
        for j, block in enumerate(geometry_blocks[1:], 1):  # 跳过第一个空字符串
            # 提取几何体编号和坐标部分
            geometry_number = block.split(":")[0]
            coordinates_text = block.split(":")[1].strip()
            # 清理字符串并分割成行
            lines = [line.strip() for line in coordinates_text.split('\n') if line.strip()]
            for line in lines:
                # 移除方括号和逗号，分割成数字
                numbers = line.replace('[', '').replace(']', '').split(',')
                # 转换每个数字为float
                point = [float(num.strip()) for num in numbers if num.strip()]
                if len(point) == 3:  # 确保是3D坐标
                    component_points.append(point)
        if component_points:
            points_list.append(np.array(component_points))
            ifc_labels.append(ifc_type)
    print(f"总共解析出 {len(points_list)} 个构件")
    return points_list, ifc_labels

def calculate_bounds(points_list: List[np.ndarray]) -> Tuple[float, float, float, float]:
    """
    计算所有点的边界

    Args:
        points_list (List[np.ndarray]): 所有几何体的交点列表
        
    Returns:
        Tuple[float, float, float, float]: (x_min, x_max, z_min, z_max)
    """
    x_min = float('inf')
    x_max = float('-inf')
    z_min = float('inf')
    z_max = float('-inf')
    
    for points in points_list:
        if len(points) == 0:
            continue
        x_min = min(x_min, np.min(points[:, 0]))
        x_max = max(x_max, np.max(points[:, 0]))
        z_min = min(z_min, np.min(points[:, 2]))
        z_max = max(z_max, np.max(points[:, 2]))
    
    return x_min, x_max, z_min, z_max

def create_canvas(x_min: float, x_max: float, z_min: float, z_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建画布并计算变换矩阵
    
    Args:
        x_min, x_max, z_min, z_max (float): 边界值
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (画布图像, 变换矩阵)
    """
    # 计算实际尺寸
    width = x_max - x_min
    height = z_max - z_min
    scale = max(2048 / width, 2048 / height)  # 将10240改回1024
    
    # 计算画布尺寸
    canvas_width = int(width * scale)
    canvas_height = int(height * scale)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 计算变换矩阵
    # 变换矩阵将实际坐标映射到画布坐标
    T1 = np.array([
        [scale, 0, -x_min * scale],
        [0, -scale, z_max * scale],  # 注意Y轴方向
        [0, 0, 1]
    ])
    
    return canvas, T1

def project_points(points: np.ndarray, T1: np.ndarray) -> np.ndarray:
    """
    将点投影到画布上
    Args:
        points (np.ndarray): 原始点坐标
        T1 (np.ndarray): 变换矩阵
    Returns:
        np.ndarray: 投影后的点坐标
    """
    # 添加齐次坐标
    homogeneous = np.column_stack((points[:, [0, 2]], np.ones(len(points))))
    # 应用变换
    projected = (T1 @ homogeneous.T).T
    # 转换为整数坐标
    return projected[:, :2].astype(int)

def get_convex_hull(points_2d: np.ndarray) -> np.ndarray:
    """
    计算二维点的凸包
    
    Args:
        points_2d (np.ndarray): 二维点集
        
    Returns:
        np.ndarray: 凸包顶点
    """
    # 确保点集至少包含3个点
    if len(points_2d) < 3:
        # print(f"警告：点集数量不足3个，无法形成凸包。当前点数量：{len(points_2d)}")
        return points_2d
    # 计算凸包
    hull = cv2.convexHull(points_2d.astype(np.float32))
    return hull.reshape(-1, 2)

def generate_random_color() -> Tuple[int, int, int]:
    """
    生成随机颜色
    Returns:
        Tuple[int, int, int]: BGR格式的颜色
    """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_convex_hull(canvas: np.ndarray, hull_points: np.ndarray, color: Tuple[int, int, int]):
    """
    在画布上绘制凸包
    Args:
        canvas (np.ndarray): 画布图像
        hull_points (np.ndarray): 凸包顶点
        color (Tuple[int, int, int]): 绘制颜色
    """
    # 绘制凸包顶点
    for point in hull_points:
        cv2.circle(canvas, tuple(point.astype(int)), 3, color, -1)
    # 绘制凸包边
    for i in range(len(hull_points)):
        pt1 = tuple(hull_points[i].astype(int))
        pt2 = tuple(hull_points[(i + 1) % len(hull_points)].astype(int))
        cv2.line(canvas, pt1, pt2, color, 2)

def process_file(input_file: str, output_dir: str):
    """
    处理输入文件并生成图像
    Args:
        input_file (str): 输入文件路径
        output_dir (str): 输出目录
    """
    # 读取文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    # 解析切片交点和IFC标签
    points_list, ifc_labels = parse_intersection_points(content)
    # 计算边界
    x_min, x_max, z_min, z_max = calculate_bounds(points_list)
    # 创建画布和变换矩阵
    canvas, T1 = create_canvas(x_min, x_max, z_min, z_max)
    # 为每个构件生成随机颜色
    colors = [generate_random_color() for _ in range(len(points_list))]
    # 绘制每个构件的点和凸包
    for i, points in enumerate(points_list):
        if len(points) == 0:
            continue
        # 投影点
        projected_points = project_points(points, T1)
        # 计算凸包（使用该构件的所有点）
        hull = get_convex_hull(projected_points)
        # 绘制原始点
        for point in projected_points:
            cv2.circle(canvas, tuple(point), 2, colors[i], -1)
        # 绘制凸包
        draw_convex_hull(canvas, hull, colors[i])
        # 在图像上添加IFC标签
        if len(hull) > 0:
            # 计算标签位置（使用凸包的中心点）
            center = np.mean(hull, axis=0).astype(int)
            cv2.putText(canvas, ifc_labels[i], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
    
    # 保存图像
    output_path = os.path.join(output_dir, 'slice_visualization.png')
    cv2.imwrite(output_path, canvas)
    print(f"图像绘制完成，已保存到: {output_path}")
    # 返回解析结果
    return points_list, ifc_labels

if __name__ == "__main__":
    input_file = "tmp.txt"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    process_file(input_file, output_dir)