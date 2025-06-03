# 目的：该文件用于将点云的垂直投影和IFC构件的切片交点合并，并生成组合图像
# 输入：点云文件路径，IFC交点文件路径，输出目录
# 输出：组合图像

import numpy as np
import cv2
import open3d as o3d
import os
import json
import re
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path

def load_transform_matrix(file_path: str) -> Optional[np.ndarray]:
    """
    加载变换矩阵
    
    Args:
        file_path (str): 变换矩阵文件路径
        
    Returns:
        Optional[np.ndarray]: 4x4变换矩阵，如果失败则返回None
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 解析4x4矩阵
        matrix = []
        for line in lines:
            values = line.strip().split()
            if len(values) == 4:
                matrix.append([float(v) for v in values])
        
        if len(matrix) != 4:
            print(f"警告: 变换矩阵格式不正确: {file_path}")
            return None
            
        return np.array(matrix)
    except Exception as e:
        print(f"加载变换矩阵出错: {e}")
        return None

def get_transform_matrix_path(point_cloud_path: str) -> Optional[str]:
    """
    根据点云文件路径获取对应的变换矩阵文件路径
    
    Args:
        point_cloud_path (str): 点云文件路径
        
    Returns:
        Optional[str]: 变换矩阵文件路径，如果失败则返回None
    """
    try:
        # 加载文件映射
        with open('../data_preprocess/file_mapping.json', 'r') as f:
            mapping = json.load(f)
        
        # 获取点云文件名
        pc_filename = os.path.basename(point_cloud_path)
        
        # 查找对应的IFC文件名
        ifc_filename = None
        for ifc, pc in mapping.items():
            if pc == pc_filename:
                ifc_filename = ifc
                break
        
        if ifc_filename is None:
            print(f"警告: 未找到点云文件 {pc_filename} 对应的IFC文件")
            return None
        
        # 构建变换矩阵文件路径
        # 从点云路径中提取train/test目录
        path_parts = Path(point_cloud_path).parts
        train_test_dir = None
        for part in path_parts:
            if part in ['train', 'test']:
                train_test_dir = part
                break
        
        if train_test_dir is None:
            print(f"警告: 无法从路径中识别train/test目录: {point_cloud_path}")
            return None
        
        # 构建变换矩阵文件路径
        # point_cloud和mat_pc2obj是同级目录
        transform_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(point_cloud_path))),  # 回到point_cloud的上级目录
            'mat_pc2obj',  # 进入mat_pc2obj目录
            train_test_dir,  # 进入对应的train/test目录
            ifc_filename.replace('.ifc', '.txt')  # 替换文件扩展名
        )
        
        # 打印调试信息
        print(f"点云文件路径: {point_cloud_path}")
        print(f"变换矩阵文件路径: {transform_path}")
        
        return transform_path
    except Exception as e:
        print(f"获取变换矩阵路径出错: {e}")
        return None

def load_BIMNet_point_cloud(file_path: str) -> Optional[o3d.geometry.PointCloud]:
    """
    加载BIMNet格式的点云文件并应用变换矩阵
    
    Args:
        file_path (str): 点云文件路径
        
    Returns:
        Optional[o3d.geometry.PointCloud]: 加载并变换后的点云，如果失败则返回None
    """
    try:
        # 读取txt文件
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 解析每一行
        points = []
        colors = []
        for line in lines:
            # 分割每一行
            values = line.strip().split()
            if len(values) < 6:  # 至少需要6个值（x,y,z,r,g,b）
                continue
                
            # 提取坐标和颜色
            x, y, z = float(values[0]), float(values[1]), float(values[2])
            r, g, b = float(values[3]), float(values[4]), float(values[5])
            
            points.append([x, y, z])
            colors.append([r, g, b])
        
        if not points:
            print(f"警告: 文件不包含有效点: {file_path}")
            return None
        
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 加载并应用变换矩阵
        transform_path = get_transform_matrix_path(file_path)
        if transform_path is None:
            print(f"警告: 无法获取变换矩阵路径")
            return None
        
        transform_matrix = load_transform_matrix(transform_path)
        if transform_matrix is None:
            print(f"警告: 无法加载变换矩阵")
            return None
            
        # 应用变换
        pcd.transform(transform_matrix)
        
        return pcd
    except Exception as e:
        print(f"加载BIMNet点云文件出错: {e}")
        return None

def parse_ifc_components(file_path: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    从文件中解析IFC构件的角点信息
    
    Args:
        file_path (str): IFC构件文件路径
        
    Returns:
        Tuple[List[np.ndarray], List[str]]: (每个构件的角点数组列表, 对应的IFC类型标签列表)
    """
    points_list = []
    ifc_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 移除最后四行
    text = ''.join(lines[:-4])
    
    # 分割成构件
    components = text.split("构件 ")
    
    for i, component in enumerate(components[1:], 1):
        if "IFC类型:" not in component or "几何体 No." not in component:
            continue
            
        # 提取IFC类型
        ifc_type = component.split("IFC类型:")[1].split("\n")[0].strip()
        print(f"解析构件 {i}, IFC类型: {ifc_type}")
        
        # 分割成几何体
        geometry_blocks = component.split("几何体 No.")
        
        # 收集该构件的所有点
        component_points = []
        for j, block in enumerate(geometry_blocks[1:], 1):
            geometry_number = block.split(":")[0]
            coordinates_text = block.split(":")[1].strip()
            
            # 处理坐标文本
            lines = [line.strip() for line in coordinates_text.split('\n') if line.strip()]
            for line in lines:
                # 移除所有方括号并分割
                numbers = line.replace('[', '').replace(']', '').split(',')
                try:
                    point = [float(num.strip()) for num in numbers if num.strip()]
                    if len(point) == 3:
                        component_points.append(point)
                    else:
                        print(f"警告: 构件 {i} 几何体 {j} 的坐标维度不正确: {point}")
                except ValueError as e:
                    print(f"警告: 构件 {i} 几何体 {j} 的坐标解析失败: {line}")
                    continue
        
        if component_points:
            points_list.append(np.array(component_points))
            ifc_labels.append(ifc_type)
            print(f"成功添加构件 {i}, 包含 {len(component_points)} 个角点")
        else:
            print(f"警告: 构件 {i} 没有有效的角点")
    
    print(f"\n总共解析出 {len(points_list)} 个构件")
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
    
    # 计算缩放比例
    scale = max(2048 / width, 2048 / height)
    
    # 计算画布尺寸
    canvas_width = int(width * scale)
    canvas_height = int(height * scale)
    
    # 创建空白画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 计算变换矩阵
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
    if len(points_2d) < 3:
        return points_2d
    
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

def project_point_cloud(pcd: o3d.geometry.PointCloud, T1: np.ndarray, canvas_width: int, canvas_height: int) -> Optional[np.ndarray]:
    """
    将点云投影到平面上，并计算密度图
    
    Args:
        pcd (o3d.geometry.PointCloud): 原始点云
        T1 (np.ndarray): 变换矩阵
        canvas_width (int): 画布宽度
        canvas_height (int): 画布高度
        
    Returns:
        Optional[np.ndarray]: 密度图
    """
    if pcd is None:
        return None
    
    # 获取所有点
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        return None
    
    # 创建密度图（初始化为白色）
    density_map = np.ones((canvas_height, canvas_width), dtype=np.float32) * 255
    
    # 创建临时密度计数图
    density_count = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    
    # 投影点并计算密度
    for point in points:
        # 投影点
        homogeneous = np.array([point[0], point[2], 1])
        projected = (T1 @ homogeneous)[:2].astype(int)
        
        # 确保点在图像范围内
        if (0 <= projected[0] < canvas_width and 
            0 <= projected[1] < canvas_height):
            density_count[projected[1], projected[0]] += 1
    
    # 归一化密度计数
    if np.max(density_count) > 0:
        # 使用90%分位数的密度值作为归一化基准
        density_max = np.percentile(density_count[density_count > 0], 95)
        density_count = np.clip(density_count / density_max, 0, 1)
        # 将密度值映射到0-255范围，并反转（使密度高的地方更暗）
        density_map = 255 - (density_count * 255).astype(np.uint8)
    
    return density_map

def process_combined_projection(pcd_path: str, intersection_file: str, output_dir: str):
    """
    处理点云投影和IFC构件的组合
    
    Args:
        pcd_path (str): 点云文件路径
        intersection_file (str): IFC构件文件路径
        output_dir (str): 输出目录
    """
    # 加载点云
    pcd = load_BIMNet_point_cloud(pcd_path)
    if pcd is None:
        print(f"无法加载点云文件: {pcd_path}")
        return
        
    # 解析IFC构件
    points_list, ifc_labels = parse_ifc_components(intersection_file)
    if not points_list:
        print(f"无法解析IFC构件文件: {intersection_file}")
        return
    
    # 获取点云坐标
    pcd_points = np.asarray(pcd.points)
    
    # 计算所有点的边界（包括点云和IFC构件）
    x_min = float('inf')
    x_max = float('-inf')
    z_min = float('inf')
    z_max = float('-inf')
    
    # 计算点云边界
    if len(pcd_points) > 0:
        x_min = min(x_min, np.min(pcd_points[:, 0]))
        x_max = max(x_max, np.max(pcd_points[:, 0]))
        z_min = min(z_min, np.min(pcd_points[:, 2]))
        z_max = max(z_max, np.max(pcd_points[:, 2]))
    
    # 计算IFC构件边界
    for points in points_list:
        if len(points) == 0:
            continue
        x_min = min(x_min, np.min(points[:, 0]))
        x_max = max(x_max, np.max(points[:, 0]))
        z_min = min(z_min, np.min(points[:, 2]))
        z_max = max(z_max, np.max(points[:, 2]))
    
    # 计算实际尺寸
    width = x_max - x_min
    height = z_max - z_min
    
    # 计算缩放比例（保持宽高比）
    max_dimension = 2048
    scale = min(max_dimension / width, max_dimension / height)
    
    # 计算画布尺寸（保持宽高比）
    canvas_width = int(width * scale)
    canvas_height = int(height * scale)
    
    print(f"原始尺寸: {width:.2f} x {height:.2f}")
    print(f"画布尺寸: {canvas_width} x {canvas_height}")
    
    # 创建变换矩阵
    T1 = np.array([
        [scale, 0, -x_min * scale],
        [0, -scale, z_max * scale],
        [0, 0, 1]
    ])
    
    # 生成点云密度图
    density_map = project_point_cloud(pcd, T1, canvas_width, canvas_height)
    if density_map is None:
        print("警告: 无法生成点云密度图")
        return
    
    # 创建彩色画布
    canvas = cv2.cvtColor(density_map, cv2.COLOR_GRAY2BGR)
    
    # 绘制IFC构件
    colors = [generate_random_color() for _ in range(len(points_list))]
    for i, points in enumerate(points_list):
        if len(points) == 0:
            continue
            
        # 投影点（直接在这里进行投影变换）
        homogeneous = np.column_stack((points[:, [0, 2]], np.ones(len(points))))
        projected_points = (T1 @ homogeneous.T).T[:, :2].astype(int)
        
        # 计算凸包
        hull = get_convex_hull(projected_points)
        
        # 绘制原始点
        for point in projected_points:
            cv2.circle(canvas, tuple(point), 2, colors[i], -1)
        
        # 绘制凸包
        draw_convex_hull(canvas, hull, colors[i])
        
        # 添加IFC标签
        if len(hull) > 0:
            center = np.mean(hull, axis=0).astype(int)
            cv2.putText(canvas, ifc_labels[i], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
    
    # 保存图像
    output_path = os.path.join(output_dir, 'projection_combined.png')
    cv2.imwrite(output_path, canvas)
    print(f"组合图像已保存到: {output_path}")

def main():
    """
    主函数
    """
    # 示例用法
    pcd_path = "../data/raw/BIMNet/point_cloud/train/1pXnuDYAj8r.txt"  # 输入点云文件
    intersection_file = "tmp.txt"  # IFC交点文件
    output_dir = "output"  # 输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    process_combined_projection(pcd_path, intersection_file, output_dir)

if __name__ == "__main__":
    main() 