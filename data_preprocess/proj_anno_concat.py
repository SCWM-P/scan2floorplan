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
import math
from scipy.spatial import Delaunay
ROOT_DIR = Path(__file__).resolve().parent.parent

############################################################
############### 加载点云模型和IFC模型 #################
############################################################

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
        with open(ROOT_DIR / 'data_preprocess/file_mapping.json', 'r') as f:
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
        #     print(f"成功添加构件 {i}, 包含 {len(component_points)} 个角点")
        # else:
        #     print(f"警告: 构件 {i} 没有有效的角点")
    
    print(f"\n总共解析出 {len(points_list)} 个构件")
    return points_list, ifc_labels

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


############################################################
######################### 开始创建投影图 ####################
############################################################

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

def process_combined_projection(pcd_path: str, intersection_file: str, output_dir: str) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[str], float]:
    """
    处理点云投影和IFC构件的组合，返回处理结果
    
    Args:
        pcd_path (str): 点云文件路径
        intersection_file (str): IFC构件文件路径
        output_dir (str): 输出目录
        
    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[str], float]: 
            - 点云投影图像
            - 组合图像
            - 凸包顶点列表
            - IFC标签列表
            - 缩放比例
    """
    # 加载点云
    pcd = load_BIMNet_point_cloud(pcd_path)
    if pcd is None:
        print(f"无法加载点云文件: {pcd_path}")
        return None, None, [], [], 0
    
    # 解析IFC构件
    points_list, ifc_labels = parse_ifc_components(intersection_file)
    if not points_list:
        print(f"无法解析IFC构件文件: {intersection_file}")
        return None, None, [], [], 0
    
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
    
    # 画布四周加200像素白边
    margin = 200
    canvas_width = int(width * scale) + 2 * margin
    canvas_height = int(height * scale) + 2 * margin
    
    print(f"原始尺寸: {width:.2f} x {height:.2f}")
    print(f"画布尺寸: {canvas_width} x {canvas_height}")
    
    # 创建变换矩阵，平移部分加margin
    T1 = np.array([
        [scale, 0, -x_min * scale + margin],
        [0, -scale, z_max * scale + margin],
        [0, 0, 1]
    ])
    
    # 生成点云密度图
    density_map = project_point_cloud(pcd, T1, canvas_width, canvas_height)
    if density_map is None:
        print("警告: 无法生成点云密度图")
        return None, None, [], [], 0
    
    # 创建点云投影图像（灰度图）
    pcd_canvas = cv2.cvtColor(density_map, cv2.COLOR_GRAY2BGR)
    
    # 创建组合图像
    canvas = pcd_canvas.copy()
    
    # 在绘制IFC构件时收集hull和标签
    hull_points_list = []
    ifc_labels_list = []
    
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
        
        # 收集hull和标签
        hull_points_list.append(hull)
        ifc_labels_list.append(ifc_labels[i])
        
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
    
    return pcd_canvas, canvas, hull_points_list, ifc_labels_list, scale

def save_projection_images(pcd_canvas: np.ndarray, combined_canvas: np.ndarray, pcd_output_path: str, combined_output_path: str) -> Tuple[str, str]:
    """
    保存投影图像
    
    Args:
        pcd_canvas (np.ndarray): 点云投影图像
        combined_canvas (np.ndarray): 组合图像
        pcd_output_path (str): 点云投影图像保存路径
        combined_output_path (str): 组合图像保存路径
        
    Returns:
        Tuple[str, str]: 保存的图像路径（点云投影图像路径，组合图像路径）
    """
    # 保存点云投影图像
    cv2.imwrite(pcd_output_path, pcd_canvas)
    print(f"点云投影图像已保存到: {pcd_output_path}")
    
    # 保存组合图像
    cv2.imwrite(combined_output_path, combined_canvas)
    print(f"组合图像已保存到: {combined_output_path}")
    
    return pcd_output_path, combined_output_path

############################################################
############### 从这里开始创建labelme的标注 #################
############################################################

def expand_hull_to_polygon(hull_points: np.ndarray, extend: float, scale: float, pcd_image_path: str) -> np.ndarray:
    """
    将凸包边界向外膨胀，方向为中心到每个顶点的方向（xy交换，符号不变）
    然后对膨胀区域内的非白色点求凸包作为精确边界
    
    Args:
        hull_points (np.ndarray): 凸包顶点
        extend (float): 膨胀距离（实际距离）
        scale (float): 缩放比例
        pcd_image_path (str): 点云投影图像路径
        
    Returns:
        np.ndarray: 膨胀后的多边形顶点
    """
    extend_pixels = extend * scale
    center = np.mean(hull_points, axis=0)
    expanded_points = []
    for pt in hull_points:
        direction = pt - center
        dx, dy = direction[0], direction[1]
        swapped_direction = np.array([
            np.copysign(abs(dy), dx),
            np.copysign(abs(dx), dy)
        ])
        norm = np.linalg.norm(swapped_direction)
        if norm == 0:
            expanded_points.append(pt)
        else:
            unit_dir = swapped_direction / norm
            expanded_pt = pt + unit_dir * extend_pixels
            expanded_points.append(expanded_pt)
    
    # 将膨胀后的点转换为numpy数组
    expanded_points = np.array(expanded_points)
    
    # 读取点云投影图像
    if not os.path.exists(pcd_image_path):
        print(f"警告: 找不到点云投影图像: {pcd_image_path}")
        return expanded_points
        
    pcd_image = cv2.imread(pcd_image_path, cv2.IMREAD_GRAYSCALE)
    if pcd_image is None:
        print("警告: 无法读取点云投影图像")
        return expanded_points
    
    # 使用原始图像的大小创建掩码
    height, width = pcd_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    expanded_points_int = expanded_points.astype(np.int32)
    cv2.fillPoly(mask, [expanded_points_int], 255)
    
    # 1. 找到膨胀区域内的所有点
    y_coords, x_coords = np.where(mask > 0)
    points_in_region = np.column_stack((x_coords, y_coords))
    
    # 2. 获取这些点的灰度值
    gray_values = pcd_image[y_coords, x_coords]
    
    # 3. 筛选出灰度值小于200的点
    dark_points_mask = gray_values < 225
    dark_points = points_in_region[dark_points_mask]
    
    # 如果有符合条件的点，计算它们的凹包
    if len(dark_points) > 0:
        # 计算凹包，alpha值可以根据需要调整
        exact_expand_points = get_concave_hull(dark_points, alpha=0.05)  # alpha小一点，会更接近于凸包，这样效果更好。大概可能0.05或者0.1合适。
        return exact_expand_points
    
    return expanded_points

def create_labelme_annotation(polygon_points: np.ndarray, ifc_label: str, image_path: str) -> Dict:
    """
    创建labelme格式的标注
    
    Args:
        polygon_points (np.ndarray): 多边形顶点
        ifc_label (str): IFC标签
        image_path (str): 图像路径
        
    Returns:
        Dict: labelme格式的标注
    """
    # 将点转换为列表格式
    points = polygon_points.tolist()
    
    # 创建labelme格式的标注
    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": ifc_label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": None,
        "imageWidth": None
    }
    
    return annotation

def process_hull_to_labelme(hull_points: np.ndarray, ifc_label: str, extend: float, scale: float, image_path: str) -> Dict:
    """
    处理凸包，生成labelme格式的标注
    
    Args:
        hull_points (np.ndarray): 凸包顶点
        ifc_label (str): IFC标签
        extend (float): 膨胀距离（实际距离）
        scale (float): 缩放比例
        image_path (str): 图像路径
        
    Returns:
        Dict: labelme格式的标注
    """
    # 膨胀凸包
    expanded_points = expand_hull_to_polygon(hull_points, extend, scale, image_path)
    
    # 创建labelme标注
    annotation = create_labelme_annotation(expanded_points, ifc_label, image_path)
    
    return annotation

def process_multiple_hulls_to_labelme(hull_points_list: List[np.ndarray], ifc_labels_list: List[str], extend: float, scale: float, image_path: str, pcd_image_path: str) -> Dict:
    """
    处理多个凸包，生成labelme标准格式的标注
    
    Args:
        hull_points_list (List[np.ndarray]): 凸包顶点列表
        ifc_labels_list (List[str]): IFC标签列表
        extend (float): 膨胀距离（实际距离）
        scale (float): 缩放比例
        image_path (str): 图像路径
        pcd_image_path (str): 点云投影图像路径
        
    Returns:
        Dict: labelme标准格式的标注
    """
    import cv2
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    shapes = []
    for hull_points, ifc_label in zip(hull_points_list, ifc_labels_list):
        expanded_points = expand_hull_to_polygon(hull_points, extend, scale, pcd_image_path)
        shape = {
            "label": ifc_label,
            "points": expanded_points.tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }
    return annotation

def get_concave_hull(points_2d: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    计算二维点的凹包（Alpha Shape）
    
    Args:
        points_2d (np.ndarray): 二维点集
        alpha (float): Alpha参数，控制凹包的程度。值越大，结果越接近凸包；值越小，结果越接近点集
        
    Returns:
        np.ndarray: 凹包顶点
    """
    if len(points_2d) < 3:
        return points_2d
        
    # 1. 计算Delaunay三角剖分
    tri = Delaunay(points_2d)
    
    # 2. 计算每个三角形的外接圆半径
    circum_r = []
    for i in range(len(tri.simplices)):
        # 获取三角形的三个顶点
        coords = points_2d[tri.simplices[i]]
        # 计算三边长
        a = np.sqrt(np.sum((coords[0] - coords[1]) ** 2))
        b = np.sqrt(np.sum((coords[1] - coords[2]) ** 2))
        c = np.sqrt(np.sum((coords[2] - coords[0]) ** 2))
        # 计算半周长
        s = (a + b + c) / 2.0
        # 计算面积
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # 计算外接圆半径
        circum_r.append(a * b * c / (4.0 * area))
    
    # 3. 根据alpha值筛选三角形
    # 如果外接圆半径小于1/alpha，则保留该三角形
    alpha_mask = np.array(circum_r) < 1.0/alpha
    valid_triangles = tri.simplices[alpha_mask]
    
    # 4. 提取边界边
    edges = []
    for triangle in valid_triangles:
        for i in range(3):
            edge = tuple(sorted([triangle[i], triangle[(i+1)%3]]))
            edges.append(edge)
    
    # 5. 统计每条边出现的次数
    edge_counts = {}
    for edge in edges:
        edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    # 6. 只保留出现一次的边（这些边就是凹包的边界）
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    # 7. 将边界边连接成有序的顶点序列
    if not boundary_edges:
        return get_convex_hull(points_2d)  # 如果没有边界边，返回凸包
        
    # 构建邻接表
    adjacency = {}
    for edge in boundary_edges:
        v1, v2 = edge
        if v1 not in adjacency:
            adjacency[v1] = []
        if v2 not in adjacency:
            adjacency[v2] = []
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)
    
    # 找到起始点（选择度数最小的点）
    start_point = min(adjacency.keys(), key=lambda x: len(adjacency[x]))
    
    # 构建有序的顶点序列
    ordered_points = [start_point]
    current = start_point
    while True:
        neighbors = adjacency[current]
        if not neighbors:
            break
        next_point = neighbors[0]
        ordered_points.append(next_point)
        # 移除已访问的边
        adjacency[current].remove(next_point)
        adjacency[next_point].remove(current)
        current = next_point
        if current == start_point:
            break
    
    # 8. 返回有序的顶点坐标
    return points_2d[ordered_points]

def main():
    """
    主函数
    """
    # 示例用法
    pcd_path = ROOT_DIR / "data/raw/BIMNet/point_cloud/train/1pXnuDYAj8r.txt"
    intersection_file = ROOT_DIR / "data_preprocess/tmp.txt"
    output_dir = ROOT_DIR / "data_preprocess/output"
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 1. 处理投影并获取结果
    pcd_canvas, combined_canvas, hull_points_list, ifc_labels_list, scale = process_combined_projection(
        pcd_path, intersection_file, output_dir
    )
    
    if pcd_canvas is None:
        print("处理投影失败")
        return
    
    # 2. 保存投影图像
    pcd_output_path = output_dir / 'projection_only_pcd.png'
    combined_output_path = output_dir / 'projection_combined.png'
    pcd_image_path, combined_image_path = save_projection_images(
        pcd_canvas, 
        combined_canvas, 
        pcd_output_path,
        combined_output_path
    )
    
    # 3. 生成labelme标注（使用点云投影图像作为参考）
    extend = 0.1  # 膨胀距离（米）
    annotation = process_multiple_hulls_to_labelme(
        hull_points_list, 
        ifc_labels_list, 
        extend, 
        scale, 
        pcd_image_path,
        pcd_image_path
    )
    
    # 4. 保存标注文件
    annotation_path = output_dir / 'projection_only_pcd.json' # 更改标注文件名以匹配图像
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    print(f"标注文件已保存到: {annotation_path}")

if __name__ == "__main__":
    main() 