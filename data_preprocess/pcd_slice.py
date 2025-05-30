import numpy as np
import cv2
import open3d as o3d
import os
import json
from typing import Tuple, Optional
from pathlib import Path

def load_point_cloud(file_path: str) -> Optional[o3d.geometry.PointCloud]:
    """
    加载点云文件
    
    Args:
        file_path (str): 点云文件路径
        
    Returns:
        Optional[o3d.geometry.PointCloud]: 加载的点云，如果失败则返回None
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            print(f"警告: 文件不包含点: {file_path}")
            return None
        return pcd
    except Exception as e:
        print(f"加载点云文件出错: {e}")
        return None

def slice_point_cloud(pcd: o3d.geometry.PointCloud, height: float, thickness: float) -> Tuple[Optional[o3d.geometry.PointCloud], Tuple[float, float]]:
    """
    对点云进行切片
    
    Args:
        pcd (o3d.geometry.PointCloud): 原始点云
        height (float): 切片起始高度
        thickness (float): 切片厚度
        
    Returns:
        Tuple[Optional[o3d.geometry.PointCloud], Tuple[float, float]]: (切片后的点云, 实际高度范围)
    """
    if pcd is None:
        return None, (0, 0)
    
    # 计算切片范围
    start_height = height
    end_height = height + thickness
    
    # 提取切片内的点
    points = np.asarray(pcd.points)
    mask = (points[:, 1] >= start_height) & (points[:, 1] <= end_height)  # 使用y轴坐标
    sliced_points = points[mask]
    
    if len(sliced_points) == 0:
        return None, (start_height, end_height)
    
    # 创建新的点云
    sliced_pcd = o3d.geometry.PointCloud()
    sliced_pcd.points = o3d.utility.Vector3dVector(sliced_points)
    
    # 如果有颜色信息，也进行切片
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        sliced_colors = colors[mask]
        sliced_pcd.colors = o3d.utility.Vector3dVector(sliced_colors)
    
    return sliced_pcd, (start_height, end_height)

def calculate_bounds(points: np.ndarray) -> Tuple[float, float, float, float]:
    """
    计算点的边界
    
    Args:
        points (np.ndarray): 点坐标数组
        
    Returns:
        Tuple[float, float, float, float]: (x_min, x_max, z_min, z_max)
    """
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    z_min = np.min(points[:, 2])  # 使用z轴坐标
    z_max = np.max(points[:, 2])  # 使用z轴坐标
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

def draw_points(canvas: np.ndarray, points: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), size: int = 2):
    """
    在画布上绘制点
    
    Args:
        canvas (np.ndarray): 画布图像
        points (np.ndarray): 要绘制的点坐标
        color (Tuple[int, int, int]): 点的颜色，默认为红色
        size (int): 点的大小
    """
    for point in points:
        cv2.circle(canvas, tuple(point), size, color, -1)

def process_slice(pcd: o3d.geometry.PointCloud, height: float, thickness: float, output_path: str) -> bool:
    """
    处理点云切片并生成图像
    
    Args:
        pcd (o3d.geometry.PointCloud): 原始点云
        height (float): 切片起始高度
        thickness (float): 切片厚度
        output_path (str): 输出图像路径
        
    Returns:
        bool: 是否处理成功
    """
    try:
        # 对点云进行切片
        sliced_pcd, height_range = slice_point_cloud(pcd, height, thickness)
        if sliced_pcd is None:
            print(f"警告: 在高度范围 {height_range} 内没有找到点")
            return False
        
        # 获取切片后的点
        points = np.asarray(sliced_pcd.points)
        
        # 计算边界
        x_min, x_max, z_min, z_max = calculate_bounds(points)
        
        # 创建画布和变换矩阵
        canvas, T1 = create_canvas(x_min, x_max, z_min, z_max)
        
        # 投影点
        projected_points = project_points(points, T1)
        
        # 绘制点
        draw_points(canvas, projected_points)
        
        # 保存图像
        cv2.imwrite(output_path, canvas)
        print(f"切片图像已保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"处理切片时出错: {e}")
        return False

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
        with open('data_preprocess/file_mapping.json', 'r') as f:
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

if __name__ == "__main__":
    # 示例用法
    input_file = "data/raw/BIMNet/point_cloud/train/1pXnuDYAj8r.txt"  # 输入点云文件
    output_dir = "data_preprocess/output"     # 输出目录
    height = 2.3             # 切片起始高度（米）
    thickness = 0.05         # 切片厚度（米）
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载点云
    pcd = load_BIMNet_point_cloud(input_file)  # 使用新的加载函数
    if pcd is None:
        print("无法加载点云文件")
        exit(1)
    
    # 处理切片并生成图像
    output_path = os.path.join(output_dir, f"slice_{height:.2f}m_{thickness:.2f}m.png")
    success = process_slice(pcd, height, thickness, output_path)
    
    if not success:
        print("处理切片失败") 