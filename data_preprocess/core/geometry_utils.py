# data_preprocess/core/geometry_utils.py
import numpy as np
import trimesh
import cv2  # OpenCV for convex hull
from typing import List, Optional, Tuple


def slice_component_geometries_at_height(
        component_geometries: List[trimesh.Trimesh],
        slice_y_level: float,
        slice_thickness: float
) -> List[np.ndarray]:
    """
    对一个构件的所有几何体进行切片，返回每个几何体在XZ平面上的2D切片轮廓点集。
    注意：高度轴是Y轴。

    Args:
        component_geometries (List[trimesh.Trimesh]): 构件的trimesh几何体列表。
        slice_y_level (float): 切片中心Y坐标。
        slice_thickness (float): 切片的厚度。

    Returns:
        List[np.ndarray]: 每个成功切片的几何体的2D轮廓点集 (Nx2, XZ平面) 列表。
                         如果某个几何体没有交点，则不包含在返回列表中。
    """
    all_slice_polygons_xz = []

    # 定义切片平面的法向量 (指向Y轴正方向) 和原点
    plane_normal = [0, 1, 0]
    plane_origin_bottom = [0, slice_y_level - slice_thickness / 2, 0]
    plane_origin_top = [0, slice_y_level + slice_thickness / 2, 0]

    for mesh in component_geometries:
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            continue

        try:
            # 使用 trimesh 的 slice_plane 功能来模拟厚切片
            # 实际上是进行两次切片，然后找到两次切片之间的部分
            # 或者更简单的方式是，先用 bounding_box_oriented 切割出大致范围，再取中心切片
            # 这里我们简化为在中心高度进行一次精确切片，假设厚度内的变化不大

            # 创建一个非常薄的box来近似厚切片（或者多次切片取并集）
            # 为了简化，这里我们只在中心高度切一刀
            path3d_list = mesh.section_multiplane(plane_origin=[0, slice_y_level, 0],
                                                  plane_normal=plane_normal,
                                                  heights=[0])  # heights相对于plane_origin

            if path3d_list and path3d_list[0] is not None:
                # path3d_list[0] 是一个 (N, M, 3) 的 ndarray, N是路径数, M是每条路径的点数
                # 我们需要将所有路径合并，并只取XZ坐标
                # 对于每个闭合轮廓，它应该是一个独立的标注
                for path_segment_group in path3d_list:  # path_segment_group 可能是多个不相连的切片轮廓
                    if path_segment_group is None: continue
                    for path_segment in path_segment_group:  # path_segment 是一条 (M,3) 的路径
                        if len(path_segment) >= 3:  # 至少3个点才能形成一个多边形
                            # 提取 X 和 Z 坐标 (Y值是切片高度)
                            slice_points_xz = path_segment[:, [0, 2]]
                            all_slice_polygons_xz.append(slice_points_xz.astype(np.float32))

        except Exception as e:
            # trimesh 切片可能会因为各种几何问题失败
            # print(f"Warning: Trimesh slicing failed for a geometry: {e}")
            pass

    return all_slice_polygons_xz


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    应用4x4变换矩阵到3D点集 (Nx3 or Nx4)。
    如果点是Nx3，则转换为齐次坐标。
    """
    if points.shape[1] == 3:
        points_h = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
    elif points.shape[1] == 4:
        points_h = points
    else:
        raise ValueError("Input points must be Nx3 or Nx4.")

    transformed_points_h = points_h @ matrix.T

    # 转换回非齐次坐标
    transformed_points = transformed_points_h[:, :3] / transformed_points_h[:, 3, np.newaxis]
    return transformed_points


def calculate_convex_hull_2d(points_2d: np.ndarray) -> Optional[np.ndarray]:
    """
    计算2D点集的凸包。
    Args:
        points_2d (np.ndarray): Nx2 点集数组。
    Returns:
        Optional[np.ndarray]: Mx2 凸包顶点数组，如果点数少于3则返回None。
    """
    if points_2d is None or points_2d.shape[0] < 3:
        return None
    try:
        # OpenCV 需要 float32 类型
        hull = cv2.convexHull(points_2d.astype(np.float32))
        return hull.squeeze()  # 去掉不必要的维度
    except Exception as e:
        # print(f"Error calculating convex hull: {e}")
        return None


def calculate_min_area_rect_2d(points_2d: np.ndarray) -> Optional[np.ndarray]:
    """
    计算2D点集的最小外接矩形。
    Args:
        points_2d (np.ndarray): Nx2 点集数组。
    Returns:
        Optional[np.ndarray]: 4x2 最小外接矩形的四个角点，如果点数少于3则返回None。
    """
    if points_2d is None or points_2d.shape[0] < 3:
        return None
    try:
        rect = cv2.minAreaRect(points_2d.astype(np.float32))
        box_points = cv2.boxPoints(rect)
        return box_points.astype(np.float32)
    except Exception as e:
        # print(f"Error calculating min area rectangle: {e}")
        return None


def project_to_xz_plane(points_3d: np.ndarray) -> np.ndarray:
    """将3D点投影到XZ平面 (即丢弃Y坐标)。"""
    if points_3d.ndim == 1 and points_3d.shape[0] == 3:  # 单个点
        return points_3d[[0, 2]]
    if points_3d.ndim == 2 and points_3d.shape[1] == 3:  # 点集
        return points_3d[:, [0, 2]]
    raise ValueError("Input points_3d must be Nx3 or a single 3-element array.")