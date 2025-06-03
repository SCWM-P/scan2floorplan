# data_preprocess/core/pointcloud_processor.py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import cv2  # For image generation from points


class PointCloudProcessor:
    """处理点云数据的加载、切片和2D投影。"""

    def __init__(self, pcd_file_path: Path):
        self.pcd_file_path = pcd_file_path
        if not self.pcd_file_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {self.pcd_file_path}")

    def load_pcd_bimnet(self) -> Optional[np.ndarray]:
        """
        加载BIMNet格式的点云 (.txt, x y z r g b label)。
        返回 Nx6 的 numpy 数组 (x, y, z, r, g, b)。标签被丢弃。
        **重要：假设BIMNet点云文件中的坐标已经是Y轴朝上，如果不是，需要在此处或加载后进行转换。**
        根据BIMNet data_organization.md, 点云已经是 x y z r g b label, Y是高度。
        所以，如果 mat_pc2obj 是将原始扫描（可能Z朝上）转到这个Y朝上的OBJ坐标系，那么这里直接用就好。
        """
        try:
            # BIMNet点云是 x y z r g b label (7列)
            # 我们通常只需要 x, y, z, r, g, b (6列)
            data = np.loadtxt(str(self.pcd_file_path), dtype=np.float32, usecols=(0, 1, 2, 3, 4, 5))
            return data
        except Exception as e:
            print(f"Error loading BIMNet point cloud {self.pcd_file_path}: {e}")
            return None

    def apply_transform(self, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """应用4x4变换矩阵到点云 (Nx3 or Nx6)"""
        if points.shape[1] < 3:
            raise ValueError("Points must have at least 3 columns (x,y,z)")

        coords = points[:, :3]
        if coords.shape[1] == 3:
            coords_h = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        else:  # Should not happen if points are Nx3
            coords_h = coords

        transformed_coords_h = coords_h @ matrix.T
        transformed_coords = transformed_coords_h[:, :3] / transformed_coords_h[:, 3, np.newaxis]

        # 如果原始点云包含颜色等信息，则保留它们
        if points.shape[1] > 3:
            return np.hstack((transformed_coords, points[:, 3:]))
        else:
            return transformed_coords

    def slice_pcd_at_y(self, points_xyz_rgb: np.ndarray, slice_y_level: float, slice_thickness: float) -> Optional[
        np.ndarray]:
        """
        对点云在指定Y高度进行切片。
        Args:
            points_xyz_rgb (np.ndarray): Nx6 点云数据 (x, y, z, r, g, b)。Y是高度轴。
            slice_y_level (float): 切片中心Y坐标。
            slice_thickness (float): 切片厚度。
        Returns:
            Optional[np.ndarray]: 切片后的点云 (x, y, z, r, g, b)，如果无点则返回None。
        """
        if points_xyz_rgb is None or points_xyz_rgb.shape[0] == 0:
            return None

        y_coords = points_xyz_rgb[:, 1]  # Y是高度
        mask = (y_coords >= (slice_y_level - slice_thickness / 2)) & \
               (y_coords <= (slice_y_level + slice_thickness / 2))

        sliced_points = points_xyz_rgb[mask]
        return sliced_points if sliced_points.shape[0] > 0 else None

    def project_pcd_to_top_down_image(
            self,
            points_xyz_rgb: np.ndarray,  # 应该是切片后的点云，或者需要投影的3D点
            image_wh: Tuple[int, int],
            world_bounds_xz: Tuple[float, float, float, float],  # (x_min, z_min, x_max, z_max)
            point_size: int = 1,
            use_density: bool = False,  # True则生成密度图，False则用点云颜色
            default_color: Tuple[int, int, int] = (0, 0, 0)  # BGR
    ) -> np.ndarray:
        """
        将点云（通常是切片后的）的XZ坐标投影到2D图像上。

        Args:
            points_xyz_rgb (np.ndarray): Nx6 点云数据 (x, y, z, r, g, b)。Y是高度轴。
            image_wh (Tuple[int, int]): 输出图像的 (宽度, 高度)。
            world_bounds_xz (Tuple[float, float, float, float]): 点云在世界坐标系中的XZ平面边界 (x_min, z_min, x_max, z_max)。
                                                             用于计算缩放和平移。
            point_size (int): 绘制点的半径。
            use_density (bool): 是否生成密度图而不是彩色点图。
            default_color (Tuple[int,int,int]): 默认绘制点的颜色 (BGR)，如果use_density=False且点云无颜色时使用。

        Returns:
            np.ndarray: 生成的2D图像 (H, W, 3) in BGR format。
        """
        if points_xyz_rgb is None or points_xyz_rgb.shape[0] == 0:
            return np.ones((image_wh[1], image_wh[0], 3), dtype=np.uint8) * 255  # White background

        img_w, img_h = image_wh
        x_coords_world = points_xyz_rgb[:, 0]
        z_coords_world = points_xyz_rgb[:, 2]

        if points_xyz_rgb.shape[1] >= 6:  # 包含颜色信息
            colors_rgb = points_xyz_rgb[:, 3:6]  # r,g,b
        else:  # 没有颜色信息
            colors_rgb = np.array([default_color[::-1]] * len(points_xyz_rgb))  # BGR to RGB

        x_min_world, z_min_world, x_max_world, z_max_world = world_bounds_xz

        world_width = x_max_world - x_min_world
        world_height = z_max_world - z_min_world

        if world_width <= 0 or world_height <= 0:  # 避免除零
            return np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

        scale_x = img_w / world_width
        scale_z = img_h / world_height
        # 使用统一的比例因子，保持长宽比，取较小的那个，然后居中
        scale = min(scale_x, scale_z)

        # 计算图像像素坐标
        # (coord_world - min_world) * scale + offset_to_center
        # 这里的投影方向：世界Z轴对应图像Y轴，世界X轴对应图像X轴
        # 图像原点(0,0)在左上角
        # 图像Y轴向下为正，图像X轴向右为正
        # 世界Z轴正方向通常是“深入”屏幕，对应图像Y轴向下。世界X轴正方向是向右，对应图像X轴向右。

        # 目标：将 (x_min_world, z_min_world) 映射到图像的某个位置 (如左上角或考虑边距)
        #       将 (x_max_world, z_max_world) 映射到图像的某个位置 (如右下角或考虑边距)

        # 新的图像画布尺寸，基于scale和world_width/height
        new_img_w = int(world_width * scale)
        new_img_h = int(world_height * scale)

        # 偏移量，使得投影居中
        offset_x = (img_w - new_img_w) / 2
        offset_z = (img_h - new_img_h) / 2  # 对应图像y轴

        # 转换到像素坐标 (相对于新图像区域的左上角)
        x_pixel = ((x_coords_world - x_min_world) * scale + offset_x).astype(int)
        # Z轴通常是深度，映射到图像的Y轴。Z值越大，图像Y值越大（向下）
        z_pixel = ((z_coords_world - z_min_world) * scale + offset_z).astype(int)

        if use_density:
            density_map = np.zeros((img_h, img_w), dtype=np.float32)
            valid_mask = (x_pixel >= 0) & (x_pixel < img_w) & (z_pixel >= 0) & (z_pixel < img_h)
            valid_x_pixel = x_pixel[valid_mask]
            valid_z_pixel = z_pixel[valid_mask]

            if len(valid_x_pixel) > 0:
                # 使用histogram2d更高效地创建密度图
                hist, _, _ = np.histogram2d(valid_z_pixel, valid_x_pixel, bins=[img_h, img_w],
                                            range=[[0, img_h], [0, img_w]])
                density_map = hist.astype(np.float32)

            if np.max(density_map) > 0:
                density_map = (density_map / np.max(density_map) * 255).astype(np.uint8)

            # 将灰度密度图转换为BGR图像
            image = cv2.cvtColor(density_map, cv2.COLOR_GRAY2BGR)
            # 可以选择应用伪彩色
            # image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        else:  # 使用点云颜色
            image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255  # White background

            for i in range(len(x_pixel)):
                px, pz = x_pixel[i], z_pixel[i]
                if 0 <= px < img_w and 0 <= pz < img_h:
                    # 点云颜色是R,G,B, OpenCV是B,G,R
                    color_bgr = (int(colors_rgb[i, 2]), int(colors_rgb[i, 1]), int(colors_rgb[i, 0]))
                    cv2.circle(image, (px, pz), radius=point_size, color=color_bgr, thickness=-1)

        return image