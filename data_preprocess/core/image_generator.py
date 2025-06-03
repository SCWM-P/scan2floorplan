# data_preprocess/core/image_generator.py
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict


class YOLOImageGenerator:
    """
    生成带有标注的2D图像，并将标注转换为YOLO格式。
    """

    def __init__(self, image_wh: Tuple[int, int], world_bounds_xz: Optional[Tuple[float, float, float, float]] = None):
        """
        Args:
            image_wh (Tuple[int, int]): 输出图像的 (宽度, 高度)。
            world_bounds_xz (Optional[Tuple[float, float, float, float]]):
                用于将世界坐标（XZ平面）映射到图像像素坐标的边界 (x_min, z_min, x_max, z_max)。
                如果为None，则需要在 `create_image_and_yolo_labels` 中动态计算或传入。
        """
        self.img_w, self.img_h = image_wh
        self.world_bounds_xz = world_bounds_xz

        if self.world_bounds_xz:
            self._calculate_transform_parameters()
        else:
            self.scale = 1.0
            self.offset_x_img = 0.0
            self.offset_z_img = 0.0  # 对应图像的y偏移
            self.valid_bounds = False

    def _calculate_transform_parameters(self):
        """根据世界边界和图像尺寸计算缩放和平移参数。"""
        if not self.world_bounds_xz:
            self.valid_bounds = False
            return

        x_min_world, z_min_world, x_max_world, z_max_world = self.world_bounds_xz

        world_width = x_max_world - x_min_world
        world_depth = z_max_world - z_min_world  # Z轴对应图像的Y轴（深度）

        if world_width <= 1e-6 or world_depth <= 1e-6:  # 避免除零或边界无效
            print(f"Warning: Invalid world bounds for image generation: {self.world_bounds_xz}")
            self.valid_bounds = False
            return

        scale_x = self.img_w / world_width
        scale_z = self.img_h / world_depth  # Z轴对应图像Y轴

        self.scale = min(scale_x, scale_z)  # 统一缩放比例，保持长宽比

        # 投影后新内容的尺寸
        self.content_w_img = int(world_width * self.scale)
        self.content_h_img = int(world_depth * self.scale)  # Z轴对应图像Y轴

        # 计算偏移量，使得内容在图像中居中
        self.offset_x_img = (self.img_w - self.content_w_img) / 2
        self.offset_z_img = (self.img_h - self.content_h_img) / 2  # Z轴对应图像Y轴

        # 变换参数: x_pixel = (x_world - x_min_world) * scale + offset_x_img
        #            y_pixel = (z_world - z_min_world) * scale + offset_z_img  (注意这里是z_world到y_pixel)
        self.x_min_world = x_min_world
        self.z_min_world = z_min_world
        self.valid_bounds = True

    def update_world_bounds(self, world_bounds_xz: Tuple[float, float, float, float]):
        """如果初始未提供边界，或者需要为每个场景更新边界，则调用此方法。"""
        self.world_bounds_xz = world_bounds_xz
        self._calculate_transform_parameters()

    def _world_xz_to_pixel(self, points_world_xz: np.ndarray) -> np.ndarray:
        """将世界XZ坐标转换为图像像素坐标 (x,y)。"""
        if not self.valid_bounds:
            raise RuntimeError("World bounds not set or invalid for coordinate transformation.")

        if points_world_xz.ndim == 1:  # 单个点
            points_world_xz = points_world_xz.reshape(1, -1)

        pixel_coords = np.zeros_like(points_world_xz, dtype=np.int32)

        # X坐标 (图像横轴)
        pixel_coords[:, 0] = ((points_world_xz[:, 0] - self.x_min_world) * self.scale + self.offset_x_img).astype(int)
        # Z坐标 (世界深度轴) 映射到 图像Y轴 (图像纵轴，向下为正)
        pixel_coords[:, 1] = ((points_world_xz[:, 1] - self.z_min_world) * self.scale + self.offset_z_img).astype(int)

        return pixel_coords

    def _generate_yolo_label_line(self, class_id: int, polygon_pixels: np.ndarray) -> Optional[str]:
        """
        根据像素坐标的多边形计算YOLO格式的标注。
        Args:
            class_id (int): 物体的类别ID。
            polygon_pixels (np.ndarray): 物体标注多边形的像素坐标 (Mx2)。
        Returns:
            Optional[str]: YOLO格式的标注行 "class_id cx cy w h" (均为归一化值)，如果无效则返回None。
        """
        if polygon_pixels is None or polygon_pixels.shape[0] < 3:
            return None

        # 使用最小外接矩形作为YOLO的bounding box
        x_coords = polygon_pixels[:, 0]
        y_coords = polygon_pixels[:, 1]

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        if x_max <= x_min or y_max <= y_min:  # 检查有效性
            return None

        # 中心点坐标
        center_x_pixel = (x_min + x_max) / 2
        center_y_pixel = (y_min + y_max) / 2

        # 宽度和高度
        width_pixel = x_max - x_min
        height_pixel = y_max - y_min

        # 归一化
        center_x_norm = center_x_pixel / self.img_w
        center_y_norm = center_y_pixel / self.img_h
        width_norm = width_pixel / self.img_w
        height_norm = height_pixel / self.img_h

        # 确保归一化值在 [0, 1] 范围内
        center_x_norm = np.clip(center_x_norm, 0.0, 1.0)
        center_y_norm = np.clip(center_y_norm, 0.0, 1.0)
        width_norm = np.clip(width_norm, 0.0, 1.0)
        height_norm = np.clip(height_norm, 0.0, 1.0)

        if width_norm <= 1e-3 or height_norm <= 1e-3:  # 忽略面积过小的框
            return None

        return f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

    def create_image_and_yolo_labels(
            self,
            component_slices: List[Dict[str, any]],
            # [{'ifc_type': str, 'slice_polygons_world_xz': [np.ndarray(Mx2), ...]}, ...]
            ifc_class_to_yolo_id_map: Dict[str, int],
            pcd_background_image: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        生成一张带有标注的图像和对应的YOLO格式标签。

        Args:
            component_slices (List[Dict]]):
                每个构件的切片信息列表。
                每个dict包含:
                    'ifc_type': IFC类型字符串。
                    'slice_polygons_world_xz': 此构件在此切片高度产生的所有2D轮廓点集 (世界XZ坐标)的列表。
                                              一个构件可能产生多个不相连的轮廓。
            ifc_class_to_yolo_id_map (Dict[str, int]): IFC类型到YOLO类别ID的映射。
            pcd_background_image (Optional[np.ndarray]): 可选的点云切片投影图作为背景 (BGR格式)。

        Returns:
            Tuple[Optional[np.ndarray], List[str]]:
                - 生成的图像 (H, W, 3) BGR格式，如果无法生成则为None。
                - YOLO格式的标签行列表。
        """
        if not self.valid_bounds and not pcd_background_image:  # 如果没有预设边界，且没有背景图来确定边界
            # 尝试从构件切片动态计算边界
            all_polygons_world_xz = []
            for comp_slice in component_slices:
                all_polygons_world_xz.extend(comp_slice.get('slice_polygons_world_xz', []))

            if not all_polygons_world_xz:
                return None, []  # 没有构件可绘制

            all_points = np.vstack(all_polygons_world_xz)
            x_min, z_min = np.min(all_points, axis=0)
            x_max, z_max = np.max(all_points, axis=0)
            padding = max(x_max - x_min, z_max - z_min) * 0.1  # 10% padding
            self.update_world_bounds((x_min - padding, z_min - padding, x_max + padding, z_max + padding))
            if not self.valid_bounds:  # 如果计算后边界仍然无效
                return None, []

        if pcd_background_image is not None:
            # 确保背景图像尺寸与YOLO图像尺寸一致
            if pcd_background_image.shape[1] != self.img_w or pcd_background_image.shape[0] != self.img_h:
                current_image = cv2.resize(pcd_background_image, (self.img_w, self.img_h))
            else:
                current_image = pcd_background_image.copy()
        else:
            current_image = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255  # 白色背景

        yolo_labels = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]  # 预定义一些颜色

        for i, comp_data in enumerate(component_slices):
            ifc_type = comp_data['ifc_type']
            slice_polygons_world_xz = comp_data.get('slice_polygons_world_xz', [])

            if ifc_type not in ifc_class_to_yolo_id_map:
                continue  # 跳过不在映射表中的IFC类型

            yolo_class_id = ifc_class_to_yolo_id_map[ifc_type]
            color = colors[yolo_class_id % len(colors)]  # BGR

            for polygon_world_xz in slice_polygons_world_xz:
                if polygon_world_xz.shape[0] < 3:  # 轮廓点太少
                    continue

                # 将世界坐标的轮廓点转换为像素坐标
                polygon_pixels = self._world_xz_to_pixel(polygon_world_xz)

                # 绘制填充的多边形 (标注区域)
                cv2.drawContours(current_image, [polygon_pixels], -1, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
                # 绘制轮廓线 (可选，为了更清晰)
                cv2.drawContours(current_image, [polygon_pixels], -1, (50, 50, 50), thickness=1, lineType=cv2.LINE_AA)

                # 生成YOLO标签
                yolo_label_line = self._generate_yolo_label_line(yolo_class_id, polygon_pixels)
                if yolo_label_line:
                    yolo_labels.append(yolo_label_line)

        return current_image, yolo_labels