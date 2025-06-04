# data_preprocess/core/pointcloud_processor.py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from .geometry_processor import GeometryProcessor
import cv2  # For image operations if needed


class PointCloudProcessor:
    def __init__(self, pcd_file_path: Path):
        self.pcd_file_path = pcd_file_path
        if not self.pcd_file_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {self.pcd_file_path}")

    def load_pcd(self, y_is_up: bool = True) -> Optional[np.ndarray]:
        """
        Loads a point cloud from a .txt file (BIMNet format: x y z r g b label).
        Args:
            y_is_up (bool): If True, assumes input Z is height and converts to Y-up.
                            If False, assumes input Y is already height.
        Returns:
            Optional[np.ndarray]: Point cloud as NxK array (K>=3 for XYZ), or None if loading fails.
                                   Coordinates are ensured to be Y-up.
        """
        try:
            # BIMNet .txt format: x y z r g b label (here z is height)
            data = np.loadtxt(str(self.pcd_file_path))
            if data.ndim == 1:  # Handle case with only one point
                data = data.reshape(1, -1)

            if data.shape[1] < 3:
                print(f"Error: Point cloud file {self.pcd_file_path} has fewer than 3 coordinate columns.")
                return None

            points_xyz = data[:, :3]
            colors_rgb = data[:, 3:6] / 255.0 if data.shape[1] >= 6 else None
            # semantic_labels = data[:, 6] if data.shape[1] >= 7 else None # Not used directly for background generation

            if y_is_up:  # Original BIMNet has Z as height
                points_y_up = np.stack((points_xyz[:, 0], points_xyz[:, 2], points_xyz[:, 1]), axis=-1)
            else:  # Assume Y is already up, or input is already in desired Y-up convention
                points_y_up = points_xyz

            # For now, just return points. Colors/labels can be added if needed by image_annotator
            return points_y_up

        except Exception as e:
            print(f"Error loading point cloud {self.pcd_file_path}: {e}")
            return None

    @staticmethod
    def apply_transform(points_3d: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Applies a 4x4 transformation matrix to 3D points."""
        return GeometryProcessor.transform_points(points_3d, transform_matrix)

    @staticmethod
    def slice_pcd_at_y(
            points_3d_y_up: np.ndarray,
            slice_y_level: float,
            slice_thickness: float
    ) -> Optional[np.ndarray]:
        """
        Slices 3D points around a given Y level with a certain thickness.
        Assumes points_3d_y_up already has Y as the height axis.
        Args:
            points_3d_y_up (np.ndarray): Array of 3D points (Nx3), Y is height.
            slice_y_level (float): The Y coordinate for the center of the slice.
            slice_thickness (float): The thickness of the slice.
        Returns:
            Optional[np.ndarray]: Points within the slice (Mx3), or None.
        """
        if points_3d_y_up.shape[0] == 0:
            return None

        y_coords = points_3d_y_up[:, 1]
        half_thickness = slice_thickness / 2.0
        mask = (y_coords >= slice_y_level - half_thickness) & \
               (y_coords <= slice_y_level + half_thickness)

        sliced_points = points_3d_y_up[mask]
        return sliced_points if sliced_points.shape[0] > 0 else None

    @staticmethod
    def project_pcd_to_image_plane(
            points_3d_slice_y_up: np.ndarray,  # Points already sliced and Y-up
            image_wh: Tuple[int, int],  # Target image (width, height)
            world_bounds_xz: Tuple[float, float, float, float],
            # (x_min, z_min, x_max, z_max) of the world area to render
            density_radius_world: float = 0.05,  # Radius in world units to consider for density
            min_points_for_pixel: int = 1  # Minimum points in radius to color a pixel
    ) -> Optional[np.ndarray]:
        """
        Projects sliced 3D points (Y-up) onto the XZ plane to generate a 2D density image.
        Args:
            points_3d_slice_y_up (np.ndarray): Sliced 3D points (Nx3), Y is height. (X, Y_height, Z)
            image_wh (Tuple[int, int]): Target image (width, height).
            world_bounds_xz (Tuple[float, float, float, float]): (x_min, z_min, x_max, z_max) in world coordinates.
            density_radius_world (float): Radius in world units to calculate point density.
            min_points_for_pixel (int): Minimum number of points within density_radius_world to color a pixel.
        Returns:
            Optional[np.ndarray]: A 2D grayscale density image (height, width), or None.
        """
        if points_3d_slice_y_up is None or points_3d_slice_y_up.shape[0] == 0:
            return np.ones((image_wh[1], image_wh[0]), dtype=np.uint8) * 255  # White background

        points_xz = points_3d_slice_y_up[:, [0, 2]]  # Extract (x, original_z) for projection

        img_w, img_h = image_wh
        x_min_world, z_min_world, x_max_world, z_max_world = world_bounds_xz

        world_width = x_max_world - x_min_world
        world_height = z_max_world - z_min_world

        if world_width <= 0 or world_height <= 0:
            print("Warning: Invalid world_bounds_xz, width or height is zero or negative.")
            return np.ones((img_h, img_w), dtype=np.uint8) * 255

        scale_x = img_w / world_width
        scale_z = img_h / world_height  # Note: image height corresponds to Z world range

        # Transform world XZ to image pixel coordinates (u, v)
        # u = (x_world - x_min_world) * scale_x
        # v = (z_world - z_min_world) * scale_z  --- for typical image origin top-left
        # However, if image y-axis increases downwards:
        # v = (z_max_world - z_world) * scale_z for inverted z-axis or
        # v = (z_world - z_min_world) * scale_z (standard plot) and then flipud if needed.
        # Let's assume standard plot: (0,0) at (x_min_world, z_min_world)

        pixel_coords_x = ((points_xz[:, 0] - x_min_world) * scale_x).astype(int)
        pixel_coords_z_img_y = ((points_xz[:, 1] - z_min_world) * scale_z).astype(int)  # z_world maps to image y

        # Create density image (grayscale, 0=black, 255=white)
        # Initialize with white background
        density_image = np.ones((img_h, img_w), dtype=np.uint8) * 255

        # Filter points outside the image bounds
        valid_mask = (pixel_coords_x >= 0) & (pixel_coords_x < img_w) & \
                     (pixel_coords_z_img_y >= 0) & (pixel_coords_z_img_y < img_h)

        pixel_coords_x_valid = pixel_coords_x[valid_mask]
        pixel_coords_z_img_y_valid = pixel_coords_z_img_y[valid_mask]

        # Simple projection: mark pixels that have any point
        # A more advanced approach would be density/heatmap
        if min_points_for_pixel <= 1:  # Direct projection if min_points is 1 or less
            density_image[pixel_coords_z_img_y_valid, pixel_coords_x_valid] = 0  # Black for points
        else:  # Density based coloring
            # This is a simplified density. For a proper heatmap, use kernel density estimation
            # or count points in a radius around each pixel center.
            # The current `density_radius_world` isn't used in this simplified version.
            # Let's implement a basic density count per pixel grid cell for now.

            # Create an accumulator array
            point_counts = np.zeros((img_h, img_w), dtype=int)
            np.add.at(point_counts, (pixel_coords_z_img_y_valid, pixel_coords_x_valid), 1)

            # Color pixels where point count meets threshold
            density_mask = point_counts >= min_points_for_pixel
            density_image[density_mask] = 0  # Black for dense areas

        return density_image