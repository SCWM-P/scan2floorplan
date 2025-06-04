# data_preprocess/core/geometry_processor.py
import numpy as np
import trimesh
from typing import Optional, List, Tuple, Union
from shapely.geometry import Polygon as ShapelyPolygon
import cv2


class GeometryProcessor:
    @staticmethod
    def transform_points(points_3d: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Applies a 4x4 transformation matrix to a list of 3D points.
        Args:
            points_3d (np.ndarray): Array of 3D points, shape (N, 3).
            transform_matrix (np.ndarray): 4x4 transformation matrix.
        Returns:
            np.ndarray: Transformed 3D points, shape (N, 3).
        """
        if points_3d.shape[0] == 0:
            return points_3d
        homogeneous_points = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # (N, 4)
        transformed_homogeneous = (transform_matrix @ homogeneous_points.T).T  # (N, 4)
        # Normalize by the 4th coordinate (w) if it's not 1
        w = transformed_homogeneous[:, 3]
        # Avoid division by zero if w is zero for some points
        safe_w = np.where(w == 0, 1, w)
        transformed_points_3d = transformed_homogeneous[:, :3] / safe_w[:, np.newaxis]

        return transformed_points_3d

    @staticmethod
    def project_to_xz_plane(points_3d: np.ndarray) -> np.ndarray:
        """
        Projects 3D points onto the XZ plane (Y becomes the new Z for 2D).
        This assumes Y is UP in the 3D coordinate system.
        Args:
            points_3d (np.ndarray): Array of 3D points, shape (N, 3) -> (x, y, z).
        Returns:
            np.ndarray: Array of 2D points on the XZ plane, shape (N, 2) -> (x, z_original).
        """
        if points_3d.shape[0] == 0:
            return np.array([]).reshape(0, 2)
        return points_3d[:, [0, 2]]  # Select X and Z coordinates

    @staticmethod
    def slice_component_at_y(
            vertices: np.ndarray,
            faces: np.ndarray,
            slice_y_level: float,
            slice_thickness: float  # slice_thickness 目前在这个简化版中未被严格使用
    ) -> Optional[List[np.ndarray]]:
        """
        Slices a 3D mesh component at a given Y level and returns the 2D XZ contour(s)
        of the intersection. Assumes Y is the height axis.
        """
        if vertices is None or faces is None or vertices.shape[0] < 3 or faces.shape[0] < 1:
            return None
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        try:
            path_object_or_list = mesh.section(
                plane_origin=[0, slice_y_level, 0],
                plane_normal=[0, 1, 0]
            )
        except Exception as e:
            print(f"Trimesh section failed: {e}")
            return None
        if path_object_or_list is None:
            return None
        # Ensure we have a list of path objects
        if isinstance(path_object_or_list, (trimesh.path.Path2D, trimesh.path.Path3D)):
            path_objects = [path_object_or_list]
        elif isinstance(path_object_or_list, list):
            path_objects = path_object_or_list
        else:
            print(f"Unexpected return type from mesh.section: {type(path_object_or_list)}")
            return None

        projected_polygons = []
        for path_obj in path_objects:
            if path_obj is None or not hasattr(path_obj, 'vertices') or path_obj.vertices.shape[0] < 3:
                continue
            # Path3D.vertices gives (N, 3) array of vertices in the path
            # These vertices lie on the slicing plane, so their Y coordinate is slice_y_level
            vertices_3d_on_plane = path_obj.vertices
            # Trimesh path objects can consist of multiple disconnected line segments (entities).
            # We need to process each closed loop (discrete path) separately.
            # path_obj.discrete should give a list of numpy arrays, each being a separate path.
            if hasattr(path_obj, 'discrete') and path_obj.discrete:
                for discrete_path_vertices_3d in path_obj.discrete:
                    if discrete_path_vertices_3d.shape[0] >= 3:
                        # Project to XZ plane
                        polygon_2d = discrete_path_vertices_3d[:, [0, 2]]  # (x, z)
                        # Ensure the polygon is closed for certain operations later if needed
                        # (though for convex hull or bbox, it's not strictly necessary if points define the shape)
                        # if not np.array_equal(polygon_2d[0], polygon_2d[-1]):
                        #     polygon_2d = np.vstack([polygon_2d, polygon_2d[0]])
                        projected_polygons.append(polygon_2d)
            elif path_obj.vertices.shape[0] >= 3:  # Fallback if .discrete is not available or empty
                polygon_2d = vertices_3d_on_plane[:, [0, 2]]
                projected_polygons.append(polygon_2d)

        return projected_polygons if projected_polygons else None

    @staticmethod
    def get_polygon_bounding_box(polygon_2d: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates the (axis-aligned) bounding box of a 2D polygon.
        Args:
            polygon_2d (np.ndarray): Array of 2D points (Nx2) defining the polygon.
        Returns:
            Optional[Tuple[float, float, float, float]]: (x_min, y_min, x_max, y_max) or None.
        """
        if polygon_2d.shape[0] < 3:
            return None
        x_min, y_min = np.min(polygon_2d, axis=0)
        x_max, y_max = np.max(polygon_2d, axis=0)
        return x_min, y_min, x_max, y_max  # Here y_min/y_max correspond to z_min/z_max in XZ plane

    @staticmethod
    def get_polygon_convex_hull(polygon_2d: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculates the convex hull of a 2D polygon.
        Args:
            polygon_2d (np.ndarray): Array of 2D points (Nx2) defining the polygon.
        Returns:
            Optional[np.ndarray]: Array of 2D points (Mx2) defining the convex hull, or None.
        """
        if polygon_2d.shape[0] < 3:
            return None
        try:
            # OpenCV expects points as (N, 1, 2) and dtype int32 for some operations,
            # but convexHull takes float32. Ensure correct format.
            hull_indices = cv2.convexHull(polygon_2d.astype(np.float32), returnPoints=False)
            if hull_indices is not None and len(hull_indices) > 0:
                return polygon_2d[hull_indices.flatten()]
            return None
        except Exception as e:
            # print(f"Error calculating convex hull: {e}")
            return None

    @staticmethod
    def get_polygon_min_area_rect(polygon_2d: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculates the minimum area oriented bounding rectangle for a 2D polygon.
        Args:
            polygon_2d (np.ndarray): Array of 2D points (Nx2) defining the polygon.
        Returns:
            Optional[np.ndarray]: Array of 4 2D points (4x2) defining the OBB, or None.
        """
        if polygon_2d.shape[0] < 3:
            return None
        try:
            # minAreaRect needs contour points in int32 format for some OpenCV versions,
            # but typically works with float32 as well.
            # It's safer to use float32 if your coordinates are not integers.
            rect = cv2.minAreaRect(polygon_2d.astype(np.float32))
            box_points = cv2.boxPoints(rect)  # ((center_x, center_y), (width, height), angle_of_rotation)
            return box_points  # Returns 4 corner points of the rectangle
        except Exception as e:
            # print(f"Error calculating min area rectangle: {e}")
            return None