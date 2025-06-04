# data_preprocess/core/image_annotator.py
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .geometry_processor import GeometryProcessor


class YOLOImageAnnotator:
    def __init__(self, image_wh: Tuple[int, int],
                 world_bounds_xz: Tuple[float, float, float, float],
                 ifc_class_to_yolo_id_map: Dict[str, int]):
        """
        Args:
            image_wh (Tuple[int, int]): Target image (width, height).
            world_bounds_xz (Tuple[float, float, float, float]): (x_min, z_min, x_max, z_max) in world coordinates.
            ifc_class_to_yolo_id_map (Dict[str, int]): Mapping from IFC type string to YOLO class ID.
        """
        self.image_w, self.image_h = image_wh
        self.x_min_world, self.z_min_world, self.x_max_world, self.z_max_world = world_bounds_xz

        self.world_width = self.x_max_world - self.x_min_world
        self.world_depth = self.z_max_world - self.z_min_world  # z-axis in world corresponds to depth

        if self.world_width <= 0 or self.world_depth <= 0:
            raise ValueError("World bounds must define a positive width and depth.")

        self.scale_x = self.image_w / self.world_width
        self.scale_z = self.image_h / self.world_depth  # image_h for world_depth (z-axis)

        self.ifc_class_to_yolo_id_map = ifc_class_to_yolo_id_map
        self.colors = {}  # For consistent coloring of IFC types

    def _get_color_for_ifc_type(self, ifc_type: str) -> Tuple[int, int, int]:
        if ifc_type not in self.colors:
            # Generate a random BGR color
            self.colors[ifc_type] = tuple(np.random.randint(0, 256, 3).tolist())
        return self.colors[ifc_type]

    def _world_xz_to_pixel_uv(self, points_world_xz: np.ndarray) -> np.ndarray:
        """
        Converts world XZ coordinates to image pixel UV coordinates.
        Origin of image is top-left. U is horizontal (width), V is vertical (height).
        Args:
            points_world_xz (np.ndarray): Nx2 array of (x, z) world coordinates.
        Returns:
            np.ndarray: Nx2 array of (u, v) pixel coordinates.
        """
        if points_world_xz.shape[0] == 0:
            return np.array([]).reshape(0, 2)

        # Transform x to u (image width dimension)
        pixel_u = (points_world_xz[:, 0] - self.x_min_world) * self.scale_x

        # Transform z to v (image height dimension)
        # In image coordinates, y typically increases downwards.
        # If z_min_world corresponds to v=0 (top of image):
        pixel_v = (points_world_xz[:, 1] - self.z_min_world) * self.scale_z
        # If z_max_world corresponds to v=0 (top of image) - less common for plots:
        # pixel_v = (self.z_max_world - points_world_xz[:, 1]) * self.scale_z

        return np.column_stack((pixel_u, pixel_v)).astype(int)

    def draw_component_slice(self, image: np.ndarray,
                             slice_polygon_world_xz: np.ndarray,
                             ifc_type: str,
                             draw_label: bool = True) -> None:
        """
        Draws a single component's slice polygon and its IFC type label on the image.
        Modifies the image in-place.
        """
        if slice_polygon_world_xz is None or slice_polygon_world_xz.shape[0] < 3:
            return

        pixel_polygon = self._world_xz_to_pixel_uv(slice_polygon_world_xz)
        color = self._get_color_for_ifc_type(ifc_type)

        cv2.drawContours(image, [pixel_polygon], -1, color, thickness=1, lineType=cv2.LINE_AA)
        # cv2.fillPoly(image, [pixel_polygon], color, lineType=cv2.LINE_AA) # Optional: fill

        if draw_label:
            # Put label near the centroid of the polygon
            try:
                M = cv2.moments(pixel_polygon)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Adjust label position if it's too close to image border
                    label_pos_x = max(10, min(cx, self.image_w - 50))
                    label_pos_y = max(20, min(cy, self.image_h - 10))
                    cv2.putText(image, ifc_type, (label_pos_x, label_pos_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            except Exception:  # moments can fail for degenerate polygons
                pass

    def polygon_to_yolo_format(self, polygon_pixel_coords: np.ndarray, yolo_class_id: int) -> Optional[str]:
        """
        Converts a polygon in pixel coordinates to YOLO bounding box format (class_id cx cy w h - normalized).
        For simplicity, this uses the axis-aligned bounding box of the polygon.
        Args:
            polygon_pixel_coords (np.ndarray): Nx2 array of (u,v) pixel coordinates for the polygon.
            yolo_class_id (int): The YOLO class ID for this object.
        Returns:
            Optional[str]: A string in YOLO format, or None if polygon is invalid.
        """
        if polygon_pixel_coords.shape[0] < 3:
            return None

        x_coords = polygon_pixel_coords[:, 0]
        y_coords = polygon_pixel_coords[:, 1]

        x_min, y_min = np.min(x_coords), np.min(y_coords)
        x_max, y_max = np.max(x_coords), np.max(y_coords)

        if x_min >= x_max or y_min >= y_max:  # Degenerate box
            return None

        # Clip to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.image_w - 1, x_max)
        y_max = min(self.image_h - 1, y_max)

        if x_min >= x_max or y_min >= y_max:  # Check again after clipping
            return None

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Normalize
        norm_cx = center_x / self.image_w
        norm_cy = center_y / self.image_h
        norm_w = width / self.image_w
        norm_h = height / self.image_h

        # Clamp normalized values to [0, 1] to be safe
        norm_cx = np.clip(norm_cx, 0.0, 1.0)
        norm_cy = np.clip(norm_cy, 0.0, 1.0)
        norm_w = np.clip(norm_w, 0.0, 1.0)
        norm_h = np.clip(norm_h, 0.0, 1.0)

        return f"{yolo_class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}"

    def create_annotated_image_and_labels(
            self,
            component_slices: List[Dict[str, Any]],
            # [{'ifc_type': str, 'slice_polygons_world_xz': List[np.ndarray]}, ...]
            background_image: Optional[np.ndarray] = None,
            annotation_type: str = "convex_hull"  # or "min_bounding_rect" or "original_polygon"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Creates an image with drawn component slice annotations and generates YOLO format labels.
        Args:
            component_slices: List of dictionaries, each containing 'ifc_type' and
                              'slice_polygons_world_xz' (a list of Nx2 arrays for potentially multiple contours per component slice).
            background_image: Optional pre-rendered background (e.g., from point cloud).
                              If None, a white background is used.
            annotation_type: How to derive the YOLO annotation from the slice polygon(s).
        Returns:
            Tuple[np.ndarray, List[str]]: The annotated image and a list of YOLO label strings.
        """
        if background_image is None:
            image = np.ones((self.image_h, self.image_w, 3), dtype=np.uint8) * 255  # White background
        else:
            if background_image.ndim == 2:  # Grayscale
                image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
            elif background_image.shape[:2] != (self.image_h, self.image_w):  # Resize if needed
                image = cv2.resize(background_image, (self.image_w, self.image_h))
            else:
                image = background_image.copy()

        yolo_labels = []

        for comp_slice in component_slices:
            ifc_type = comp_slice['ifc_type']
            slice_polygons_world_xz = comp_slice['slice_polygons_world_xz']  # This is List[np.ndarray]

            if ifc_type not in self.ifc_class_to_yolo_id_map:
                # print(f"Warning: IFC type '{ifc_type}' not in class map. Skipping.")
                continue

            yolo_class_id = self.ifc_class_to_yolo_id_map[ifc_type]

            for polygon_world_xz in slice_polygons_world_xz:  # Iterate over each contour of the component's slice
                if polygon_world_xz is None or polygon_world_xz.shape[0] < 3:
                    continue

                self.draw_component_slice(image, polygon_world_xz, ifc_type, draw_label=True)

                # Convert polygon to YOLO annotation
                pixel_polygon_uv = self._world_xz_to_pixel_uv(polygon_world_xz)

                # Determine the representative polygon for YOLO bbox based on annotation_type
                if annotation_type == "convex_hull":
                    representative_polygon_uv = GeometryProcessor.get_polygon_convex_hull(pixel_polygon_uv)
                elif annotation_type == "min_bounding_rect":
                    # minAreaRect returns 4 points. For YOLO, we typically use axis-aligned bbox from these.
                    obb_points = GeometryProcessor.get_polygon_min_area_rect(pixel_polygon_uv)
                    if obb_points is not None:
                        # Convert OBB points to an axis-aligned bounding box for standard YOLO
                        x_coords = obb_points[:, 0]
                        y_coords = obb_points[:, 1]
                        x_min, y_min = np.min(x_coords), np.min(y_coords)
                        x_max, y_max = np.max(x_coords), np.max(y_coords)
                        # Create a polygon from the axis-aligned rect
                        representative_polygon_uv = np.array(
                            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
                    else:
                        representative_polygon_uv = None
                elif annotation_type == "original_polygon":  # Use the raw slice polygon
                    representative_polygon_uv = pixel_polygon_uv
                else:  # Default to axis-aligned bounding box of the original polygon
                    bbox = GeometryProcessor.get_polygon_bounding_box(pixel_polygon_uv)
                    if bbox:
                        x_min, y_min, x_max, y_max = bbox
                        representative_polygon_uv = np.array(
                            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
                    else:
                        representative_polygon_uv = None

                if representative_polygon_uv is not None and representative_polygon_uv.shape[0] >= 3:
                    yolo_label_str = self.polygon_to_yolo_format(representative_polygon_uv, yolo_class_id)
                    if yolo_label_str:
                        yolo_labels.append(yolo_label_str)

        return image, yolo_labels