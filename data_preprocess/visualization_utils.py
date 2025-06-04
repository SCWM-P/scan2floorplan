# data_preprocess/utils/visualization_utils.py
import cv2
import numpy as np
from typing import List, Tuple
import functools
import trimesh
import numpy as np
from pathlib import Path
from typing import List, Callable, Any
import imageio  # 用于保存gif或多帧图像 (可选)
from default_config import (
    VISUALIZATION_OUTPUT_DIR,
    ENABLE_SLICE_VISUALIZATION,
    MAX_VISUALIZATIONS_PER_SCENE
)
_scene_visualization_count = {}

def display_image_with_polygons(image: np.ndarray, polygons: List[np.ndarray], window_name: str = "Preview"):
    """简单显示带有绘制多边形的图像。"""
    display_img = image.copy()
    for poly in polygons:
        if poly is not None and len(poly) > 0:
            cv2.polylines(display_img, [poly.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

    cv2.imshow(window_name, display_img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def reset_scene_visualization_counter(scene_name: str):
    """重置特定场景的可视化计数器"""
    _scene_visualization_count[scene_name] = 0


def _should_visualize_component(scene_name: str, component_name: str) -> bool:
    """判断当前构件是否应该被可视化"""
    if not ENABLE_SLICE_VISUALIZATION:
        return False
    count = _scene_visualization_count.get(scene_name, 0)
    if count < MAX_VISUALIZATIONS_PER_SCENE:
        # 在这里可以加入更复杂的逻辑，比如只可视化特定名称或类型的构件
        # if "WALL" in component_name.upper() or "COLUMN" in component_name.upper():
        _scene_visualization_count[scene_name] = count + 1
        return True
    return False


def save_trimesh_scene_snapshot(scene: trimesh.Scene, filepath: Path, resolution: Tuple[int, int] = (800, 600)):
    """保存Trimesh场景的快照"""
    try:
        png_data = scene.save_image(resolution=resolution, visible=True)  # visible=True确保窗口不弹出
        if png_data:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(png_data)
            print(f"INFO: Saved visualization to {filepath}")
    except Exception as e:
        print(f"Warning: Failed to save trimesh scene snapshot to {filepath}: {e}")


def visualize_slicing_process(original_func: Callable) -> Callable:
    """
    装饰器：用于可视化 trimesh 切片过程。
    它会接收原始函数的参数，并在函数执行后，如果满足条件，
    则生成并保存包含原始mesh、切片平面和切片结果（转回3D）的场景快照。
    """
    @functools.wraps(original_func)
    def wrapper(
            component_geometries: List[trimesh.Trimesh],
            slice_y_level: float,
            *args,
            component_name_for_debug: str = "unknown_component",
            scene_name_for_debug: str = "unknown_scene",
            **kwargs
    ) -> List[np.ndarray]:

        slice_polygons_xz = original_func(component_geometries, slice_y_level, *args, **kwargs)
        if _should_visualize_component(scene_name_for_debug, component_name_for_debug):
            scene_to_visualize = trimesh.Scene()
            # 1. Add original component geometries
            raw_meshes_in_scene = []  # 用于计算整体边界
            for original_mesh in component_geometries:
                if isinstance(original_mesh, trimesh.Trimesh) and not original_mesh.is_empty:
                    mesh_copy = original_mesh.copy()
                    mesh_copy.visual.face_colors = [150, 150, 150, 100]
                    scene_to_visualize.add_geometry(mesh_copy)
                    raw_meshes_in_scene.append(original_mesh)

            # 2. Create and add slicing plane visualization
            if raw_meshes_in_scene:
                # 合并所有构件几何体以获得一个总的包围盒
                combined_mesh = trimesh.util.concatenate(raw_meshes_in_scene)
                min_coords, max_coords = combined_mesh.bounds  # shape (2,3) -> [[xmin,ymin,zmin],[xmax,ymax,zmax]]

                plane_center_x = (min_coords[0] + max_coords[0]) / 2
                plane_width = (max_coords[0] - min_coords[0]) * 1.2 + 0.01  # 加一点固定值防止宽度为0
                plane_center_z = (min_coords[2] + max_coords[2]) / 2
                plane_depth = (max_coords[2] - min_coords[2]) * 1.2 + 0.01  # 加一点固定值防止深度为0

                plane_mesh = trimesh.creation.box(bounds=[
                    [plane_center_x - plane_width / 2, slice_y_level - 0.005, plane_center_z - plane_depth / 2],
                    [plane_center_x + plane_width / 2, slice_y_level + 0.005, plane_center_z + plane_depth / 2]
                ])
                plane_mesh.visual.face_colors = [0, 255, 0, 150]  # Green semi-transparent
                scene_to_visualize.add_geometry(plane_mesh)

            # 3. Add sliced results (Path2D converted to 3D Path)
            if slice_polygons_xz:
                for poly_xz in slice_polygons_xz:
                    if poly_xz is not None and len(poly_xz) >= 2:
                        vertices_3d = np.insert(poly_xz, 1, slice_y_level, axis=1)
                        num_vertices = len(vertices_3d)
                        lines_indices = np.array([[i, (i + 1) % num_vertices] for i in range(num_vertices)])
                        # Path3D可以直接从顶点和表示连接的实体创建
                        # 这里我们创建一个Line entity代表闭合多边形
                        path_entity = trimesh.path.entities.Line(points=np.arange(num_vertices))
                        path3d_slice = trimesh.path.Path3D(
                            entities=[path_entity],
                            vertices=vertices_3d
                        )
                        try:
                            # Path3D 对象本身不能直接设置颜色，但其在Scene中渲染时会用默认色
                            # 为了突出显示，我们可以将其转换为管状mesh
                            if num_vertices > 1:
                                slice_mesh_viz = path3d_slice.to_cylinders(radius=0.01)  # 创建细管
                                if isinstance(slice_mesh_viz, list):  # to_cylinders 可能返回列表
                                    for m in slice_mesh_viz:
                                        m.visual.face_colors = [255, 0, 0, 200]  # Red
                                        scene_to_visualize.add_geometry(m)
                                elif slice_mesh_viz:
                                    slice_mesh_viz.visual.face_colors = [255, 0, 0, 200]  # Red
                                    scene_to_visualize.add_geometry(slice_mesh_viz)
                        except Exception as e_path_viz:
                            print(f"Warning: Could not create mesh visualization for slice path: {e_path_viz}")
                            # 如果转换失败，就只添加原始的Path3D，它会用默认颜色渲染
                            scene_to_visualize.add_geometry(path3d_slice)

            slice_y_str = f"{slice_y_level:.2f}".replace('.', 'p')
            # 清理构件名称中的非法字符，避免文件名问题
            safe_component_name = "".join(
                c if c.isalnum() or c in (' ', '_', '-') else '_' for c in component_name_for_debug).rstrip()
            filename = f"{scene_name_for_debug}_{safe_component_name}_slice_at_Y{slice_y_str}.png"
            output_path = VISUALIZATION_OUTPUT_DIR / scene_name_for_debug / filename

            if len(scene_to_visualize.geometry) > 0:
                save_trimesh_scene_snapshot(scene_to_visualize, output_path)

        return slice_polygons_xz

    return wrapper