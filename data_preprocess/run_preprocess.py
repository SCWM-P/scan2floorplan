# data_preprocess/scripts/run_preprocess.py
import argparse
from pathlib import Path
import numpy as np
import cv2  # 用于保存图像
from tqdm import tqdm  # 用于进度条
from typing import Optional, Tuple

# 动态导入配置文件
from default_config import (
    BIMNET_ROOT_DIR, OUTPUT_YOLO_DATASET_DIR, FILE_MAPPING_JSON,
    SLICE_HEIGHTS, SLICE_THICKNESS,
    YOLO_IMAGE_WIDTH, YOLO_IMAGE_HEIGHT,
    USE_POINTCLOUD_BACKGROUND, ANNOTATION_TYPE,
    IFC_TO_YOLO_CLASS_MAPPING, PRESET_GLOBAL_XZ_BOUNDS
)

from core.data_manager import DataManager
from core.bim_parser import BIMSceneParser
from core.geometry_utils import (
    slice_component_geometries_at_height,
    calculate_convex_hull_2d,
    calculate_min_area_rect_2d,
    project_to_xz_plane
)
from core.pointcloud_processor import PointCloudProcessor
from core.image_generator import YOLOImageGenerator


def get_scene_world_bounds_xz(
        pcd_processor: PointCloudProcessor,
        bim_parser: BIMSceneParser,
        transform_matrix: np.ndarray
) -> Optional[Tuple[float, float, float, float]]:
    """
    计算单个场景在应用变换后的全局XZ边界。
    会考虑点云和所有构件的几何信息。
    """
    all_points_xz_transformed = []

    # 处理点云
    raw_pcd = pcd_processor.load_pcd_bimnet()
    if raw_pcd is not None and raw_pcd.shape[0] > 0:
        transformed_pcd = pcd_processor.apply_transform(raw_pcd[:, :3], transform_matrix)
        all_points_xz_transformed.append(transformed_pcd[:, [0, 2]])  # XZ平面

    # 处理IFC构件
    components = bim_parser.get_components()
    for component in components:
        for geometry in component.geometries:
            if not geometry.is_empty:
                # OBJ文件中的顶点通常已经是目标坐标系，但如果需要，也可以应用变换
                # transformed_vertices = geometry_utils.transform_points(np.array(geometry.vertices), transform_matrix)
                # 假设OBJ顶点已经和点云对齐了
                vertices_xz = np.array(geometry.vertices)[:, [0, 2]]  # XZ平面
                all_points_xz_transformed.append(vertices_xz)

    if not all_points_xz_transformed:
        return None

    all_points_stacked = np.vstack(all_points_xz_transformed)
    x_min, z_min = np.min(all_points_stacked, axis=0)
    x_max, z_max = np.max(all_points_stacked, axis=0)

    # 添加一些padding
    padding_x = (x_max - x_min) * 0.05
    padding_z = (z_max - z_min) * 0.05

    return x_min - padding_x, z_min - padding_z, x_max + padding_x, z_max + padding_z


def process_scene(
        scene_name: str,
        split: str,
        data_manager: DataManager,
        config,  # 传入整个config对象
        yolo_image_generator: Optional[YOLOImageGenerator] = None  # 可选，如果不是固定全局边界
):
    """处理单个BIMNet场景。"""
    print(f"\nProcessing scene: {scene_name} (split: {split})")

    pcd_path = data_manager.get_pcd_path_for_ifc_scene(scene_name, split)
    obj_component_dir = data_manager.get_obj_component_dir_for_ifc_scene(scene_name, split)
    transform_matrix_path = data_manager.get_transform_matrix_path_for_ifc_scene(scene_name, split)

    if not pcd_path or not pcd_path.exists():
        print(f"  Point cloud not found for scene {scene_name}. Skipping.")
        return
    if not obj_component_dir.exists():
        print(f"  OBJ component directory not found for scene {scene_name}. Skipping.")
        return
    if not transform_matrix_path or not transform_matrix_path.exists():
        print(f"  Transform matrix not found for scene {scene_name}. Skipping.")
        return

    transform_matrix = data_manager.load_transform_matrix(transform_matrix_path)
    if transform_matrix is None:
        print(f"  Failed to load transform matrix for scene {scene_name}. Skipping.")
        return

    pcd_processor = PointCloudProcessor(pcd_path)
    bim_parser = BIMSceneParser(obj_component_dir)  # BIMNet中OBJ通常已是对齐的

    # 如果没有预设全局边界，则为每个场景动态计算
    current_world_bounds_xz = config.PRESET_GLOBAL_XZ_BOUNDS
    if current_world_bounds_xz is None:
        print(f"  Dynamically calculating XZ bounds for scene {scene_name}...")
        current_world_bounds_xz = get_scene_world_bounds_xz(pcd_processor, bim_parser, transform_matrix)
        if current_world_bounds_xz is None:
            print(f"  Could not determine XZ bounds for scene {scene_name}. Skipping.")
            return
        print(f"  Calculated XZ bounds: {current_world_bounds_xz}")
        # 如果不是使用全局生成器，则每次都更新生成器的边界
        if yolo_image_generator:
            yolo_image_generator.update_world_bounds(current_world_bounds_xz)
        else:  # 或者创建一个新的
            yolo_image_generator = YOLOImageGenerator(
                image_wh=(config.YOLO_IMAGE_WIDTH, config.YOLO_IMAGE_HEIGHT),
                world_bounds_xz=current_world_bounds_xz
            )

    # 加载原始点云数据一次
    raw_pcd_data = pcd_processor.load_pcd_bimnet()
    transformed_pcd_data = None
    if raw_pcd_data is not None and raw_pcd_data.shape[0] > 0:
        # 假设点云需要变换到与OBJ相同的坐标系
        transformed_pcd_data = pcd_processor.apply_transform(raw_pcd_data, transform_matrix)

    # 获取所有IFC构件
    try:
        components = bim_parser.get_components()
        if not components:
            print(f"  No IFC components found in {obj_component_dir}. Skipping scene.")
            return
    except Exception as e:
        print(f"  Error parsing BIM components for scene {scene_name}: {e}. Skipping scene.")
        return

    for slice_y in config.SLICE_HEIGHTS:
        print(f"  Processing slice at Y = {slice_y:.2f}m")
        slice_y_str = f"{slice_y:.2f}".replace('.', 'p')  # for filename

        pcd_background_img = None
        if config.USE_POINTCLOUD_BACKGROUND and transformed_pcd_data is not None:
            pcd_slice = pcd_processor.slice_pcd_at_y(transformed_pcd_data, slice_y, config.SLICE_THICKNESS)
            if pcd_slice is not None and pcd_slice.shape[0] > 0:
                pcd_background_img = pcd_processor.project_pcd_to_top_down_image(
                    pcd_slice,
                    image_wh=(config.YOLO_IMAGE_WIDTH, config.YOLO_IMAGE_HEIGHT),
                    world_bounds_xz=current_world_bounds_xz,  # 使用当前场景或全局的边界
                    point_size=1,
                    use_density=False  # 可以配置为True来生成密度图
                )
            else:
                print(f"    No points in PCD slice at Y={slice_y:.2f}m for background.")

        # 收集当前切片高度下所有构件的切片多边形
        component_slice_data_for_yolo_gen = []
        for component in components:
            # 对构件的每个几何体进行切片
            slice_polygons_world_xz = slice_component_geometries_at_height(
                component.geometries,
                slice_y_level=slice_y,
                slice_thickness=config.SLICE_THICKNESS
            )

            if slice_polygons_world_xz:
                # 对于每个构件，所有切片轮廓都属于同一个IFC类型, 将所有轮廓收集起来
                valid_polygons_for_component = []
                for poly_xz in slice_polygons_world_xz:
                    if poly_xz.shape[0] < 3: continue  # 忽略少于3个点的轮廓

                    # (可选) 在这里可以根据 ANNOTATION_TYPE 选择是凸包还是最小外接矩形
                    if config.ANNOTATION_TYPE == "convex_hull":
                        annotation_polygon = calculate_convex_hull_2d(poly_xz)
                    elif config.ANNOTATION_TYPE == "min_rect":
                        annotation_polygon = calculate_min_area_rect_2d(poly_xz)  # 返回4个角点
                    else:  # 默认为原始多边形（可能不是凸的，或有洞，YOLO通常需要凸多边形或矩形）
                        annotation_polygon = poly_xz

                    if annotation_polygon is not None and annotation_polygon.shape[0] >= 3:
                        valid_polygons_for_component.append(annotation_polygon)

                if valid_polygons_for_component:
                    component_slice_data_for_yolo_gen.append({
                        'ifc_type': component.ifc_type,
                        'slice_polygons_world_xz': valid_polygons_for_component  # 传递处理后的标注多边形列表
                    })

        if not component_slice_data_for_yolo_gen:
            print(f"    No valid component slices found at Y={slice_y:.2f}m for scene {scene_name}.")
            continue

        # 生成图像和YOLO标签
        if not yolo_image_generator.valid_bounds:
            print(
                f"  YOLOImageGenerator bounds are not valid for scene {scene_name} at Y={slice_y:.2f}. Skipping image generation.")
            continue

        yolo_image, yolo_label_lines = yolo_image_generator.create_image_and_yolo_labels(
            component_slice_data_for_yolo_gen,
            config.IFC_TO_YOLO_CLASS_MAPPING,
            pcd_background_img
        )

        if yolo_image is not None and yolo_label_lines:
            # 保存
            output_img_path = data_manager.get_output_image_path(scene_name, slice_y_str, split)
            output_lbl_path = data_manager.get_output_label_path(scene_name, slice_y_str, split)

            cv2.imwrite(str(output_img_path), yolo_image)
            with open(output_lbl_path, 'w') as f:
                for line in yolo_label_lines:
                    f.write(line + "\n")
            # print(f"    Saved: {output_img_path.name}, {output_lbl_path.name}")
        else:
            print(f"    No image or labels generated for scene {scene_name} at Y={slice_y:.2f}m.")


def main(args):
    # 使用从配置文件导入的路径
    data_manager = DataManager(
        bimnet_root_dir=Path(args.bimnet_root),
        output_yolo_dir=Path(args.output_dir),
        mapping_file=Path(args.mapping_file)
    )
    # 根据配置决定YOLOImageGenerator的初始化方式
    # 如果 PRESET_GLOBAL_XZ_BOUNDS 不为 None，则创建一个全局的生成器, 否则，将在 process_scene 内部为每个场景创建或更新生成器
    global_yolo_image_generator = None
    if PRESET_GLOBAL_XZ_BOUNDS:
        print(f"Using preset global XZ bounds for image generation: {PRESET_GLOBAL_XZ_BOUNDS}")
        global_yolo_image_generator = YOLOImageGenerator(
            image_wh=(YOLO_IMAGE_WIDTH, YOLO_IMAGE_HEIGHT),
            world_bounds_xz=PRESET_GLOBAL_XZ_BOUNDS
        )
        if not global_yolo_image_generator.valid_bounds:
            print("Error: Preset global XZ bounds are invalid. Exiting.")
            return
    # 将所有配置项收集到一个对象中传递，方便管理
    class ConfigWrapper:
        pass

    current_config = ConfigWrapper()
    current_config.SLICE_HEIGHTS = SLICE_HEIGHTS
    current_config.SLICE_THICKNESS = SLICE_THICKNESS
    current_config.YOLO_IMAGE_WIDTH = YOLO_IMAGE_WIDTH
    current_config.YOLO_IMAGE_HEIGHT = YOLO_IMAGE_HEIGHT
    current_config.USE_POINTCLOUD_BACKGROUND = USE_POINTCLOUD_BACKGROUND
    current_config.ANNOTATION_TYPE = ANNOTATION_TYPE
    current_config.IFC_TO_YOLO_CLASS_MAPPING = IFC_TO_YOLO_CLASS_MAPPING
    current_config.PRESET_GLOBAL_XZ_BOUNDS = PRESET_GLOBAL_XZ_BOUNDS

    for split in ["train", "test"]:  # BIMNet通常有train和test
        ifc_scenes = data_manager.get_ifc_scenes(split)
        print(f"\nFound {len(ifc_scenes)} scenes in '{split}' split.")

        for scene_name in tqdm(ifc_scenes, desc=f"Processing {split} scenes"):
            # 如果 PRESET_GLOBAL_XZ_BOUNDS is None, process_scene 会自己创建或更新yolo_image_generator
            # 如果 PRESET_GLOBAL_XZ_BOUNDS is not None, 我们传递全局的yolo_image_generator
            process_scene(scene_name, split, data_manager, current_config, global_yolo_image_generator)
    print("\nPreprocessing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BIMNet dataset for YOLO training.")
    parser.add_argument("--bimnet_root", type=str, default=str(BIMNET_ROOT_DIR),
                        help="Root directory of the BIMNet dataset.")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_YOLO_DATASET_DIR),
                        help="Directory to save the processed YOLO dataset.")
    parser.add_argument("--mapping_file", type=str, default=str(FILE_MAPPING_JSON),
                        help="Path to the file_mapping.json.")
    cli_args = parser.parse_args()
    main(cli_args)