# data_preprocess/run_preprocess.py
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
# 导入我们之前定义的core模块和配置
from core.data_manager import DataManager
from core.bim_parser import BIMSceneParser
from core.geometry_processor import GeometryProcessor
from core.pointcloud_processor import PointCloudProcessor
from core.image_annotator import YOLOImageAnnotator
import configs  # 导入configs.py


def main(args):
    print("Starting Scan2Floorplan Data Preprocessing...")
    print(f"BIMNet Root: {configs.BIMNET_ROOT_DIR}")
    print(f"Output YOLO Dataset Dir: {configs.OUTPUT_YOLO_DIR}")
    print(f"Preprocessing Config: {configs.PREPROCESSING_CONFIG}")

    data_manager = DataManager(
        bimnet_root_dir=configs.BIMNET_ROOT_DIR,
        output_yolo_dir=configs.OUTPUT_YOLO_DIR,
        mapping_file=configs.FILE_MAPPING_JSON
    )

    ifc_class_to_yolo_id_map = configs.PREPROCESSING_CONFIG["ifc_to_yolo_class_map"]
    target_ifc_entities = configs.PREPROCESSING_CONFIG["target_ifc_entities"]
    slice_y_levels = configs.PREPROCESSING_CONFIG["slice_y_levels"]
    slice_thickness = configs.PREPROCESSING_CONFIG["slice_thickness"]
    image_wh = configs.PREPROCESSING_CONFIG["yolo_image_size_wh"]
    use_pc_background = configs.PREPROCESSING_CONFIG["use_pointcloud_background"]
    annotation_type = configs.PREPROCESSING_CONFIG["annotation_type"]

    # 确保Y轴向上
    y_is_up_in_ifc = True  # 通常IFC是Y轴向上
    y_is_up_in_pcd_bimnet_raw = False  # BIMNet原始点云是Z轴向上

    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} split...")
        scene_names = data_manager.get_scene_names(split)
        if not scene_names:
            print(f"No scenes found for {split} split. Skipping.")
            continue

        for scene_name_ifc in tqdm(scene_names, desc=f"Scenes in {split}"):
            scene_name_base = scene_name_ifc.replace(".ifc", "")
            print(f"  Processing scene: {scene_name_ifc}")

            try:
                # 1. 加载IFC场景并解析构件
                ifc_path = data_manager.get_ifc_path(scene_name_ifc, split)
                bim_parser = BIMSceneParser(ifc_path, target_entities=target_ifc_entities)
                ifc_components = bim_parser.get_components()
                if not ifc_components:
                    print(f"    No target components found in {scene_name_ifc}. Skipping scene.")
                    continue

                # (可选) 获取并应用从点云到IFC模型的变换矩阵
                # BIMNet的mat_pc2obj是将点云对齐到OBJ/IFC。
                # 如果我们以IFC为基准，点云需要被变换。
                # 如果我们以点云为基准，IFC需要被反向变换（更复杂）。
                # 此处假设IFC坐标是我们要使用的“世界坐标”，点云需要变换。
                transform_matrix_pcd_to_ifc = None
                transform_matrix_path = data_manager.get_transform_matrix_for_ifc_scene(scene_name_ifc, split)
                if transform_matrix_path and transform_matrix_path.exists():
                    transform_matrix_pcd_to_ifc = np.loadtxt(str(transform_matrix_path))
                else:
                    if use_pc_background:  # 只有在需要点云背景时，变换矩阵才是必须的
                        print(
                            f"    Warning: Transform matrix not found for {scene_name_ifc}. Cannot generate point cloud background.")

                # 2. (可选) 加载并处理点云作为背景
                pcd_background_img_prototype = None
                if use_pc_background and transform_matrix_pcd_to_ifc is not None:
                    pcd_path = data_manager.get_pcd_path_for_ifc_scene(scene_name_ifc, split)
                    if pcd_path:
                        pc_processor = PointCloudProcessor(pcd_path)
                        # BIMNet点云是Z轴向上，加载时转换为Y轴向上
                        raw_pcd_points_y_up = pc_processor.load_pcd(y_is_up=True)
                        if raw_pcd_points_y_up is not None:
                            # 应用变换矩阵将点云对齐到IFC空间
                            transformed_pcd_points = pc_processor.apply_transform(raw_pcd_points_y_up,
                                                                                  transform_matrix_pcd_to_ifc)

                            # 计算该场景所有IFC构件顶点变换到Y-up后的XZ边界，用于点云背景图的渲染范围
                            # (或者使用一个固定的、足够大的全局边界)
                            all_component_vertices_xz = []
                            for comp in ifc_components:
                                if comp.vertices is not None:
                                    # IFC构件顶点已经是Y-up（基于ifcopenshell的USE_WORLD_COORDS）
                                    all_component_vertices_xz.append(
                                        GeometryProcessor.project_to_xz_plane(comp.vertices))

                            if not all_component_vertices_xz:
                                print(
                                    f"    Warning: No component vertices to determine bounds for {scene_name_ifc}. Skipping point cloud background.")
                                world_bounds_xz_scene = None  # Default if no component bounds
                            else:
                                combined_vertices_xz = np.vstack(all_component_vertices_xz)
                                x_coords, z_coords = combined_vertices_xz[:, 0], combined_vertices_xz[:, 1]
                                x_min, z_min = np.min(x_coords), np.min(z_coords)
                                x_max, z_max = np.max(x_coords), np.max(z_coords)
                                # Add some padding to bounds
                                padding = max(x_max - x_min, z_max - z_min) * 0.1
                                world_bounds_xz_scene = (x_min - padding, z_min - padding, x_max + padding,
                                                         z_max + padding)

                            # 为每个切片高度生成背景图 (如果每个切片高度都需要单独的点云背景)
                            # 或者只在某个代表性高度生成一次背景图
                            # 为简化，这里我们为每个切片高度都生成一次（尽管对于水平切片，XZ范围内的点云背景可能相似）
                            # pcd_slice_for_bg = pc_processor.slice_pcd_at_y(transformed_pcd_points, slice_y_levels[0], slice_thickness * 5) # 取较厚点云层做背景
                            # if pcd_slice_for_bg is not None and world_bounds_xz_scene is not None:
                            #     pcd_background_img_prototype = pc_processor.project_pcd_to_image_plane(
                            #         pcd_slice_for_bg,
                            #         image_wh,
                            #         world_bounds_xz_scene,
                            #         density_radius_world=configs.PREPROCESSING_CONFIG["pointcloud_density_radius"],
                            #         min_points_for_pixel=configs.PREPROCESSING_CONFIG["pointcloud_min_points_for_pixel"]
                            #     )
                            # else:
                            #     print(f"    No points in PCD slice for background of {scene_name_ifc}")
                            # 使用原始点云进行投影，只显示特定高度范围内的点作为“背景”
                            if transformed_pcd_points is not None and world_bounds_xz_scene is not None:
                                # 过滤点云到大致的楼层高度范围，避免渲染不相关的点
                                y_min_scene_level = min(slice_y_levels) - 1.0  # e.g. 0.5m below lowest slice
                                y_max_scene_level = max(slice_y_levels) + 1.0  # e.g. 0.5m above highest slice
                                floor_points_mask = (transformed_pcd_points[:, 1] >= y_min_scene_level) & \
                                                    (transformed_pcd_points[:, 1] <= y_max_scene_level)
                                pcd_for_bg_projection = transformed_pcd_points[floor_points_mask]

                                if pcd_for_bg_projection.shape[0] > 0:
                                    pcd_background_img_prototype = pc_processor.project_pcd_to_image_plane(
                                        pcd_for_bg_projection,  # 使用过滤后的点云进行投影
                                        image_wh,
                                        world_bounds_xz_scene,
                                        density_radius_world=configs.PREPROCESSING_CONFIG["pointcloud_density_radius"],
                                        min_points_for_pixel=configs.PREPROCESSING_CONFIG[
                                            "pointcloud_min_points_for_pixel"]
                                    )
                                else:
                                    print(f"    No points in relevant Y range for PCD background of {scene_name_ifc}")

                        else:
                            print(f"    Could not load PCD for {scene_name_ifc}")
                    else:
                        print(f"    PCD path not found for {scene_name_ifc}")

                # 3. 遍历每个切片高度
                for slice_y in slice_y_levels:
                    print(f"    Slicing at Y = {slice_y:.2f}m")
                    # 为当前场景和切片高度确定世界坐标边界
                    # (复用上面为点云背景计算的 scene_bounds_xz)
                    # 如果没有点云背景，需要在这里计算一次
                    if not use_pc_background or world_bounds_xz_scene is None:
                        all_component_vertices_xz = []
                        for comp in ifc_components:
                            if comp.vertices is not None:
                                all_component_vertices_xz.append(GeometryProcessor.project_to_xz_plane(comp.vertices))
                        if not all_component_vertices_xz:
                            print(
                                f"      Warning: No component vertices to determine bounds for slice at Y={slice_y:.2f}m. Skipping this slice level.")
                            continue
                        combined_vertices_xz = np.vstack(all_component_vertices_xz)
                        x_coords, z_coords = combined_vertices_xz[:, 0], combined_vertices_xz[:, 1]
                        x_min, z_min = np.min(x_coords), np.min(z_coords)
                        x_max, z_max = np.max(x_coords), np.max(z_coords)
                        padding = max(x_max - x_min, z_max - z_min) * 0.1  # 10% padding
                        current_world_bounds_xz = (x_min - padding, z_min - padding, x_max + padding, z_max + padding)
                    else:
                        current_world_bounds_xz = world_bounds_xz_scene

                    yolo_image_annotator = YOLOImageAnnotator(
                        image_wh=image_wh,
                        world_bounds_xz=current_world_bounds_xz,
                        ifc_class_to_yolo_id_map=ifc_class_to_yolo_id_map
                    )
                    component_slices_for_image = []
                    for component in ifc_components:
                        if component.vertices is None or component.faces is None:
                            continue
                        # IFC构件顶点已经是Y-up
                        slice_polygons_3d = GeometryProcessor.slice_component_at_y(
                            component.vertices, component.faces, slice_y, slice_thickness
                        )

                        if slice_polygons_3d:  # 可能一个构件在某个高度有多个不连续的切片轮廓
                            # 对于IFC解析，slice_polygons_3d 返回的是 List[np.ndarray], 每个ndarray是XZ坐标
                            component_slices_for_image.append({
                                "ifc_type": component.ifc_type,
                                "slice_polygons_world_xz": slice_polygons_3d  # 这已经是XZ坐标了
                            })

                    if not component_slices_for_image:
                        print(f"      No component slices generated for scene {scene_name_ifc} at Y={slice_y:.2f}m.")
                        continue

                    # 4. 生成图像和YOLO标签
                    # 为当前切片高度复制点云背景图（如果存在）
                    current_pcd_background = pcd_background_img_prototype.copy() if pcd_background_img_prototype is not None else None

                    annotated_image, yolo_label_lines = yolo_image_annotator.create_annotated_image_and_labels(
                        component_slices_for_image,
                        background_image=current_pcd_background,
                        annotation_type=annotation_type
                    )

                    # 5. 保存
                    if annotated_image is not None and yolo_label_lines:
                        slice_level_str = f"_y{slice_y:.2f}"
                        output_img_path = data_manager.get_output_image_path(scene_name_base, split, slice_level_str)
                        output_lbl_path = data_manager.get_output_label_path(scene_name_base, split, slice_level_str)

                        cv2.imwrite(str(output_img_path), annotated_image)
                        with open(output_lbl_path, 'w') as f_label:
                            for line in yolo_label_lines:
                                f_label.write(line + "\n")
                        # print(f"      Saved image to {output_img_path} and label to {output_lbl_path}")
                    else:
                        print(f"      No annotations to save for {scene_name_ifc} at Y={slice_y:.2f}m.")

            except FileNotFoundError as e:
                print(f"    Error processing scene {scene_name_ifc}: {e}. Skipping.")
            except IOError as e:  # For IFC parsing errors
                print(f"    IOError processing IFC for scene {scene_name_ifc}: {e}. Skipping.")
            except Exception as e:
                print(f"    Unexpected error processing scene {scene_name_ifc}: {e}. Skipping.")
                import traceback
                traceback.print_exc()

    print("\nPreprocessing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BIMNet dataset for YOLO training.")
    # 可以添加命令行参数来覆盖configs.py中的默认值，例如：
    # parser.add_argument('--bimnet_root', type=str, help='Override BIMNet root directory')
    # parser.add_argument('--output_dir', type=str, help='Override output YOLO dataset directory')
    # ...等等

    cmd_args = parser.parse_args()

    # 如果命令行参数被提供，可以用它们来更新configs中的值
    # Example:
    # if cmd_args.bimnet_root:
    #     configs.BIMNET_ROOT_DIR = Path(cmd_args.bimnet_root)
    # if cmd_args.output_dir:
    #     configs.OUTPUT_YOLO_DIR = Path(cmd_args.output_dir)

    start_time = time.time()
    main(cmd_args)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")