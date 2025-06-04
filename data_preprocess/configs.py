# data_preprocess/configs.py
from pathlib import Path

# 项目根目录 (假设 data_preprocess 目录位于 scan2floorplan 项目根目录下)
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scan2floorplan/

# BIMNet数据集的根目录
BIMNET_ROOT_DIR = PROJECT_ROOT / "data/raw/BIMNet"

# 输出YOLO数据集的根目录
OUTPUT_YOLO_DIR = PROJECT_ROOT / "data_preprocess/output/bimnet_yolo_dataset"

# IFC与点云文件映射关系的JSON文件路径
FILE_MAPPING_JSON = PROJECT_ROOT / "data_preprocess/file_mapping.json"

# 预处理参数
PREPROCESSING_CONFIG = {
    "slice_y_levels": [1.5],  # 进行切片的Y轴高度列表 (米) - 注意：统一使用Y轴为高度
    "slice_thickness": 0.05,    # 切片厚度 (米)
    "yolo_image_size_wh": (640, 640), # (width, height)
    "use_pointcloud_background": True, # 是否使用点云切片作为背景
    "pointcloud_density_radius": 0.05, # 点云密度图渲染时考虑的邻域半径 (米)
    "pointcloud_min_points_for_pixel": 1, # 密度图像素上最少点数才着色

    # IFC类型到YOLO类别ID的映射 (需要根据你的YOLO模型定义)
    # 这里仅为示例，你需要定义你的YOLO模型期望的类别
    "ifc_to_yolo_class_map": {
        "IFCWALL": 0,
        "IFCWALLSTANDARDCASE": 0, # BIMNet中墙体可能有这个类型
        "IFCCOLUMN": 1,
        "IFCDOOR": 2,
        "IFCWINDOW": 3,
        "IFCSLAB": 4, # 楼板
        "IFCBEAM": 5, # 梁
        # 根据需要添加更多IFC类型
        # ... 其他如 IFCSAIR, IFCRALLING, IFCFURNISHINGELEMENT 等
    },
    # 考虑只处理特定的IFC实体类型
    "target_ifc_entities": [
        "IFCWALL", "IFCWALLSTANDARDCASE", "IFCCOLUMN", "IFCDOOR", "IFCWINDOW", "IFCSLAB", "IFCBEAM"
    ],
    "annotation_type": "convex_hull",  # "convex_hull" 或 "min_bounding_rect"
    "coordinate_system_y_up": True, # 确认最终输出和处理都基于Y轴向上
}

# 确保输出目录存在
OUTPUT_YOLO_DIR.mkdir(parents=True, exist_ok=True)
for split in ["train", "val", "test"]:
    (OUTPUT_YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"BIMNet Root: {BIMNET_ROOT_DIR}")
    print(f"Output YOLO Dir: {OUTPUT_YOLO_DIR}")
    print(f"File Mapping JSON: {FILE_MAPPING_JSON}")