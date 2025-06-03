# data_preprocess/configs/default_config.py
from pathlib import Path
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path("core").resolve().parent.parent

BIMNET_ROOT_DIR = PROJECT_ROOT / "data/raw/BIMNet"
OUTPUT_YOLO_DATASET_DIR = PROJECT_ROOT / "data_preprocess/output/bimnet_yolo"
FILE_MAPPING_JSON = PROJECT_ROOT / "data_preprocess/file_mapping.json"

# 切片参数
SLICE_HEIGHTS = [1.5, 2.0]
SLICE_THICKNESS = 0.05
# YOLO图像参数
YOLO_IMAGE_WIDTH = 640
YOLO_IMAGE_HEIGHT = 640
# 数据集生成参数
USE_POINTCLOUD_BACKGROUND = True  # 是否使用点云切片作为图像背景
ANNOTATION_TYPE = "convex_hull"  # "convex_hull" 或 "min_rect"
# PRESET_GLOBAL_XZ_BOUNDS = (-10.0, -10.0, 10.0, 10.0)
PRESET_GLOBAL_XZ_BOUNDS = None # 设置为None则动态计算

# IFC类别到YOLO类别ID的映射
IFC_TO_YOLO_CLASS_MAPPING = {
    "IFCWALL": 0,
    "IFCWALLSTANDARDCASE": 0,  # 通常与IFCWALL视为一类
    "IFCCOLUMN": 1,
    "IFCDOOR": 2,
    "IFCWINDOW": 3,
    "IFCSLAB": 4, # 楼板
    "IFCBEAM": 5, # 梁
    "IFCROOF": 6, # 屋顶
    # "IFCSTAIR": 7,
    # "IFCRAILING": 8,
    # "IFCFURNISHINGELEMENT": 9,
    # ... 其他你关心的IFC类型
}

# YOLO数据集划分比例 (如果需要在此处进行划分)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.0 # 如果没有测试集