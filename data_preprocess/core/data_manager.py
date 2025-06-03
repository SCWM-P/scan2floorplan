# data_preprocess/core/data_manager.py
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

class DataManager:
    """
    管理BIMNet数据集的文件路径、映射关系和输出YOLO数据集的结构。
    """
    def __init__(self, bimnet_root_dir: Path, output_yolo_dir: Path, mapping_file: Path):
        self.bimnet_root_dir = bimnet_root_dir
        self.output_yolo_dir = output_yolo_dir
        self.mapping_file = mapping_file

        if not self.bimnet_root_dir.exists():
            raise FileNotFoundError(f"BIMNet root directory not found: {self.bimnet_root_dir}")
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"File mapping JSON not found: {self.mapping_file}")
        with open(self.mapping_file, 'r') as f:
            self._file_mapping = json.load(f)
        self.output_yolo_dir.mkdir(parents=True, exist_ok=True)

    def get_ifc_scenes(self, split: str = 'train') -> List[str]:
        """获取指定split下的所有IFC场景名称 (不含.ifc后缀)"""
        ifc_dir = self.bimnet_root_dir / "ifc" / split
        if not ifc_dir.exists():
            return []
        return sorted([f.stem for f in ifc_dir.glob("*.ifc")])

    def get_pcd_filename_for_ifc_scene(self, ifc_scene_name_with_ext: str) -> Optional[str]:
        """根据IFC文件名 (带扩展名) 获取对应的点云文件名。"""
        return self._file_mapping.get(ifc_scene_name_with_ext)

    def get_pcd_path_for_ifc_scene(self, ifc_scene_name: str, split: str = 'train') -> Optional[Path]:
        """根据IFC场景名 (不带扩展名) 和split获取点云文件完整路径。"""
        ifc_filename_with_ext = f"{ifc_scene_name}.ifc"
        pcd_filename = self.get_pcd_filename_for_ifc_scene(ifc_filename_with_ext)
        if pcd_filename:
            return self.bimnet_root_dir / "point_cloud" / split / pcd_filename
        return None

    def get_obj_component_dir_for_ifc_scene(self, ifc_scene_name: str, split: str = 'train') -> Path:
        """获取IFC场景对应的OBJ构件存放目录。"""
        # BIMNet中，OBJ文件通常存储在以IFC文件名（不含扩展名）命名的子目录中
        # 例如： data/raw/BIMNet/obj/train/1px/ 存放 1px.ifc 对应的所有构件OBJ
        return self.bimnet_root_dir / "obj" / split / ifc_scene_name

    def get_transform_matrix_path_for_ifc_scene(self, ifc_scene_name: str, split: str = 'train') -> Optional[Path]:
        """获取IFC场景对应的点云到OBJ坐标系的变换矩阵文件路径。"""
        # 变换矩阵文件名与IFC文件名一致，扩展名为.txt
        # 例如: data/raw/BIMNet/mat_pc2obj/train/1px.txt
        transform_file = self.bimnet_root_dir / "mat_pc2obj" / split / f"{ifc_scene_name}.txt"
        return transform_file if transform_file.exists() else None

    def get_output_image_path(self, base_name: str, slice_height_str: str, split: str = 'train') -> Path:
        """获取生成的YOLO图像的输出路径。"""
        image_dir = self.output_yolo_dir / split / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        return image_dir / f"{base_name}_slice_{slice_height_str}.png" # 或 .jpg

    def get_output_label_path(self, base_name: str, slice_height_str: str, split: str = 'train') -> Path:
        """获取生成的YOLO标签文件的输出路径。"""
        label_dir = self.output_yolo_dir / split / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        return label_dir / f"{base_name}_slice_{slice_height_str}.txt"

    def load_transform_matrix(self, matrix_path: Path) -> Optional[np.ndarray]:
        """加载4x4变换矩阵。"""
        if not matrix_path or not matrix_path.exists():
            return None
        try:
            return np.loadtxt(matrix_path, dtype=np.float32)
        except Exception as e:
            print(f"Error loading transformation matrix {matrix_path}: {e}")
            return None