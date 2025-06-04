# data_preprocess/core/data_manager.py
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class DataManager:
    def __init__(self, bimnet_root_dir: Path, output_yolo_dir: Path, mapping_file: Path):
        self.bimnet_root_dir = bimnet_root_dir
        self.output_yolo_dir = output_yolo_dir
        self.mapping_file = mapping_file

        if not self.bimnet_root_dir.exists():
            raise FileNotFoundError(f"BIMNet root directory not found: {self.bimnet_root_dir}")
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"File mapping JSON not found: {self.mapping_file}")
        with open(self.mapping_file, 'r') as f:
            self.file_mapping = json.load(f)

        self.output_yolo_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (self.output_yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def get_ifc_path(self, ifc_scene_name: str, split: str = 'train') -> Path:
        """获取IFC文件的完整路径 (e.g., 1px.ifc)"""
        if not ifc_scene_name.endswith(".ifc"):
            ifc_scene_name += ".ifc"
        path = self.bimnet_root_dir / "ifc" / split / ifc_scene_name
        if not path.exists():
            raise FileNotFoundError(f"IFC file not found for scene {ifc_scene_name} in split {split} at {path}")
        return path

    def get_pcd_path_for_ifc_scene(self, ifc_scene_name: str, split: str = 'train') -> Optional[Path]:
        """根据IFC场景名获取对应的点云文件路径。"""
        ifc_base_name = ifc_scene_name.replace(".ifc", "")
        # 在file_mapping中查找匹配的ifc文件名
        # file_mapping.json的key是ifc文件名，value是点云文件名
        # 例如: "1px.ifc": "1pXnuDYAj8r.txt"
        pcd_filename = None
        for ifc_key, pc_val in self.file_mapping.items():
            if ifc_key.startswith(ifc_base_name) and ifc_key.endswith(".ifc"):  # 处理可能带楼层后缀的情况，如 "7y3_1.ifc"
                pcd_filename = pc_val
                break

        if not pcd_filename:
            # 尝试直接用ifc文件名（不含.ifc）去匹配点云文件名中的一部分
            for pc_val in self.file_mapping.values():
                if ifc_base_name in pc_val:
                    pcd_filename = pc_val
                    break

        if not pcd_filename:
            print(f"Warning: No PCD file mapping found for IFC scene {ifc_scene_name} in {self.mapping_file}")
            return None
        path = self.bimnet_root_dir / "point_cloud" / split / pcd_filename
        if not path.exists():
            # 尝试在另一个split中查找（BIMNet的train/test划分可能不一致）
            other_split = 'test' if split == 'train' else 'train'
            path_other = self.bimnet_root_dir / "point_cloud" / other_split / pcd_filename
            if path_other.exists():
                path = path_other
            else:
                print(f"Warning: PCD file not found for IFC scene {ifc_scene_name} at {path} or {path_other}")
                return None
        return path

    def get_transform_matrix_for_ifc_scene(self, ifc_scene_name: str, split: str = 'train') -> Optional[Path]:
        """根据IFC场景名获取对应的mat_pc2obj变换矩阵文件路径。"""
        # mat_pc2obj 文件名通常与 IFC 文件名（去除.ifc后缀）一致，但扩展名为.txt
        # 例如, 1px.ifc -> 1px.txt
        matrix_filename = ifc_scene_name.replace(".ifc", ".txt")
        path = self.bimnet_root_dir / "mat_pc2obj" / split / matrix_filename
        if not path.exists():
            # 尝试在另一个split中查找
            other_split = 'test' if split == 'train' else 'train'
            path_other = self.bimnet_root_dir / "mat_pc2obj" / other_split / matrix_filename
            if path_other.exists():
                path = path_other
            else:
                print(
                    f"Warning: Transform matrix file not found for IFC scene {ifc_scene_name} at {path} or {path_other}")
                return None
        return path

    def get_scene_names(self, split: str = 'train') -> List[str]:
        """获取指定split下的所有IFC场景文件名（不含路径，含.ifc后缀）。"""
        ifc_dir = self.bimnet_root_dir / "ifc" / split
        if not ifc_dir.is_dir():
            return []
        return [f.name for f in ifc_dir.glob("*.ifc")]

    def get_output_image_path(self, base_name: str, split: str = 'train', slice_level_str: str = "") -> Path:
        """获取输出图像的保存路径。base_name 通常是 ifc_scene_name_without_extension + slice_info"""
        filename = f"{base_name}{slice_level_str}.png"  # 统一用png
        return self.output_yolo_dir / split / "images" / filename

    def get_output_label_path(self, base_name: str, split: str = 'train', slice_level_str: str = "") -> Path:
        """获取输出YOLO标签文件的保存路径。"""
        filename = f"{base_name}{slice_level_str}.txt"
        return self.output_yolo_dir / split / "labels" / filename