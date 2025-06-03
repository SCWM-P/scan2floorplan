# data_preprocess/core/bim_parser.py
import trimesh
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class IFCComponent:
    """代表一个IFC构件及其几何信息。"""

    def __init__(self, name: str, ifc_type: str, guid: str, geometries: List[trimesh.Trimesh]):
        self.name = name
        self.ifc_type = ifc_type
        self.guid = guid  # GUID 或其他唯一标识符
        self.geometries = geometries  # 一个构件可能由多个分离的几何体组成

    def __repr__(self):
        return f"<IFCComponent name='{self.name}' type='{self.ifc_type}' geometries={len(self.geometries)}>"


class BIMSceneParser:
    """解析一个BIM场景（通常对应一个IFC文件，其构件分解为多个OBJ文件）"""

    def __init__(self, obj_component_dir: Path):
        """
        Args:
            obj_component_dir (Path): 存放该场景所有IFC构件对应OBJ文件的目录。
                                     例如 '.../BIMNet/obj/train/1px/'
        """
        self.obj_component_dir = obj_component_dir
        if not self.obj_component_dir.is_dir():
            raise FileNotFoundError(f"OBJ component directory not found: {self.obj_component_dir}")

        # 尝试加载 ifcinstances.json (如果存在)
        self.ifc_instances_data = None
        ifc_json_path = self.obj_component_dir / "ifcinstances.json"  # BIMNet通常有这个文件
        if ifc_json_path.exists():
            try:
                with open(ifc_json_path, 'r', encoding='utf-8') as f:
                    self.ifc_instances_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load or parse {ifc_json_path}: {e}")

    def get_components(self) -> List[IFCComponent]:
        """
        加载目录中所有OBJ文件作为IFC构件。
        每个OBJ文件被视为一个构件的一个或多个几何体。
        """
        components = []
        obj_files = sorted(list(self.obj_component_dir.glob("*.obj")))
        for obj_file_path in obj_files:
            component_name = obj_file_path.stem
            ifc_type = "UNKNOWN"
            guid = component_name  # 默认使用文件名作为GUID
            # 从 ifcinstances.json 获取更准确的IFC类型和GUID
            if self.ifc_instances_data:
                for instance in self.ifc_instances_data:
                    if instance.get('path') == obj_file_path.name:
                        ifc_type = instance.get('ifctype', 'UNKNOWN')
                        guid = instance.get('guid', component_name)
                        break
            else:
                # 如果没有json，尝试从文件名解析 (BIMNet的OBJ文件名通常包含IFC类型)
                # 例如 IFCWALLSTANDARDCASE-IFC#3202-RVT#238117-CurvedFalse-el34.obj
                if '-' in component_name:
                    ifc_type_from_name = component_name.split('-')[0]
                    if ifc_type_from_name.upper().startswith("IFC"):
                        ifc_type = ifc_type_from_name
            try:
                # trimesh.load_mesh 可以加载包含多个独立几何体的OBJ文件作为一个Scene对象
                # 或者加载单个几何体的OBJ文件为一个Trimesh对象
                loaded_data = trimesh.load_mesh(str(obj_file_path), process=False)  # process=False 保留原始数据
                geometries = []
                if isinstance(loaded_data, trimesh.Scene):
                    # 如果OBJ包含多个几何体（在Scene的geometry属性中）
                    for geom_name in loaded_data.geometry:
                        geometries.append(loaded_data.geometry[geom_name])
                elif isinstance(loaded_data, trimesh.Trimesh):
                    # 如果OBJ只包含一个几何体
                    geometries.append(loaded_data)
                else:
                    print(f"Warning: Skipping {obj_file_path.name}, unsupported mesh type: {type(loaded_data)}")
                    continue

                if geometries:
                    component = IFCComponent(name=component_name,
                                             ifc_type=ifc_type,
                                             guid=guid,
                                             geometries=geometries)
                    components.append(component)
                else:
                    print(f"Warning: No valid geometries found in {obj_file_path.name}")

            except Exception as e:
                print(f"Error loading OBJ file {obj_file_path.name}: {e}")

        return components