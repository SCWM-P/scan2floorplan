# 目的：该文件用于读取IFC构件的.obj文件，并获取构件的属性信息和投影交点

# 在3D建模中，通常使用以下坐标轴约定：
# X轴：左右方向（水平）
# Y轴：上下方向（垂直/高度）
# Z轴：前后方向（深度）

import trimesh
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

class IFCReader:
    def __init__(self):
        """
        初始化IFC读取器
        """
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.normals = None
        self.geometries = None
        
    def load_obj(self, file_path: str) -> bool:
        """
        加载.obj文件
        
        Args:
            file_path (str): .obj文件的路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            # 加载文件
            scene = trimesh.load(file_path)
            
            # 如果是Scene对象，不合并几何体，而是分别处理
            if isinstance(scene, trimesh.Scene):
                if len(scene.geometry) == 0:
                    print(f"No geometry found in {file_path}")
                    return False
                # 保存所有几何体
                self.geometries = list(scene.geometry.values())
                # 默认使用第一个几何体
                self.mesh = self.geometries[0]
            else:
                self.mesh = scene
                self.geometries = [scene]
                
            # 获取顶点和面
            self.vertices = np.array(self.mesh.vertices)
            self.faces = np.array(self.mesh.faces)
            self.normals = np.array(self.mesh.vertex_normals)
            
            return True
        except Exception as e:
            print(f"Error loading .obj file: {e}")
            return False
            
    def get_mesh_info(self) -> Dict:
        """
        获取网格信息
        
        Returns:
            Dict: 包含顶点数、面数等信息的字典
        """
        if self.mesh is None:
            return {}
            
        return {
            "vertices": self.vertices,
            "faces": self.faces,
            "bounds": self.mesh.bounds,
            "volume": self.mesh.volume,
            "area": self.mesh.area
        }
        
    def get_vertices(self) -> Optional[np.ndarray]:
        """
        获取顶点坐标
        
        Returns:
            Optional[np.ndarray]: 顶点坐标数组，如果未加载则返回None
        """
        return self.vertices

    def get_nonduplicated_vertices(self) -> Optional[np.ndarray]:
        """
        获取非重复的顶点坐标
        
        Returns:
            Optional[np.ndarray]: 去重后的顶点坐标数组，如果未加载则返回None
        """
        if self.vertices is None:
            return None
        # 使用numpy的unique函数去重，axis=0表示按行去重
        return np.unique(self.vertices, axis=0)
        
    def get_faces(self) -> Optional[np.ndarray]:
        """
        获取面索引
        
        Returns:
            Optional[np.ndarray]: 面索引数组，如果未加载则返回None
        """
        return self.faces
        
    def get_normals(self) -> Optional[np.ndarray]:
        """
        获取顶点法线
        
        Returns:
            Optional[np.ndarray]: 顶点法线数组，如果未加载则返回None
        """
        return self.normals
        
    def visualize(self):
        """
        可视化网格
        """
        if self.mesh is not None:
            self.mesh.show()
            
    def save_processed_data(self, output_path: str):
        """
        保存处理后的数据
        
        Args:
            output_path (str): 输出文件路径
        """
        if self.mesh is not None:
            try:
                self.mesh.export(output_path)
                print(f"Data saved to {output_path}")
            except Exception as e:
                print(f"Error saving data: {e}")

    def get_ifc_info(self, file_path: str) -> Dict:
        """
        获取IFC对象的属性信息
        
        Args:
            file_path (str): .obj文件的路径
            
        Returns:
            Dict: 包含IFC对象属性的字典，包括类型、名称等信息
        """
        try:
            # 从文件路径中提取IFC类型
            file_name = os.path.basename(file_path)
            ifc_type = file_name.split('-')[0]
            
            # 从ifcinstances.json中读取属性信息
            json_path = os.path.join(os.path.dirname(file_path), 'ifcinstances.json')
            if os.path.exists(json_path):
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    instances = json.load(f)
                    for instance in instances:
                        if instance['path'] == file_name:
                            return {
                                'ifc_type': instance['ifctype'],
                                'name': instance['name'],
                                'type': instance['type'],
                                'id': instance['id'],
                                'guid': instance['guid']
                            }
            
            # 如果找不到对应的属性信息，返回基本类型信息
            return {
                'ifc_type': ifc_type,
                'name': file_name,
                'type': 'unknown'
            }
            
        except Exception as e:
            print(f"Error getting IFC properties: {e}")
            return {
                'ifc_type': 'unknown',
                'name': os.path.basename(file_path),
                'type': 'unknown'
            }

    def get_all_vertices(self) -> List[np.ndarray]:
        """
        获取所有构件的顶点
        
        Returns:
            List[np.ndarray]: 所有几何体的顶点列表，按X坐标排序
        """
        all_vertices = []
        
        for geometry in self.geometries:
            if geometry.vertices is None or geometry.faces is None:
                continue
            
            # 对顶点进行去重
            rounded_vertices = np.round(geometry.vertices, 6)
            _, unique_indices, inverse_indices = np.unique(rounded_vertices, axis=0, return_index=True, return_inverse=True)
            vertices = geometry.vertices[unique_indices]
            
            # 更新面片中的顶点索引
            faces = inverse_indices[geometry.faces]
            
            # 合并共面三角面片
            def merge_coplanar_faces(vertices, faces):
                # 计算每个面的法向量
                face_normals = []
                valid_faces = []
                
                for i, face in enumerate(faces):
                    v1, v2, v3 = vertices[face]
                    # 计算法向量
                    normal = np.cross(v2 - v1, v3 - v1)
                    # 检查法向量是否为零向量
                    norm = np.linalg.norm(normal)
                    if norm < 1e-6:  # 如果法向量太小，跳过这个面
                        continue
                    # 归一化
                    normal = normal / norm
                    face_normals.append(normal)
                    valid_faces.append(i)
                # 如果没有有效的面，返回空列表
                if not valid_faces:
                    return []
                # 更新面片索引
                faces = faces[valid_faces]
                face_normals = np.array(face_normals)
                # 找出共面的面片
                coplanar_groups = []
                used_faces = set()
                
                for i in range(len(faces)):
                    if i in used_faces:
                        continue
                    group = [i]
                    used_faces.add(i)
                    
                    # 找出与当前面共面的其他面
                    for j in range(i+1, len(faces)):
                        if j in used_faces:
                            continue
                        
                        # 检查法向量是否平行
                        dot_product = np.abs(np.dot(face_normals[i], face_normals[j]))
                        if not np.allclose(dot_product, 1.0, atol=1e-4):
                            continue
                        
                        # 检查是否有公共顶点
                        face1_vertices = set(faces[i])
                        face2_vertices = set(faces[j])
                        if not face1_vertices.intersection(face2_vertices):
                            continue
                        
                        group.append(j)
                        used_faces.add(j)
                        
                    coplanar_groups.append(group)
                
                # 收集所有顶点
                all_vertices = set()
                for group in coplanar_groups:
                    for face_idx in group:
                        face = faces[face_idx]
                        all_vertices.update(face)
                
                return list(all_vertices)
            
            # 获取合并后的顶点
            vertex_indices = merge_coplanar_faces(vertices, faces)
            
            if vertex_indices:
                # 获取对应的顶点坐标
                unique_points = vertices[vertex_indices]
                
                # 按X坐标排序
                all_vertices.append(unique_points[np.argsort(unique_points[:, 0])])
        
        return all_vertices

    def process_ifc_components(self, folder_path: str) -> List[Dict]:
        """
        处理文件夹下的所有IFC构件
        
        Args:
            folder_path (str): 包含IFC构件的文件夹路径
            
        Returns:
            List[Dict]: 包含每个构件信息的列表，每个构件信息包含：
                - reader: IFCReader实例
                - height_range: 高度范围
                - ifc_info: IFC属性信息
                - file_path: 文件路径
        """
        components = []
        failed_files = []
        
        # 遍历文件夹下的所有.obj文件
        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.obj'):
                continue
                
            file_path = os.path.join(folder_path, file_name)
            
            try:
                # 创建新的reader实例
                reader = IFCReader()
                
                # 加载.obj文件
                if not reader.load_obj(file_path):
                    print(f"Failed to load {file_name}")
                    failed_files.append((file_name, "Failed to load .obj file"))
                    continue
                    
                # 获取非重复顶点
                nonduplicated_vertices = reader.get_nonduplicated_vertices()
                if nonduplicated_vertices is None or len(nonduplicated_vertices) == 0:
                    print(f"No vertices found in {file_name}")
                    failed_files.append((file_name, "No vertices found"))
                    continue
                    
                # 计算高度范围
                y_coords = nonduplicated_vertices[:, 1]
                height_range = (np.min(y_coords), np.max(y_coords))
                
                # 获取IFC属性信息
                ifc_info = reader.get_ifc_info(file_path)
                
                # 保存构件信息
                components.append({
                    'reader': reader,
                    'height_range': height_range,
                    'ifc_info': ifc_info,
                    'file_path': file_path
                })
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                failed_files.append((file_name, str(e)))
                continue
            
        # 打印处理结果
        print(f"\n成功处理了 {len(components)} 个IFC构件")
        if failed_files:
            print(f"处理失败的 {len(failed_files)} 个文件:")
            for file_name, error in failed_files:
                print(f"- {file_name}: {error}")
            
        return components

# 使用示例
if __name__ == "__main__":
    # 创建IFC读取器实例
    reader = IFCReader()
    
    # 定义需要处理的IFC类型
    target_ifc_types = [
        'IFCDOOR',
        'IFCCOLUMN',
        'IFCWINDOW',
        'IFCWALLSTANDARDCASE',
        'IFCBEAM'
    ]
    
    # 处理文件夹下的所有IFC构件
    project_root = Path(__file__).resolve().parent.parent.parent
    folder_path = project_root / "data/raw/BIMNet/obj/train/1px"
    components = reader.process_ifc_components(folder_path)
    output_path = project_root / "data_preprocess/tmp.txt"
    valid_components = 0  # 统计有角点的构件数量
    filtered_components = 0  # 统计符合IFC类型的构件数量
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"处理了 {len(components)} 个IFC构件\n")
        f.write(f"目标IFC类型: {', '.join(target_ifc_types)}\n\n")
        
        for i, comp in enumerate(components):
            try:
                # 检查IFC类型是否在目标列表中
                ifc_type = comp['ifc_info']['ifc_type']
                if ifc_type not in target_ifc_types:
                    continue
                    
                filtered_components += 1
                
                # 获取构件角点
                vertices = comp['reader'].get_all_vertices()
                
                # 只写入有角点的构件
                if vertices:
                    valid_components += 1
                    # 写入构件信息
                    f.write(f"构件 {i+1}:\n")
                    f.write(f"文件路径: {comp['file_path']}\n")
                    f.write(f"IFC类型: {comp['ifc_info']['ifc_type']}\n")
                    f.write(f"高度范围: {comp['height_range']}\n")
                    f.write(f"角点:\n")
                    for j, vertex_group in enumerate(vertices, 1):
                        f.write(f"几何体 No.{j}:\n")
                        # 格式化输出，确保所有数值都显示4位小数
                        formatted_points = np.array2string(vertex_group, 
                                                         formatter={'float_kind': lambda x: f"{x:.4f}"},
                                                         separator=', ')
                        f.write(f"{formatted_points}\n")
                    f.write("\n")
            except Exception as e:
                f.write(f"构件 {i+1} 处理出错: {str(e)}\n\n")
        
        # 写入统计信息
        f.write(f"\n共有 {filtered_components} 个构件符合目标IFC类型")
        f.write(f"\n其中 {valid_components} 个构件有角点")
    
    print(f"\n处理了 {len(components)} 个IFC构件")
    print(f"符合目标IFC类型的构件: {filtered_components} 个")
    print(f"其中 {valid_components} 个构件有角点")
    print(f"结果已保存到 {output_path}")

        