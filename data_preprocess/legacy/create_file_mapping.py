import os
import json
from pathlib import Path
base_dir = Path(r"E:\BIMNet")


def create_file_mapping():
    # 定义源目录,创建映射字典
    point_cloud_dirs = {
        'train': base_dir / "point_cloud/train",
        'test': base_dir / "point_cloud/test"
    }
    ifc_dirs = {
        'train': base_dir / "ifc/train",
        'test': base_dir / "ifc/test"
    }
    mapping = {}
    # 处理train和test目录
    for split in ['train', 'test']:
        point_cloud_files = sorted([f for f in point_cloud_dirs[split].glob("*.txt")])
        ifc_files = sorted([f for f in ifc_dirs[split].glob("*.ifc")])
        used_pc_files = set()  # 用于跟踪已经匹配的point cloud文件
        # 建立映射关系
        for ifc_file in ifc_files:
            ifc_stem = ifc_file.stem.lower()  # 获取不带扩展名的文件名并转为小写
            base_ifc_stem = ifc_stem.replace('_1', '')  # 移除_1后缀以获取基本名称
            best_match = None
            for pc_file in point_cloud_files:
                if str(pc_file) in used_pc_files:
                    continue
                pc_stem = pc_file.stem.lower()
                base_pc_stem = pc_stem.replace('_1', '')  # 移除_1后缀以获取基本名称
                # 检查基本名称是否匹配
                if base_ifc_stem in base_pc_stem:
                    # 如果两个文件都有或都没有_1后缀，或者是第一次匹配到的文件
                    if ('_1' in ifc_stem) == ('_1' in pc_stem):
                        best_match = pc_file
                        break
                    elif not best_match:  # 如果还没有找到任何匹配
                        best_match = pc_file
            
            if best_match:
                # 只保存文件名，不包含完整路径
                mapping[f"{ifc_file.name}"] = best_match.name
                used_pc_files.add(str(best_match))
    
    # 确保data_preprocess目录存在
    output_dir = Path("../data_preprocess")
    output_dir.mkdir(exist_ok=True)
    # 保存映射到JSON文件
    output_file = output_dir / "file_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=4)
    
    print(f"文件映射已保存到: {output_file}")
    print(f"共找到 {len(mapping)} 对文件映射")
    
    # 打印映射结果以供验证
    print("\n映射结果:")
    for ifc, pc in sorted(mapping.items()):
        print(f"{ifc} -> {pc}")
    
    # 打印未匹配的文件
    print("\n未匹配的IFC文件:")
    matched_ifc_files = set(mapping.keys())
    for split in ['train', 'test']:
        for ifc_file in sorted(ifc_dirs[split].glob("*.ifc")):
            if ifc_file.name not in matched_ifc_files:
                print(f"- {ifc_file.name}")
    
    print("\n未匹配的Point Cloud文件:")
    for split in ['train', 'test']:
        for pc_file in sorted(point_cloud_dirs[split].glob("*.txt")):
            if pc_file.name not in mapping.values():
                print(f"- {pc_file.name}")

if __name__ == "__main__":
    create_file_mapping() 