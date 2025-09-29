import cv2
import numpy as np
import json
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent

class AnnotationAugmentation:
    def __init__(self, margin=100):
        self.margin = margin
        self.points = []
        self.drawing = False
        self.current_point = None
        self.image = None
        self.window_name = 'Select Region'
        self.rect_start = None
        self.rect_end = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rect_start = (x, y)
            self.rect_end = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.rect_start is not None and self.rect_end is None:
                img_copy = self.image.copy()
                cv2.rectangle(img_copy, self.rect_start, (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.rect_end = (x, y)
            
    def select_region(self, image_path):
        """交互式选择区域"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            # 如果已经选择了矩形，持续显示
            if self.rect_start is not None and self.rect_end is not None:
                img_copy = self.image.copy()
                cv2.rectangle(img_copy, self.rect_start, self.rect_end, (0, 255, 0), 2)
                cv2.imshow(self.window_name, img_copy)
            else:
                cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                return None
            elif key == ord('y') and self.rect_start is not None and self.rect_end is not None:
                # 确认选择
                break
            elif key == ord('n') and self.rect_start is not None and self.rect_end is not None:
                # 重新选择
                self.rect_start = None
                self.rect_end = None
                cv2.imshow(self.window_name, self.image)
                
        cv2.destroyAllWindows()
        
        if self.rect_start is None or self.rect_end is None:
            return None
        # 将矩形转换为多边形点
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        points = np.array([
            [min(x1, x2), min(y1, y2)],  # 左上
            [max(x1, x2), min(y1, y2)],  # 右上
            [max(x1, x2), max(y1, y2)],  # 右下
            [min(x1, x2), max(y1, y2)]   # 左下
        ])
        return points
    
    def process_annotation(self, json_path, points):
        """处理标注文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 获取图片尺寸
        img_height = data['imageHeight']
        img_width = data['imageWidth']
        
        # 创建mask
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        # 确保points是整数类型
        points_int = points.astype(np.int32)
        cv2.fillPoly(mask, [points_int], 255)
        
        # 计算选择区域的边界
        x_min = int(min(points[:, 0]))
        y_min = int(min(points[:, 1]))
        x_max = int(max(points[:, 0]))
        y_max = int(max(points[:, 1]))
        
        # 处理每个标注
        new_shapes = []
        for shape in data['shapes']:
            points_array = np.array(shape['points'])
            # 确保points_array是整数类型
            points_array_int = points_array.astype(np.int32)
            
            # 创建shape的mask
            shape_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.fillPoly(shape_mask, [points_array_int], 255)
            
            # 检查是否有交集
            intersection = cv2.bitwise_and(shape_mask, mask)
            if cv2.countNonZero(intersection) > 0:
                # 检查是否完全在选区内
                if cv2.countNonZero(shape_mask) == cv2.countNonZero(intersection):
                    # 完全在选区内，直接转换坐标
                    new_shape = shape.copy()
                    for point in new_shape['points']:
                        point[0] = float(point[0] - x_min)  # 转换为Python float类型
                        point[1] = float(point[1] - y_min)
                    new_shapes.append(new_shape)
                else:
                    # 部分在选区内，计算交集
                    new_shape = self._calculate_intersection(shape, points, x_min, y_min)
                    if new_shape is not None:
                        new_shapes.append(new_shape)
                    
        return new_shapes
    
    def _calculate_intersection(self, shape, region_points, x_min, y_min):
        """计算标注框与选择区域的交集"""
        # 获取标注框的所有点
        shape_points = np.array(shape['points'])
        
        # 将两个多边形都转换为局部坐标系
        local_shape_points = shape_points - np.array([x_min, y_min])
        local_region_points = region_points - np.array([x_min, y_min])
        
        # 确保点是整数类型
        local_shape_points = local_shape_points.astype(np.int32)
        local_region_points = local_region_points.astype(np.int32)
        
        # 创建两个mask
        h = int(max(local_region_points[:, 1])) + 1
        w = int(max(local_region_points[:, 0])) + 1
        shape_mask = np.zeros((h, w), dtype=np.uint8)
        region_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 填充两个mask
        cv2.fillPoly(shape_mask, [local_shape_points], 255)
        cv2.fillPoly(region_mask, [local_region_points], 255)
        
        # 计算交集
        intersection = cv2.bitwise_and(shape_mask, region_mask)
        
        # 找到交集的轮廓
        contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 使用最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 如果轮廓太小（小于500像素），忽略
        if cv2.contourArea(max_contour) < 500:
            return None
            
        # 创建新的shape
        new_shape = shape.copy()
        new_points = []
        
        # 获取轮廓点（已经在局部坐标系中）
        for point in max_contour:
            x, y = point[0]
            new_points.append([float(x), float(y)])  # 转换为Python float类型
            
        # 计算非白色点的凸包围框
        non_white_points = []
        for y in range(h):
            for x in range(w):
                if intersection[y, x] > 0:  # 非白色点
                    non_white_points.append([float(x), float(y)])
        
        if non_white_points:
            # 计算凸包
            non_white_points = np.array(non_white_points)
            hull = cv2.convexHull(non_white_points.astype(np.float32))
            exact_expand_points = hull.reshape(-1, 2).tolist()
            new_shape['exact_expand_points'] = exact_expand_points
            
        new_shape['points'] = new_points
        
        return new_shape
    
    def add_margin(self, image, points):
        """添加margin"""
        # 计算边界框
        x_min = int(min(points[:, 0]))
        y_min = int(min(points[:, 1]))
        x_max = int(max(points[:, 0]))
        y_max = int(max(points[:, 1]))
        
        # 裁剪图片
        cropped = image[y_min:y_max, x_min:x_max]
        
        # 创建带白边的图片
        h, w = cropped.shape[:2]
        new_h = h + 2 * self.margin
        new_w = w + 2 * self.margin
        new_image = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
        new_image[self.margin:self.margin+h, self.margin:self.margin+w] = cropped
        
        # 调整点坐标
        adjusted_points = points - np.array([x_min, y_min]) + np.array([self.margin, self.margin])
        
        return new_image, adjusted_points, (x_min, y_min)
    
    def generate_labelme_json(self, image_path, json_path, output_dir, points):
        """生成新的labelme格式文件"""
        # 读取原始图片和标注
        image = cv2.imread(image_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 添加margin并调整坐标
        cropped_image, adjusted_points, offset = self.add_margin(image, points)
        
        # 处理标注
        new_shapes = self.process_annotation(json_path, points)
        
        # 为所有标注点添加margin偏移
        for shape in new_shapes:
            for point in shape['points']:
                point[0] += self.margin
                point[1] += self.margin
        
        # 获取输入文件的目录和文件名（不含扩展名）
        input_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 查找当前目录下已有的最大编号
        existing_files = [f for f in os.listdir(input_dir) if f.startswith(base_name) and f.endswith('.png')]
        max_num = 0
        for file in existing_files:
            try:
                num = int(file.split('_')[-1].split('.')[0])
                max_num = max(max_num, num)
            except ValueError:
                continue
        
        # 生成新的编号
        new_num = max_num + 1
        
        # 构建新的文件名
        new_image_name = f"{base_name}_{new_num}.png"
        new_json_name = f"{base_name}_{new_num}.json"
        
        # 创建新的标注数据
        new_data = {
            'version': data['version'],
            'flags': data['flags'],
            'shapes': new_shapes,
            'imagePath': new_image_name,
            'imageData': None,
            'imageHeight': cropped_image.shape[0],
            'imageWidth': cropped_image.shape[1]
        }
        
        # 保存新的图片和标注文件
        output_image_path = os.path.join(input_dir, new_image_name)
        output_json_path = os.path.join(input_dir, new_json_name)
        
        cv2.imwrite(output_image_path, cropped_image)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
            
        return output_image_path, output_json_path

def main():
    # 初始化增广器
    augmentor = AnnotationAugmentation(margin=100)
    image_path = ROOT_DIR / 'data_preprocess/output/projection_only_pcd.png'
    json_path = ROOT_DIR / 'data_preprocess/output/projection_only_pcd.json'
    while True:
        try:
            # 选择区域
            points = augmentor.select_region(image_path)
            if points is None:
                print("已取消选择")
                break
            # 生成新的标注文件
            output_image, output_json = augmentor.generate_labelme_json(
                image_path, json_path, None, points
            )
            print(f"增广完成！")
            print(f"输出图片：{output_image}")
            print(f"输出标注：{output_json}")
            # 询问是否继续
            choice = input("是否继续增广？(y/n): ").lower()
            if choice != 'y':
                break
        except Exception as e:
            print(f"处理过程中出现错误：{str(e)}")
            break

if __name__ == '__main__':
    main()
