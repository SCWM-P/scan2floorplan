import json
import os
from pathlib import Path

# Define label mapping
LABEL_MAP = {
    "IFCDOOR": 0,
    "IFCWALLSTANDARDCASE": 1,
    "IFCWINDOW": 2,
}

def convert_json_to_yolo(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_w = data['imageWidth']
    image_h = data['imageHeight']   
    
    txt_path = os.path.join(output_dir, Path(json_path).stem + '.txt')
    with open(txt_path, 'w') as out_file:
        for shape in data['shapes']:
            label = shape['label']
            if label not in LABEL_MAP:
                continue
            
            # Get polygon points
            points = shape['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            # Calculate bounding box
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            # Convert to YOLO format
            x_center = (x_min + x_max) / 2 / image_w
            y_center = (y_min + y_max) / 2 / image_h
            width = (x_max - x_min) / image_w
            height = (y_max - y_min) / image_h
            
            # Write to file
            out_file.write(f"{LABEL_MAP[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")