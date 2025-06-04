Data organization
=========================

The main dataset resides in the "data" directory.  There is a separate subdirectory for every house, which is named by a unique string (e.g., "1pXnuDYAj8r").  Within each house directory, there are separate directories for different types of data as follows:

Data provided directly by Matterport:
- matterport_camera_intrinsics = camera intrinsics provided by matterport
- matterport_camera_poses = camera extrinsics provided by matterport
- matterport_color_images = color images provided by matterport
- matterport_depth_images = depth images provided by matterport
- matterport_hdr_images = HDR images provided by matterport
- matterport_mesh = textured mesh provided by matteport
- matterport_skybox_images = skybox images provided by matteport

Data computed for the convenience of future research:
- undistorted_camera_parameters = camera intrinsics and extrinsics after undistortion
- undistorted_color_images = color images after undistortion
- undistorted_depth_images = depth images after undistortion
- undistorted_normal_images = normal and boundary images aligned with undistorted depth images
- poisson_meshes = low-resolution meshes resulting from poisson mesh reconstruction

Manually specified annotations:
- cameras = camera extrinsics for manually chosen good view(s)
- house_segmentations = manually specified semantic segmentations of houses into regions
- region_segmentations = manually specified semantic segmentations of regions into objects 

Details about each of the data directories follow.



General file naming conventions for matterport images:
---------------------

The Matterport Pro camera consists of three Primesense Carmine depth and RGB sensors placed on an eye-height tripod.   The
three cameras (camera index 0-2) are oriented diagonally up, flat, and diagonally down, cumulatively covering most of the vertical field of view. During capture, the camera rotates  around its vertical axis and stops at 60-degree intervals (yaw index 0-5). So, for each tripod placement (panorama_uuid), a total of 18 images are captured.  Files associated with these images adhere to the following naming convention:

    <panorama_uuid>_<imgtype><camera_index>_<yaw_index>.<extension>

where <panorama_uuid> is a unique string, <camera_index> is [0-5], and <yaw_index> is [0-2].  <imgtype> is 'j' for HDR images, 'i' for tone-mapped color images, 'd' for depth images, "skybox" for skybox images, "pose" for camera pose files, and "intrinsics" for camera intrinsics files.  The extension is ".jxr" for HDR images, ".jpg" for tone-mapped color images, and ".png" for depth and normal images.


matterport_hdr_images
---------------------

Raw HDR images in jxr format.   

Each RGB channel provides a 16-bit value (jxr_val[i]) that can be mapped to intensity (col_val[i]) with:

    for (int i = 0; i < 3; i++) {
      if (jxr_val[i] <= 3000) col_val[i] = jxr_val[i] * 8e-8
      else col_val[i] = 0.00024 * 1.0002 ** (jxr_val[i] - 3000)
    }
    
    col_val[0] *= 0.8
    col_val[1] *= 1.0
    col_val[2] *= 1.6


​    
matterport_color_images
---------------------

Tone-mapped color images in jpg format.


matterport_depth_images
---------------------

Reprojected depth images aligned with the
color images.  Every depth image a 16 bit
PNG containing the pixel's distance in the z-direction from the
camera center (_not_ the euclidean distance from the camera
center), 0.25 mm per value (divide by 4000 to get meters).
A zero value denotes 'no reading'.


matterport_camera_intrinsics
---------------------

Intrinsic parameters for every camera, stored as ASCII in the following format (using OpenCV's notation):

    width height fx fy cx cy k1 k2 p1 p2 k3

The Matterport camera axis convention is:
    x-axis: right
    y-axis: down
    z-axis: "look"

The image origin is at the top-left of the image.

Sample code to map from coordinates without and with radial distortion follows:

    double nx = (undistorted_position_x - cx) / fx;
    double ny = (undistorted_position_y - cy) / fy;
    double rr = nx*nx + ny*ny;
    double rrrr = rr*rr;
    double s = 1.0 + rr*k1 + rrrr*k2 + rrrr*rr*k3 + rrrr*rrrr*k4;
    nx = s*nx + p2*(rr + 2*nx*nx) + 2*p1*nx*ny;
    ny = s*ny + p1*(rr + 2*ny*ny) + 2*p2*nx*ny;
    double distorted_position_x = nx*fx + cx;
    double distorted_position_y = ny*fy + cy;

*Note that camea intrinsics are the same across different yaw_indices, so yaw_index is omitted in their filenames*


matterport_camera_poses 
---------------------

Camera pose files contain a 4x4 matrix that transforms column
vectors from camera to global coordinates.  Translation units are
in meters.  Global coordinate system is arbitrary, but the z-axis
generally points upward if the camera was level during capture.


matterport_mesh
---------------------
Textured mesh for the entire house.  The subdirectory contains a single .obj file, a single .mtl file, and textures in .jpg and .png format as referenced by the mtl file.


undistorted_camera_parameters
---------------------

An ascii file indicating the intrinsic and extrinsic camera parameters for every image.   Each line of the file is a separate command with the following format, where <n> is the number of scan commands (imaages) in the file, fx and fy are focal lengths in pixels, cx and cy specific the optical center of the image in pixels (not necessarily the center of the image), depth and color image filenames are relative to the depth and color directories, and camera-to-world-matrix is 16 values in row-major order providing a matrix that takes a canonical camera into a right-handed world coordinate system (the inverse of a typical extrinsics matrix).   The canonical camera has its eye at the origin, its view direction at (0,0-1), and its up direction at (0,1,0).   Note that all images have their origins in the bottom-left of the image.

    dataset matterport
    n_images <n>
    depth_directory undistorted_depth_images
    color_directory undistorted_color_images
    intrinsics_matrix <fx> 0 <cx>  0 <fy> <cy> 0 0 1
    scan <depth_image_filename> <color_image_filename> <camera-to-world-matrix>


undistorted_color_images 
---------------------

Tone-mapped color images after undistortion.   Radial distortion is removed, but (cx,cy) is not shifted to the center of the image.  Files are in JPG format.  Note that all images have their origins in the bottom-left of the image.


undistorted_depth_images 
---------------------

Depth images after undistortion.   Pixels of these depth images should (approximately) map to corresponding pixels in the undistorted_color_images.   The files are in 16-bit PNG format with the same scaling as matterport_depth_images (0.25mm per unit).  Note that all images have their origins in the bottom-left of the image.


undistorted_normal_images
---------------------

Normal, boundary, and radius maps estimated from the undistorted_depth_images.  

Normal maps are stored in three 16-bit PNG files (_nx.png, _ny.png, and _nz.png), where the integer values in the file are 32768*(1 + n), where n is a normal coordinate in range [-1,1].

Boundary maps are stored in a 16-bit PNG file (_boundary.png), where each pixel has one of the following values:
1 = on the border, 
2 = on a silhouette edge (boundary of an occluder), and
4 = on a shadow edge (boundary for occluded surface ... must be adjacent to a silhouette edge).

Radius maps are stored in a 16-bit PNG file (_radius.png), where integer values in the file are 4000 times the "radius" (average distance) to the pixel's neighbors in 3D.

Note that all images have their origins in the bottom-left of the image.


poisson_meshes
---------------------
Surface meshes reconstructed from the depth images using [Screened Poisson Surface Reconstruction](http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version9.01/).  

The files are binary PLY format.  xxx.ply provides a mesh that has been simplified to a reasonable polygon count (e.g., 500K) and thus provides a plausible trade-off between accuracy and size.  Additionally, three meshes are provided corresponding to different depth levels of the poisson surface reconstruction (e.g., xxx_9.ply corresponds to depth 9, xxx_10.ply corresponds to depth 10, etc.).   Higher depths correspond to finer resolution meshes.    

These meshes are generally lower resolution than the ones in house_segmentation/xxx.ply, which were generated by splitting the house into regions, reconstructing a mesh for each region, and then merging into one mesh.


cameras
---------------------
Camera extrinsics for manually chosen good view(s).

    exterior.cam - manually chosen camera viewpoints to view houses from a bird's eye view

Each .cam file has one line per camera with ascii numbers indicating the following camera parameters separated by spaces:

    vx vy vz  tx ty tz  ux uy uz  xfov yfov 1

where (vx, vy, vz) is the eye viewpoint of the camera, (tx, ty, tz) is the view direction, (ux, uy, uz) is the up direction, and xfov and yfov are the half-angles of the horizontal and vertical fields of view of the camera in radians (the angle from the central ray to the leftmost/bottommost ray in the field of view).


house_segmentations
---------------------
A manually specified decomposition of a house into levels, room-like regions, and objects with semantic labels.

The data for each house xxx is represented in four files:

    xxx.house = ascii file containing a list of regions, objects, etc.
        The .house file has a sequence of ascii lines with fields separated by spaces in the following format:
    
        H name label #images #panoramas #vertices #surfaces #segments #objects #categories #regions #portals #levels  0 0 0 0 0  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        L level_index #regions label  px py pz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        R region_index level_index 0 0 label  px py pz  xlo ylo zlo xhi yhi zhi  height  0 0 0 0
        P portal_index region0_index region1_index label  xlo ylo zlo xhi yhi zhi  0 0 0 0
        S surface_index region_index 0 label px py pz  nx ny nz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        V vertex_index surface_index label  px py pz  nx ny nz  0 0 0
        P name  panorama_index region_index 0  px py pz  0 0 0 0 0
        I image_index panorama_index  name camera_index yaw_index e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33  i00 i01 i02  i10 i11 i12 i20 i21 i22  width height  px py pz  0 0 0 0 0
        C category_index category_mapping_index category_mapping_name mpcat40_index mpcat40_name 0 0 0 0 0
        O object_index region_index category_index px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2 0 0 0 0 0 0 0 0 
        E segment_index object_index id area px py pz xlo ylo zlo xhi yhi zhi  0 0 0 0 0
       
        where xxx_index indicates the index of the xxx in the house file (starting at 0),
        #xxxs indicates how many xxxs will appear later in the file that back reference (associate) to this entry,
        (px,py,pz) is a representative position, (nx,ny,nz) is a normal direction,
        (xlo, ylo, zlo, xhi, yhi, zhi) is an axis-aligned bounding box,
        camera_index is in [0-5], yaw_index is in [0-2],
        (e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33) are the extrinsic matrix of a camera,
        (i00 i01 i02  i10 i11 i12 i20 i21 i22) are the intrinsic matrix for a camera,
        (px, py, pz, a0x, a0y, a0z, a1x, a1y, a1z, r0, r1, r2) define the center, axis directions, and radii of an oriented bounding box, 
        height is the distance from the floor, and 
        0 is a value that can be ignored.
    
        The extent of each region is defined by a prism with its vertical extent dictated by its height and
        its horizontal cross-section dictated by the counter-clockwise set of polygon vertices associated
        with each surface assocated with the region.
    
        The extent of each object is defined by the oriented bounding box of the 'O' command.
        The set of faces associated with each segment are ones whose 'face_material' field
        in the xxx.ply file (described next) matches the segment 'id' in the 'S' command.
    
    xxx.ply = 3D mesh in ply format.   In addition to the usual fields, there are three additional fields for each face:
        face_material = unique id of segment containing this face
        face_segment = unique id of object instance containing this face
        face_category = unique id of the category label for the object instance containing this face 
            (i.e., mapping to the "index" column of the category.tsv file)
        
    xxx.fsegs.json = JSON file indicating which segment contains each face of the mesh in regionX.ply, where
        segIndices = an array of unique segment IDs, one per face in the order it appears in the mesh 
               (i.e., the Kth entry provides the unique segment ID for the Kth face of the mesh)   
               
    xxx.semseg.json = JSON file containing an array of object instances with category labels, where
        segGroups = an array of object instances, each with the following fields
        label = string indicating the raw label provided by a human annotator for that object instance
                (this label maps to the "raw category" column of categories.tsv)
        segments = an array containing the unique ids for all segments in this object instance
                (the unique ids for segments map to ones found in regionX.fsegs.json)

The 'label' of each region is a string with the following conventions:

    'a' = bathroom (should have a toilet and a sink)
    'b' = bedroom
    'c' = closet
    'd' = dining room (includes “breakfast rooms” other rooms people mainly eat in)
    'e' = entryway/foyer/lobby (should be the front door, not any door)
    'f' = familyroom (should be a room that a family hangs out in, not any area with couches)
    'g' = garage
    'h' = hallway
    'i' = library (should be room like a library at a university, not an individual study)
    'j' = laundryroom/mudroom (place where people do laundry, etc.)
    'k' = kitchen
    'l' = living room (should be the main “showcase” living room in a house, not any area with couches)
    'm' = meetingroom/conferenceroom
    'n' = lounge (any area where people relax in comfy chairs/couches that is not the family room or living room
    'o' = office (usually for an individual, or a small set of people)
    'p' = porch/terrace/deck/driveway (must be outdoors on ground level)
    'r' = rec/game (should have recreational objects, like pool table, etc.)
    's' = stairs
    't' = toilet (should be a small room with ONLY a toilet)
    'u' = utilityroom/toolroom 
    'v' = tv (must have theater-style seating)
    'w' = workout/gym/exercise
    'x' = outdoor areas containing grass, plants, bushes, trees, etc.
    'y' = balcony (must be outside and must not be on ground floor)
    'z' = other room (it is clearly a room, but the function is not clear)
    'B' = bar
    'C' = classroom
    'D' = dining booth
    'S' = spa/sauna
    'Z' = junk (reflections of mirrors, random points floating in space, etc.)
    '-' = no label 

The label of each object is defined by its category index.   For each category, the .house file provides a category_mapping_index, category_mapping_name, mcat40_index, and mcat40_name.   The category_mapping_index maps into the first column of [metadata/category_mapping.tsv](metadata/category_mapping.tsv).  Further information these object categories (including how they map to WordNet synsets) can be
found in [metadata/category_mapping.tsv](metadata/category_mapping.tsv).  The mpcat40_index maps into the first column of [metadata/mpcat40.tsv](metadata/mpcat40.tsv), which provides further information about them (including a standard color to display for each one).


region_segmentations
---------------------
A set of manually specified segment, object instance, and semantic category labels for walls, floors, ceilings, doors, windows, and "furniture-sized" objects for each region of each house. 

The region_segmentations and labels are provided as annotations on 3D meshes.   A ply file provides the raw geometry for each region.   Json files indicate how each triangle of the mesh is associated with a "segment", how segments are associated with object instances, and, how object instances are associated with semantic categories as follows (this is the same as for house_segmentations):

    regionX.ply = 3D mesh in ply format.   In addition to the usual fields, there are three additional fields for each face:
        face_material = unique id of segment containing this face
        face_segment = unique id of object instance containing this face
        face_category = unique id of the category label for the object instance containing this face 
            (i.e., mapping to the "index" column of the category.tsv file)
        
    regionX.fsegs.json = JSON file indicating which segment contains each face of the mesh in regionX.ply, where
        segIndices = an array of unique segment IDs, one per face in the order it appears in the mesh 
               (i.e., the Kth entry provides the unique segment ID for the Kth face of the mesh)   
               
    regionX.semseg.json = JSON file containing an array of object instances with category labels, where
        segGroups = an array of object instances, each with the following fields
            label = string indicating the raw label provided by a human annotator for that object instance
                (this label maps to the "raw category" column of categories.tsv)
            segments = an array containing the unique ids for all segments in this object instance
                (the unique ids for segments map to ones found in regionX.fsegs.json)





---



# **数据组织**

主要数据集位于 "data" 目录下。每个房子都有一个单独的子目录，目录名由一个唯一的字符串命名（例如："1pXnuDYAj8r"）。在每个房子目录下，有针对不同类型数据的独立目录，如下所示：

Matterport 直接提供的数据：

- matterport_camera_intrinsics = Matterport 提供的相机内参
- matterport_camera_poses = Matterport 提供的相机外参
- matterport_color_images = Matterport 提供的彩色图像
- matterport_depth_images = Matterport 提供的深度图像
- matterport_hdr_images = Matterport 提供的 HDR 图像
- matterport_mesh = Matterport 提供的带纹理的网格模型
- matterport_skybox_images = Matterport 提供的天空盒图像

为方便未来研究而计算的数据：

- undistorted_camera_parameters = 去畸变后的相机内参和外参
- undistorted_color_images = 去畸变后的彩色图像
- undistorted_depth_images = 去畸变后的深度图像
- undistorted_normal_images = 与去畸变深度图像对齐的法线和边界图像
- poisson_meshes = 通过泊松网格重建得到的低分辨率网格模型

手动指定的标注：

- cameras = 手动选择的良好视角对应的相机外参
- house_segmentations = 手动指定的房屋语义分割（分割成区域）
- region_segmentations = 手动指定的区域语义分割（分割成物体）

以下是关于每个数据目录的详细信息。

## Matterport 图像的通用文件命名约定：

Matterport Pro 相机由三个 Primesense Carmine 深度和 RGB 传感器组成，安装在人眼高度的三脚架上。这三个相机（相机索引 0-2）分别向上倾斜、水平和向下倾斜，共同覆盖了大部分垂直视场。在拍摄过程中，相机会绕其垂直轴旋转，并在每隔 60 度（偏航角索引 0-5）处停止。因此，对于每个三脚架位置（panorama_uuid），总共会拍摄 18 张图像。与这些图像关联的文件遵循以下命名约定：

```
<panorama_uuid>_<imgtype><camera_index>_<yaw_index>.<extension>
```

其中 `<panorama_uuid>` 是一个唯一的字符串，`<camera_index>` 是 [0-2]，`<yaw_index>` 是 [0-5]。`<imgtype>` 对于 HDR 图像是 'j'，对于色调映射的彩色图像是 'i'，对于深度图像是 'd'，对于天空盒图像是 "skybox"，对于相机姿态文件是 "pose"，对于相机内参文件是 "intrinsics"。扩展名对于 HDR 图像是 ".jxr"，对于色调映射的彩色图像是 ".jpg"，对于深度和法线图像是 ".png"。

## matterport_hdr_images

原始 HDR 图像，格式为 jxr。

每个 RGB 通道提供一个 16 位的值 (jxr_val[i])，可以使用以下公式映射到强度值 (col_val[i])：

```
for (int i = 0; i < 3; i++) {
  if (jxr_val[i] <= 3000) col_val[i] = jxr_val[i] * 8e-8
  else col_val[i] = 0.00024 * 1.0002 ** (jxr_val[i] - 3000)
}

col_val[0] *= 0.8
col_val[1] *= 1.0
col_val[2] *= 1.6
```

## matterport_color_images

色调映射的彩色图像，格式为 jpg。

## matterport_depth_images

重新投影的深度图像，与彩色图像对齐。每个深度图像都是一个 16 位的 PNG 文件，包含像素点在 z 方向上到相机中心的距离（*不是*到相机中心的欧几里得距离），每个值代表 0.25 毫米（除以 4000 得到米）。零值表示“没有读数”。

## matterport_camera_intrinsics

每个相机的内参，以 ASCII 格式存储，格式如下（使用 OpenCV 的表示法）：

```
width height fx fy cx cy k1 k2 p1 p2 k3
```

Matterport 相机的坐标轴约定是：

x 轴：向右

y 轴：向下

z 轴：“朝向”

图像原点位于图像的左上角。

以下是将坐标从无径向畸变映射到有径向畸变的示例代码：

```
double nx = (undistorted_position_x - cx) / fx;
double ny = (undistorted_position_y - cy) / fy;
double rr = nx*nx + ny*ny;
double rrrr = rr*rr;
double s = 1.0 + rr*k1 + rrrr*k2 + rrrr*rr*k3 + rrrr*rrrr*k4;
nx = s*nx + p2*(rr + 2*nx*nx) + 2*p1*nx*ny;
ny = s*ny + p1*(rr + 2*ny*ny) + 2*p2*nx*ny;
double distorted_position_x = nx*fx + cx;
double distorted_position_y = ny*fy + cy;
```

*请注意，相机内参在不同的偏航角索引下是相同的，因此文件名中省略了偏航角索引。*

## matterport_camera_poses

相机姿态文件包含一个 4x4 的矩阵，用于将列向量从相机坐标系转换到全局坐标系。平移单位是米。全局坐标系是任意的，但如果相机在拍摄过程中保持水平，则 z 轴通常指向上方。

## matterport_mesh

整个房子的带纹理的网格模型。该子目录包含一个 `.obj` 文件，一个 `.mtl` 文件，以及 `.mtl` 文件引用的 `.jpg` 和 `.png` 格式的纹理。

## undistorted_camera_parameters

一个 ASCII 文件，指示每个图像的相机内参和外参。文件中的每一行都是一个单独的命令，格式如下：其中 `<n>` 是文件中的扫描命令（图像）数量，fx 和 fy 是以像素为单位的焦距，cx 和 cy 指定图像的光学中心（以像素为单位，不一定位于图像中心），深度和彩色图像文件名是相对于深度和彩色目录的路径，camera-to-world-matrix 是 16 个按行主序排列的值，提供一个将规范相机坐标系转换为右手世界坐标系的矩阵（是典型外参矩阵的逆矩阵）。规范相机的原点位于其视点，视线方向为 (0,0,-1)，向上方向为 (0,1,0)。请注意，所有图像的原点都位于图像的左下角。

```
dataset matterport
n_images <n>
depth_directory undistorted_depth_images
color_directory undistorted_color_images
intrinsics_matrix <fx> 0 <cx>  0 <fy> <cy> 0 0 1
scan <depth_image_filename> <color_image_filename> <camera-to-world-matrix>
```

## undistorted_color_images

去畸变后的色调映射彩色图像。径向畸变已被移除，但 (cx,cy) 未移动到图像中心。文件格式为 JPG。请注意，所有图像的原点都位于图像的左下角。

## undistorted_depth_images

去畸变后的深度图像。这些深度图像的像素应该（近似地）映射到 `undistorted_color_images` 中的对应像素。文件格式为 16 位 PNG，具有与 `matterport_depth_images` 相同的比例（每单位 0.25 毫米）。请注意，所有图像的原点都位于图像的左下角。

## undistorted_normal_images

从去畸变深度图像估计的法线图、边界图和半径图。

法线图存储在三个 16 位 PNG 文件中（`_nx.png`、`_ny.png` 和 `_nz.png`），文件中的整数值是 `32768*(1 + n)`，其中 n 是范围在 [-1,1] 的法线坐标。

边界图存储在一个 16 位 PNG 文件中（_boundary.png），其中每个像素具有以下值之一：

1 = 在边界上，

2 = 在轮廓边缘（遮挡物的边界）上，并且

4 = 在阴影边缘（被遮挡表面的边界...必须与轮廓边缘相邻）上。

半径图存储在一个 16 位 PNG 文件中（`_radius.png`），文件中的整数值是像素点到其 3D 邻居的“半径”（平均距离）的 4000 倍。

请注意，所有图像的原点都位于图像的左下角。

## poisson_meshes

使用 [Screened Poisson Surface Reconstruction](http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version9.01/) 从深度图像重建的表面网格模型。

文件格式为二进制 PLY。`xxx.ply` 提供了一个简化到合理多边形数量（例如，50 万）的网格模型，从而在精度和大小之间取得了合理的平衡。此外，还提供了三个对应于泊松表面重建不同深度级别的网格模型（例如，`xxx_9.ply` 对应于深度 9，`xxx_10.ply` 对应于深度 10，等等）。更高的深度对应于更高分辨率的网格模型。

这些网格模型通常比 `house_segmentation/xxx.ply` 中的网格模型分辨率低，后者是通过将房屋分割成区域，为每个区域重建网格，然后合并成一个网格生成的。

## cameras

手动选择的良好视角的相机外参。

`exterior.cam` - 手动选择的用于从鸟瞰视角查看房屋的相机视点。

每个 `.cam` 文件每行包含一个相机的参数，参数之间用空格分隔，格式如下：

```
vx vy vz  tx ty tz  ux uy uz  xfov yfov 1
```

其中 (vx, vy, vz) 是相机的视点，(tx, ty, tz) 是视线方向，(ux, uy, uz) 是向上方向，xfov 和 yfov 是相机水平和垂直视场的一半角度（以弧度为单位，是从中心光线到视场最左/最下光线的角度）。

## house_segmentations

手动指定的房屋分解，包括楼层、类似房间的区域以及带有语义标签的物体。

每个房子 xxx 的数据由四个文件表示：

```
xxx.house = 包含区域、物体等列表的 ASCII 文件。
.house 文件包含一系列 ASCII 行，字段之间用空格分隔，格式如下：

H name label #images #panoramas #vertices #surfaces #segments #objects #categories #regions #portals #levels  0 0 0 0 0  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
L level_index #regions label  px py pz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
R region_index level_index 0 0 label  px py pz  xlo ylo zlo xhi yhi zhi  height  0 0 0 0
P portal_index region0_index region1_index label  xlo ylo zlo xhi yhi zhi  0 0 0 0
S surface_index region_index 0 label px py pz  nx ny nz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
V vertex_index surface_index label  px py pz  nx ny nz  0 0 0
P name  panorama_index region_index 0  px py pz  0 0 0 0 0
I image_index panorama_index  name camera_index yaw_index e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33  i00 i01 i02  i10 i11 i12 i20 i21 i22  width height  px py pz  0 0 0 0 0
C category_index category_mapping_index category_mapping_name mpcat40_index mpcat40_name 0 0 0 0 0
O object_index region_index category_index px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2 0 0 0 0 0 0 0 0
E segment_index object_index id area px py pz xlo ylo zlo xhi yhi zhi  0 0 0 0 0

其中 xxx_index 指示 xxx 在房屋文件中的索引（从 0 开始），
#xxxs 指示文件中稍后将出现多少个 xxx 会反向引用（关联）到此条目，
(px,py,pz) 是一个代表性位置，(nx,ny,nz) 是一个法线方向，
(xlo, ylo, zlo, xhi, yhi, zhi) 是一个轴对齐的边界框，
camera_index 的取值范围是 [0-5]，yaw_index 的取值范围是 [0-2]，
(e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33) 是相机的外参矩阵，
(i00 i01 i02  i10 i11 i12 i20 i21 i22) 是相机的内参矩阵，
(px, py, pz, a0x, a0y, a0z, a1x, a1y, a1z, r0, r1, r2) 定义了一个有向边界框的中心、轴方向和半径，
height 是到地板的距离，并且
0 是一个可以忽略的值。

每个区域的范围由一个棱柱定义，其垂直范围由其高度决定，水平横截面由与该区域关联的每个表面相关联的逆时针多边形顶点集决定。

每个物体的范围由 'O' 命令的有向边界框定义。
与每个片段关联的面是那些在 `xxx.ply` 文件（接下来描述）中 'face_material' 字段与 'S' 命令中的片段 'id' 匹配的面。

xxx.ply = PLY 格式的 3D 网格模型。
除了通常的字段外，每个面还有三个额外的字段：
    face_material = 包含此面的片段的唯一 ID
    face_segment = 包含此面的物体实例的唯一 ID
    face_category = 包含此面的物体实例的类别标签的唯一 ID
        （即，映射到 `category.tsv` 文件的 "index" 列）

xxx.fsegs.json = JSON 文件，指示 `regionX.ply` 中网格的每个面包含在哪个片段中，其中
    segIndices = 一个唯一的片段 ID 数组，每个面对应一个 ID，顺序与它们在网格中出现的顺序相同
        （即，第 K 个条目提供网格中第 K 个面的唯一片段 ID）

xxx.semseg.json = JSON 文件，包含一个带有类别标签的物体实例数组，其中
    segGroups = 一个物体实例数组，每个实例都包含以下字段
        label = 字符串，指示人工标注者为该物体实例提供的原始标签
            （此标签映射到 `categories.tsv` 文件的 "raw category" 列）
        segments = 一个包含此物体实例中所有片段的唯一 ID 的数组
            （片段的唯一 ID 映射到 `regionX.fsegs.json` 中找到的 ID）
```

每个区域的 'label' 是一个字符串，遵循以下约定：

```
'a' = 浴室（应包含马桶和水槽）
'b' = 卧室
'c' = 衣橱
'd' = 餐厅（包括“早餐室”和其他主要用餐的房间）
'e' = 入口/门厅/大堂（应该是前门，而不是任何门）
'f' = 家庭活动室（应该是家人常待的房间，而不是任何有沙发的区域）
'g' = 车库
'h' = 走廊
'i' = 图书馆（应该是像大学图书馆一样的房间，而不是个人书房）
'j' = 洗衣房/杂物间（人们洗衣服等的地方）
'k' = 厨房
'l' = 客厅（应该是房屋主要的“展示”客厅，而不是任何有沙发的区域）
'm' = 会议室
'n' = 休息室（任何人们在舒适的椅子/沙发上放松的区域，不是家庭活动室或客厅）
'o' = 办公室（通常供个人或少数人使用）
'p' = 门廊/露台/甲板/车道（必须是地面上的室外区域）
'r' = 娱乐/游戏室（应包含娱乐设施，如台球桌等）
's' = 楼梯
't' = 厕所（应该是一个只有马桶的小房间）
'u' = 杂物间/工具间
'v' = 电视房（必须有剧院式座位）
'w' = 健身房/锻炼室
'x' = 包含草、植物、灌木、树木等的室外区域
'y' = 阳台（必须是室外的，且不在一楼）
'z' = 其他房间（很明显是一个房间，但功能不明确）
'B' = 酒吧
'C' = 教室
'D' = 卡座
'S' = 水疗/桑拿
'Z' = 垃圾（镜子反射、空间中漂浮的随机点等）
'-' = 无标签
```

每个物体的标签由其类别索引定义。对于每个类别，`.house` 文件提供了 category_mapping_index、category_mapping_name、mcat40_index 和 mcat40_name。category_mapping_index 映射到 metadata/category_mapping.tsv 的第一列。关于这些物体类别的更多信息（包括它们如何映射到 WordNet 同义词集）可以在 metadata/category_mapping.tsv 中找到。mpcat40_index 映射到 metadata/mpcat40.tsv 的第一列，该文件提供了关于它们的更多信息（包括每种类别要显示的标准颜色）。

## region_segmentations

一组手动指定的片段、物体实例和语义类别标签，用于每个房屋每个区域的墙壁、地板、天花板、门、窗户和“家具大小”的物体。

区域分割和标签作为 3D 网格模型的标注提供。一个 PLY 文件提供了每个区域的原始几何信息。JSON 文件指示网格的每个三角形如何与一个“片段”关联，片段如何与物体实例关联，以及物体实例如何与语义类别关联，如下所示（与房屋分割相同）：

```
regionX.ply = PLY 格式的 3D 网格模型。
除了通常的字段外，每个面还有三个额外的字段：
    face_material = 包含此面的片段的唯一 ID
    face_segment = 包含此面的物体实例的唯一 ID
    face_category = 包含此面的物体实例的类别标签的唯一 ID
        （即，映射到 `category.tsv` 文件的 "index" 列）

regionX.fsegs.json = JSON 文件，指示 `regionX.ply` 中网格的每个面包含在哪个片段中，其中
    segIndices = 一个唯一的片段 ID 数组，每个面对应一个 ID，顺序与它们在网格中出现的顺序相同
        （即，第 K 个条目提供网格中第 K 个面的唯一片段 ID）

regionX.semseg.json = JSON 文件，包含一个带有类别标签的物体实例数组，其中
    segGroups = 一个物体实例数组，每个实例都包含以下字段
        label = 字符串，指示人工标注者为该物体实例提供的原始标签
            （此标签映射到 `categories.tsv` 文件的 "raw category" 列）
        segments = 一个包含此物体实例中所有片段的唯一 ID 的数组
            （片段的唯一 ID 映射到 `regionX.fsegs.json` 中找到的 ID）
```
