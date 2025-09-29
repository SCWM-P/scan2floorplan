# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from scipy.sparse import diags
# from scipy.sparse.linalg import spsolve

# # # 64×64 示例：两个“黑色金属块”
# h, w = 64, 64
# mask = np.zeros((h, w), dtype=bool)
# mask[10:20, 15:25] = True       # 块1
# mask[40:50, 35:45] = True       # 块2
# rho = np.where(mask, -1.0, 0.0) # -1 代表负电荷
# # IMAGE_PATH = "test0.png"
# # image = Image.open(IMAGE_PATH).convert('L')
# # rho = np.array(image).astype(int)
# # rho[rho < 255] = -1
# # rho[rho == 255] = 0
# # print(rho)
# # h, w = rho.shape

# def laplacian_matrix(n, m):
#     # 构造 A = (n*m)×(n*m) 五点差分矩阵
#     N = n * m
#     main = -4 * np.ones(N)
#     side1 = np.ones(N-1)
#     side1[m-1::m] = 0      # 断开每行末尾
#     side2 = np.ones(N-m)
#     diags_list = [main, side1, side1, side2, side2]
#     offsets = [0, -1, 1, -m, m]
#     return diags(diags_list, offsets, format='csr')

# A = laplacian_matrix(h, w)
# b = -rho.ravel()
# phi = spsolve(A, b).reshape(h, w)
# quiver_skip = min(h, w) // 15

# # 中心差分
# Ex = - np.gradient(phi, axis=1)
# Ey = - np.gradient(phi, axis=0)
# # Ex = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / 2.0
# # Ey = -(np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / 2.0

# plt.figure(figsize=(12,4))
# plt.subplot(1,3,1); plt.title('Charge ρ')
# plt.imshow(rho, cmap='seismic'); plt.colorbar()

# plt.subplot(1,3,2); plt.title('Potential φ')
# plt.imshow(phi, cmap='viridis'); plt.colorbar()

# plt.subplot(1,3,3); plt.title('Electric field')
# plt.imshow(np.sqrt(Ex**2 + Ey**2), cmap='magma'); plt.colorbar()
# # 叠加矢量箭头
# Y, X = np.mgrid[0:h:quiver_skip, 0:w:quiver_skip]
# plt.quiver(X, Y, Ex[::quiver_skip, ::quiver_skip], Ey[::quiver_skip, ::quiver_skip], color='white')
# plt.show()



import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numba
import time
import argparse

@numba.jit(nopython=True, fastmath=True)
def solve_potential_sor(potential, is_charge, omega=1.9, max_iter=5000, tol=1e-4):
    """
    使用 Numba 加速的 SOR 方法求解电势。
    Args:
        potential (np.array): 初始化后的电势网格。
        is_charge (np.array): 布尔数组，标记电荷位置。
        omega (float): 超松弛因子 (1 < omega < 2)。
        max_iter (int): 最大迭代次数。
        tol (float): 收敛容差。
    Returns:
        int: 实际迭代次数。
    """
    H, W = potential.shape
    for iteration in range(max_iter):
        max_diff = 0.0
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if not is_charge[i, j]:
                    old_val = potential[i, j]
                    # 计算邻居电势的平均值
                    avg_neighbors = (
                            potential[i-1, j] + potential[i+1, j] +
                            potential[i, j-1] + potential[i, j+1]
                    ) / 4.0
                    # SOR 更新规则
                    new_val = (1 - omega) * old_val + omega * avg_neighbors
                    potential[i, j] = new_val
                    diff = abs(new_val - old_val)
                    if diff > max_diff:
                        max_diff = diff
        # 检查是否收敛
        if max_diff < tol:
            print(f"在 {iteration+1} 次迭代后收敛。")
            return iteration + 1
    print(f"达到最大迭代次数 {max_iter}，未完全收敛。")
    return max_iter

def solve_and_visualize_field(image_path, omega=1.9, max_iter=5000):
    """
    加载图像，求解电场，并进行可视化。
    """
    # --- 1. 加载和预处理图像 ---
    print(f"加载图像: {image_path}")
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"无法找到或打开图像: {image_path}")
        
    # 将图像二值化：0为前景（电荷），255为背景
    # h, w = 64, 64
    # mask = np.zeros((h, w), dtype=bool)
    # mask[10:20, 15:25] = True       # 块1
    # mask[40:50, 35:45] = True       # 块2
    # rho = np.where(mask, 0.0, 255.0) # -1 代表负电荷
    _, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # binary_img = rho
    # --- 2. 设置边界条件 ---
    H, W = binary_img.shape
    potential = np.zeros((H, W), dtype=np.float64)
    is_charge = (binary_img == 0)
    
    # 前景电荷电势设为 1.0，边界电势为 0 (已是默认值)
    potential[is_charge] = 1.0
    # --- 3. 求解电势分布 (SOR) ---
    print("开始求解电势分布 (使用 Numba-JIT SOR)...")
    start_time = time.time()
    solve_potential_sor(potential, is_charge, omega=omega, max_iter=max_iter)
    end_time = time.time()
    print(f"求解耗时: {end_time - start_time:.4f} 秒")

    # --- 4. 计算电场 (E = -∇V) ---
    Ey, Ex = np.gradient(-potential)
    Ey = -Ey
    # --- 5. 计算电场强度 ---
    magnitude = np.sqrt(Ex**2 + Ey**2)
    # 在电荷内部场强无定义，设为0以便于可视化
    magnitude[is_charge] = 0

    # --- 6. 可视化 ---
    print("生成可视化图像...")
    # 设置matplotlib以支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'"{image_path}" 的电场模拟', fontsize=18)

    # a. 电势分布图
    ax = axes[0]
    im = ax.imshow(potential, cmap='viridis', origin='upper')
    ax.contour(is_charge, levels=[0.5], colors='white', linewidths=0.8)
    ax.set_title('电势分布 (Potential)', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    # b. 电场强度热力图
    ax = axes[1]
    # 使用对数尺度来更好地显示弱场区域
    im = ax.imshow(magnitude, cmap='inferno', origin='upper', 
                   norm=LogNorm(vmin=magnitude[magnitude>0].min(), vmax=magnitude.max()))
    ax.contour(is_charge, levels=[0.5], colors='white', linewidths=0.8)
    ax.set_title('电场强度 |E| (对数热力图)', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    # c. 电场箭头图
    ax = axes[2]
    # 在热力图上叠加箭头图
    ax.imshow(magnitude, cmap='inferno', origin='upper',
              norm=LogNorm(vmin=magnitude[magnitude>0].min(),vmax=magnitude.max()))
    
    # 降低箭头密度以便观察
    skip = max(1, H // 30) # 每隔 skip 个像素画一个箭头
    Y, X = np.mgrid[0:H:skip, 0:W:skip]
    Ex_skip = Ex[Y, X]
    Ey_skip = Ey[Y, X]
    
    # 归一化箭头长度，使其只表示方向
    norm = np.sqrt(Ex_skip**2 + Ey_skip**2)
    # 防止除以零
    nz = norm > 0
    Ex_norm, Ey_norm = np.zeros_like(Ex_skip), np.zeros_like(Ey_skip)
    Ex_norm[nz] = Ex_skip[nz] / norm[nz]
    Ey_norm[nz] = Ey_skip[nz] / norm[nz]
    
    # 在非电荷区域绘制箭头
    charge_mask_skip = is_charge[Y, X]
    ax.quiver(X[~charge_mask_skip], Y[~charge_mask_skip], 
              Ex_norm[~charge_mask_skip], Ey_norm[~charge_mask_skip],
              color='cyan', scale=40, width=0.003, headwidth=4)
    
    ax.contour(is_charge, levels=[0.5], colors='white', linewidths=0.8)
    ax.set_title('电场方向矢量图 (Arrow Plot)', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("electric_field_visualization.png", dpi=300)
    print("可视化结果已保存到 'electric_field_visualization.png'")
    plt.show()


if __name__ == '__main__':
    # 使用 argparse 来处理命令行参数
    parser = argparse.ArgumentParser(description="求解并可视化二值图的电场分布。")
    parser.add_argument('--image_path', type=str, default="test.png", help="输入二值图的路径 (例如: test.png)")
    parser.add_argument('--omega', type=float, default=1.9, help="SOR方法的超松弛因子 (1-2之间, 推荐1.8-1.95)")
    parser.add_argument('--iter', type=int, default=10000, help="最大迭代次数")
    args = parser.parse_args()
    
    solve_and_visualize_field(args.image_path, omega=args.omega, max_iter=args.iter)