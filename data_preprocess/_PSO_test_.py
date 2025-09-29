import numpy as np
import numba
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max


class Config:
    def __init__(self, **kwargs):
        self.IMAGE_PATH = 'test1.png'
        try:
            image = cv2.imread(self.IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
            _, self.binary_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            # image = image.resize((256, 256))
            self.img = np.array(image)
            self.width, self.height = image.shape[1], image.shape[0]
        except FileNotFoundError:
            raise FileNotFoundError(f"[Error]: Image file not found at {self.IMAGE_PATH}")

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

# --- 初始化和求解电势 ---
config = Config()
foreground_mask = (config.binary_img == 0)
potential = np.zeros((config.height, config.width), dtype=np.float64)
potential[foreground_mask] = 1.0
print("开始求解电势分布 (使用 Numba-JIT SOR)...")
start_time = time.time()
solve_potential_sor(potential, foreground_mask, omega=1.9, max_iter=5000)
end_time = time.time()
print(f"求解耗时: {end_time - start_time:.4f} 秒")
# --- 1. 计算电场强度 ---
print("计算电场强度...")
Ey, Ex = np.gradient(-potential)
Ey = -Ey
magnitude = np.sqrt(Ex**2 + Ey**2)
magnitude[foreground_mask] = 0
# --- 2. 寻找电场强度局部极大值 ---
print("寻找局部极大值...")
min_dist = int(0.01*min(config.width, config.height)) # 两个特征点之间的最小距离
threshold_abs = np.percentile(magnitude[magnitude > 0], 95)
coordinates_skimage = peak_local_max(
    magnitude,
    min_distance=min_dist,      # min_distance: 两个峰值之间的最小像素距离，用于抑制密集的峰值
    threshold_abs=threshold_abs,        # threshold_abs: 峰值的绝对最小强度
    exclude_border=False, # 不排除边界上的点
    threshold_rel=0.25       # 相对于最大值的相对最小强度
)
# --- 3. 可视化结果 ---
print("生成可视化图像...")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# a. 原始图像和找到的特征点
ax = axes[0]
ax.imshow(config.img, cmap='gray')
ax.scatter(coordinates_skimage[:, 1], coordinates_skimage[:, 0], c='red', s=40, marker='o', label='特征点')
ax.set_title('原始图像与找到的特征点')
ax.legend()
ax.axis('off')
# b. 电势分布
ax = axes[1]
im = ax.imshow(potential, cmap='viridis')
ax.set_title(r'电势 (Potential $\phi$)')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.axis('off')
# c. 电场强度热力图
ax = axes[2]

# 使用对数尺度可以更好地观察场强分布
im = ax.imshow(magnitude, cmap='inferno', norm=LogNorm())
ax.set_title('电场强度 |E| (对数尺度)')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.axis('off')
plt.tight_layout()
plt.savefig("feature_points_detection.png", dpi=150)
print("可视化结果已保存到 feature_points_detection.png")
plt.show()


