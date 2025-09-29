import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import distance_transform_edt
from PIL import Image
import random


# --- 超参数与配置 (Hyperparameters & Configuration) ---
class Config:
    # 文件与模拟控制
    IMAGE_PATH = 'test.png'
    MAX_PARTICLES = 150  # 粒子数量上限
    SIMULATION_STEPS = 500  # 模拟总帧数
    DT = 1.0  # 时间步长 (Delta Time)

    # 粒子物理属性
    PARTICLE_MASS = 1.0  # 粒子质量
    MAX_LIFE = 100.0  # 粒子生命值

    # 力模型权重
    W_ATTRACTION = 0.08  # 前景吸引力权重
    W_REPLUSION = 150.0  # 粒子间排斥力权重
    W_SPRING = 0.1  # 连杆弹簧力权重
    W_DAMPING = 0.05  # 前景阻尼权重

    # 生命周期规则
    LIFE_RECOVERY_RATE = 2.0  # 在前景上的生命恢复速率
    LIFE_DECAY_RATE = 0.5  # 在背景中的生命衰减速率
    PROLIFERATION_THRESHOLD_DISTANCE = 30.0  # 连杆分裂的距离阈值
    BG_RATIO_THRESHOLD = 0.5  # 连杆分裂的背景像素比例阈值
    STABLE_TIME_THRESHOLD = 15  # 连杆分裂所需的稳定时间
    SEED_PROLIFERATION_VEL_THRESHOLD = 0.1  # 种子增殖的速度阈值

    # 其他
    IDEAL_SPRING_LENGTH = 15.0  # 连杆理想长度
    REPULSION_RADIUS = 50.0  # 粒子排斥力作用半径


class Particle:
    """定义单个粒子的数据结构"""

    def __init__(self, x, y, world_shape):
        self.id = id(self)
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.force = np.zeros(2, dtype=float)
        self.life = Config.MAX_LIFE / 2
        self.neighbors = set()
        self.time_on_foreground = 0
        self.world_shape = world_shape

    def update_state(self, on_foreground):
        """更新生命值和前景停留时间"""
        if on_foreground:
            self.life = min(Config.MAX_LIFE, self.life + Config.LIFE_RECOVERY_RATE)
            self.time_on_foreground += 1
        else:
            self.life -= Config.LIFE_DECAY_RATE
            self.time_on_foreground = 0

    def apply_force(self):
        """根据合力更新速度和位置"""
        acceleration = self.force / Config.PARTICLE_MASS
        self.vel += acceleration * Config.DT
        self.pos += self.vel * Config.DT

        # 边界约束
        self.pos[0] = np.clip(self.pos[0], 0, self.world_shape[1] - 1)
        self.pos[1] = np.clip(self.pos[1], 0, self.world_shape[0] - 1)

        # 重置力以便下一帧计算
        self.force = np.zeros(2, dtype=float)


class ParticleSystem:
    """管理整个粒子系统的模拟、生命周期和渲染"""

    def __init__(self, image_path):
        self.config = Config()
        self.particles = {}
        self.next_particle_id = 0
        self.preprocess_image(image_path)
        self.seed_initial_particles(10)

    def preprocess_image(self, image_path):
        """加载图像并进行预处理"""
        print("Preprocessing image...")
        try:
            image = Image.open(image_path).convert('L')
            self.image_shape = image.size
            # NumPy数组中，(height, width) -> (y, x)
            img_array = np.array(image)
            # 前景为1 (黑), 背景为0 (白)
            self.foreground_mask = (img_array < 128).astype(int)
            # 计算距离场 (phi)
            self.dist_transform = distance_transform_edt(self.foreground_mask == 0)
            # 计算梯度场 (nabla_phi)
            gy, gx = np.gradient(self.dist_transform)
            self.gradient_field = np.stack([gx, gy], axis=-1)
            print("Preprocessing complete.")
        except FileNotFoundError:
            print(f"Error: Image file not found at '{image_path}'")
            exit()

    def add_particle(self, x, y):
        """向系统中添加一个新粒子"""
        if len(self.particles) >= self.config.MAX_PARTICLES:
            return None
        p = Particle(x, y, (self.image_shape[1], self.image_shape[0]))
        self.particles[p.id] = p
        return p

    def seed_initial_particles(self, num_particles):
        """在前景附近播撒初始粒子"""
        for _ in range(num_particles):
            # 在前景像素点上随机选择位置
            fg_pixels = np.argwhere(self.foreground_mask > 0)
            if len(fg_pixels) > 0:
                py, px = random.choice(fg_pixels)
                self.add_particle(px, py)

    def is_on_foreground(self, pos):
        """检查一个位置是否在前景上"""
        x, y = int(pos[0]), int(pos[1])
        if 0 <= y < self.foreground_mask.shape[0] and 0 <= x < self.foreground_mask.shape[1]:
            return self.foreground_mask[y, x] == 1
        return False

    def update(self):
        """执行一个模拟步长"""
        if not self.particles:
            return
        # 1. 计算所有力
        self.calculate_forces()
        # 2. 更新粒子状态 (位置、速度、生命)
        for p in list(self.particles.values()):
            on_fg = self.is_on_foreground(p.pos)
            p.update_state(on_fg)
            p.apply_force()
        # 3. 处理生命周期 (死亡与增殖)
        self.handle_lifecycle()

    def calculate_forces(self):
        """为每个粒子计算合力"""
        all_particles = list(self.particles.values())
        for i, p in enumerate(all_particles):
            # 前景吸引力 (外部电场, 内部屏蔽)
            on_fg = self.is_on_foreground(p.pos)
            if not on_fg:
                grad_x, grad_y = self.gradient_field[int(p.pos[1]), int(p.pos[0])]
                f_attraction = -self.config.W_ATTRACTION * np.array([grad_x, grad_y])
                p.force += f_attraction
            # 前景阻尼力
            else:
                f_damping = -self.config.W_DAMPING * p.vel
                p.force += f_damping

            # 粒子间排斥力
            for j in range(i + 1, len(all_particles)):
                p2 = all_particles[j]
                vec = p.pos - p2.pos
                dist_sq = np.sum(vec ** 2)
                if 0 < dist_sq < self.config.REPULSION_RADIUS ** 2:
                    dist = np.sqrt(dist_sq)
                    direction = vec / dist
                    # 力的大小与距离平方成反比
                    magnitude = self.config.W_REPLUSION / dist_sq
                    f_repulsion = magnitude * direction
                    p.force += f_repulsion
                    p2.force -= f_repulsion  # 牛顿第三定律

            # 连杆弹簧力
            # for neighbor_id in p.neighbors:
            #     if neighbor_id in self.particles:
            #         p2 = self.particles[neighbor_id]
            #         vec = p2.pos - p.pos
            #         dist = np.linalg.norm(vec)
            #         if dist > 0:
            #             direction = vec / dist
            #             # 胡克定律
            #             magnitude = self.config.W_SPRING * (dist - self.config.IDEAL_SPRING_LENGTH)
            #             f_spring = magnitude * direction
            #             p.force += f_spring

    def handle_lifecycle(self):
        """处理粒子的死亡和增殖"""
        # 死亡
        dead_particles = [pid for pid, p in self.particles.items() if p.life <= 0]
        for pid in dead_particles:
            p = self.particles.pop(pid)
            for neighbor_id in p.neighbors:
                if neighbor_id in self.particles:
                    self.particles[neighbor_id].neighbors.discard(pid)

        # 增殖
        # 规则1: 中点增殖
        links_to_split = []
        processed_links = set()
        for pid, p in self.particles.items():
            for neighbor_id in p.neighbors:
                if (pid, neighbor_id) in processed_links or (neighbor_id, pid) in processed_links:
                    continue

                p2 = self.particles[neighbor_id]
                dist = np.linalg.norm(p.pos - p2.pos)

                if dist > self.config.PROLIFERATION_THRESHOLD_DISTANCE and \
                        p.time_on_foreground > self.config.STABLE_TIME_THRESHOLD and \
                        p2.time_on_foreground > self.config.STABLE_TIME_THRESHOLD:

                    # 检查连杆路径上的背景像素比例
                    num_samples = int(dist / 5) + 1
                    bg_count = 0
                    for k in range(1, num_samples):
                        sample_pos = p.pos + k / num_samples * (p2.pos - p.pos)
                        if not self.is_on_foreground(sample_pos):
                            bg_count += 1

                    if bg_count / num_samples > self.config.BG_RATIO_THRESHOLD:
                        links_to_split.append((pid, neighbor_id))

                processed_links.add((pid, neighbor_id))

        for pid1, pid2 in links_to_split:
            if pid1 in self.particles and pid2 in self.particles:
                p1 = self.particles[pid1]
                p2 = self.particles[pid2]

                # 创建新粒子
                mid_pos = (p1.pos + p2.pos) / 2
                new_p = self.add_particle(mid_pos[0], mid_pos[1])
                if new_p:
                    new_p.vel = (p1.vel + p2.vel) / 2
                    new_p.life = (p1.life + p2.life) / 2

                    # 更新网络拓扑
                    p1.neighbors.remove(pid2)
                    p2.neighbors.remove(pid1)
                    p1.neighbors.add(new_p.id)
                    p2.neighbors.add(new_p.id)
                    new_p.neighbors.add(pid1)
                    new_p.neighbors.add(pid2)

        # 规则2: 孤立点增殖
        for p in list(self.particles.values()):
            if len(p.neighbors) == 0 and \
                    self.is_on_foreground(p.pos) and \
                    np.linalg.norm(p.vel) < self.config.SEED_PROLIFERATION_VEL_THRESHOLD:

                new_pos = p.pos + np.random.rand(2) * 5 - 2.5  # 在附近随机位置
                new_p = self.add_particle(new_pos[0], new_pos[1])
                if new_p:
                    p.neighbors.add(new_p.id)
                    new_p.neighbors.add(p.id)
                    p.vel = np.zeros(2)
                    new_p.vel = np.zeros(2)


# --- 可视化 ---
fig, ax = plt.subplots(figsize=(8, 8))
system = ParticleSystem(Config.IMAGE_PATH)


def animate(frame):
    ax.clear()

    # 1. 更新系统
    system.update()

    # 2. 绘制背景
    ax.imshow(system.foreground_mask, cmap='gray_r', extent=(0, system.image_shape[0], system.image_shape[1], 0))

    # 3. 绘制粒子和连杆
    if system.particles:
        positions = np.array([p.pos for p in system.particles.values()])
        colors = [p.life / Config.MAX_LIFE for p in system.particles.values()]

        # 绘制连杆
        for pid, p in system.particles.items():
            for neighbor_id in p.neighbors:
                if neighbor_id > pid and neighbor_id in system.particles:  # 避免重复绘制
                    p2 = system.particles[neighbor_id]
                    ax.plot([p.pos[0], p2.pos[0]], [p.pos[1], p2.pos[1]], 'b-', lw=1.5, alpha=0.7)

        # 绘制粒子
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, cmap='viridis', s=40, zorder=3, vmin=0, vmax=1)

    ax.set_title(f"Frame: {frame + 1}/{Config.SIMULATION_STEPS}, Particles: {len(system.particles)}")
    ax.set_aspect('equal')
    ax.set_xlim(0, system.image_shape[0])
    ax.set_ylim(system.image_shape[1], 0)  # y轴反转以匹配图像坐标


ani = FuncAnimation(fig, animate, frames=Config.SIMULATION_STEPS, interval=30, repeat=False)
plt.tight_layout()
plt.show()

# 如果你想将动画保存为gif
print("Saving animation... (This may take a while)")
ani.save('particle_system.gif', writer='imagemagick', fps=30)
print("Animation saved as particle_system.gif")