import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import warnings
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
import itertools  # 用于生成粒子ID

# --- 你的 Config 和 Particle 类 (只修正了 move 方法) ---
random.seed(2025)
def laplacian_matrix(n, m):
    N = n * m
    main = -4 * np.ones(N)
    side1 = np.ones(N - 1)
    side1[m - 1::m] = 0
    side2 = np.ones(N - m)
    diags_list = [main, side1, side1, side2, side2]
    offsets = [0, -1, 1, -m, m]
    return diags(diags_list, offsets, format='csr')


class Config:
    def __init__(self, **kwargs):
        self.IMAGE_PATH = 'test.png'
        try:
            image = Image.open(self.IMAGE_PATH).convert('L')
            image = image.resize((256, 256))
            self.img = np.array(image)
            self.width, self.height = image.size
        except FileNotFoundError:
            self.img = np.zeros((256, 256))  # Fallback if image not found
            self.width, self.height = 256, 256
            print(f"Warning: Image file not found at {self.IMAGE_PATH}. Using a blank image.")

        self.MAX_LIFE = 100.0
        self.MASS = 1.0
        self.DT = 1.0
        self.MAX_PARTICLES = 150  # 增加一点数量以展示增殖效果
        self.SIMULATION_STEPS = 300
        self.INIT_PARTICLES = 30
        self.QUIVER_DENSITY = int(min(self.width, self.height) / 15)
        self.LIFE_RECOVERY_RATE = 5.0
        self.LIFE_DECAY_RATE = 1.0  # 降低衰减率让粒子存活更久
        self.BG_RATIO_THRESHOLD = 0.5
        self.STABLE_TIME = int(self.MAX_LIFE / self.LIFE_RECOVERY_RATE)
        self.W_ATTRACTION = 10.0
        self.W_REPULSION = 100.0  # 增加排斥力以看得更清楚
        self.W_DAMPING = 0.1  # 阻尼不应过大
        self.MAX_LINK_LEN = 40  # 连杆分裂的最大长度
        self.MIN_LINK_LEN = 5  # 弹簧力的理想长度
        self.W_SPRING = 0.01  # 弹簧力权重

        for k, v in kwargs.items():
            setattr(self, k, v)


config = Config()
particle_id_counter = itertools.count()


class Particle:
    def __init__(self, x, y, vel=None, force=None):
        self.id = next(particle_id_counter)  # 使用计数器保证ID唯一
        self.pos = np.array([x, y], dtype=float)
        self.vel = vel if vel is not None else np.zeros(2, dtype=float)
        self.force = force if force is not None else np.zeros(2, dtype=float)
        self.life = config.MAX_LIFE
        self.mass = config.MASS
        self.neighbors = {}
        self.time_on_foreground = 0
        self.on_fg = False

    def update_life(self):
        if self.on_fg:
            self.life = min(config.MAX_LIFE, self.life + config.LIFE_RECOVERY_RATE)
            self.time_on_foreground += 1
        else:
            self.life -= config.LIFE_DECAY_RATE
            self.time_on_foreground = 0

    def move(self):
        accel = self.force / self.mass
        self.vel += accel * config.DT
        self.pos += self.vel * config.DT

        # 边界碰撞反弹
        if self.pos[0] <= 0:
            self.pos[0] = 0
            self.vel[0] *= -0.5
        if self.pos[0] >= config.width - 1:
            self.pos[0] = config.width - 1
            self.vel[0] *= -0.5
        if self.pos[1] <= 0:
            self.pos[1] = 0
            self.vel[1] *= -0.5
        if self.pos[1] >= config.height - 1:
            self.pos[1] = config.height - 1
            self.vel[1] *= -0.5


class World:
    def __init__(self):
        self.particles = {}
        self.foreground_mask = (config.img < 128).astype(float)
        self.nabla2 = laplacian_matrix(config.height, config.width)
        self.field_gx = np.zeros((config.height, config.width), dtype=float)
        self.field_gy = np.zeros((config.height, config.width), dtype=float)

        rho_attract = -self.foreground_mask  # 吸引电荷密度
        b_attract = -rho_attract.ravel()
        with warnings.catch_warnings():  # 忽略稀疏矩阵求解的警告
            warnings.simplefilter("ignore")
            phi_attract = spsolve(self.nabla2, b_attract).reshape(config.height, config.width)

        self.gy_a, self.gx_a = np.gradient(phi_attract)  # 吸引力场 E = -grad(phi)
        self.gy_a *= config.W_ATTRACTION
        self.gx_a *= config.W_ATTRACTION

        # 静电屏蔽
        self.gy_a[self.foreground_mask > 0] = 0.0
        self.gx_a[self.foreground_mask > 0] = 0.0

    def add_particle(self, x, y, vel=None, force=None):
        if len(self.particles) >= config.MAX_PARTICLES:
            return None
        particle = Particle(x, y, vel, force)
        self.particles[particle.id] = particle
        return particle

    def remove_particle(self, particle_id):
        if particle_id not in self.particles: return
        particle = self.particles[particle_id]
        # 从邻居中删除自己
        for neighbor_id in particle.neighbors:
            if neighbor_id in self.particles and particle.id in self.particles[neighbor_id].neighbors:
                del self.particles[neighbor_id].neighbors[particle.id]
        del self.particles[particle_id]

    def get_line_pixels_bg_ratio(self, p1, p2):
        x1, y1 = int(p1.pos[0]), int(p1.pos[1])
        x2, y2 = int(p2.pos[0]), int(p2.pos[1])
        num_points = int(np.ceil(np.linalg.norm(p1.pos - p2.pos)))
        if num_points < 2: return 0.0
        x = np.linspace(x1, x2, num_points).astype(int)
        y = np.linspace(y1, y2, num_points).astype(int)
        x = np.clip(x, 0, config.width - 1)
        y = np.clip(y, 0, config.height - 1)
        line_mask_values = self.foreground_mask[y, x]
        bg_count = np.sum(line_mask_values == 0)
        return bg_count / num_points

    def initialize(self, num_particles):
        fg_pixels = np.argwhere(self.foreground_mask > 0)
        if len(fg_pixels) == 0:
            print("No foreground pixels to initialize particles.")
            return
        num = min(num_particles, len(fg_pixels), config.MAX_PARTICLES)
        if num != num_particles: warnings.warn(f"{num_particles} particles are too much, only {num} can be added.")
        for _ in range(num):
            py, px = random.choice(fg_pixels)
            self.add_particle(px, py)

    def calcu_force_field(self):
        pos_xy = np.array([p.pos for p in self.particles.values()]) if self.particles else np.empty((0, 2))
        if pos_xy.size == 0:
            self.field_gx = self.gx_a.copy()
            self.field_gy = self.gy_a.copy()
            return

        pos_x, pos_y = pos_xy[:, 0], pos_xy[:, 1]

        # 性能注意点：histogram2d 在每个step都计算，但如果粒子移动不快，可以隔几帧算一次
        repulse_mask, _, _ = np.histogram2d(
            pos_y, pos_x, bins=(config.height, config.width),
            range=[[0, config.height], [0, config.width]]
        )
        rho_repulse = repulse_mask
        b_repulse = rho_repulse.ravel()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phi_repulse = spsolve(self.nabla2, b_repulse).reshape(config.height, config.width)

        gy_r, gx_r = np.gradient(phi_repulse)
        gy_r *= config.W_REPULSION
        gx_r *= config.W_REPULSION

        self.field_gx = self.gx_a + gx_r
        self.field_gy = self.gy_a + gy_r

    def update_lifecycle(self):
        dead_particles = [p_id for p_id, p in self.particles.items() if p.life <= 0]
        for p_id in dead_particles:
            self.remove_particle(p_id)

    def proliferate(self):
        new_links = []
        links_to_break = []

        # 遍历现有连接，检查是否需要分裂
        processed_links = set()
        for p_id, p in list(self.particles.items()):
            for neighbor_id in list(p.neighbors.keys()):
                link_id = tuple(sorted((p_id, neighbor_id)))
                if link_id in processed_links:
                    continue
                processed_links.add(link_id)

                neighbor = self.particles.get(neighbor_id)
                if not neighbor: continue

                dist = np.linalg.norm(p.pos - neighbor.pos)
                if dist > config.MAX_LINK_LEN:
                    bg_ratio = self.get_line_pixels_bg_ratio(p, neighbor)
                    if (p.time_on_foreground > config.STABLE_TIME and
                            neighbor.time_on_foreground > config.STABLE_TIME and
                            bg_ratio > config.BG_RATIO_THRESHOLD):

                        # 标记断开旧连接
                        links_to_break.append(link_id)

                        # 创建新粒子
                        new_pos = (p.pos + neighbor.pos) / 2
                        new_vel = (p.vel + neighbor.vel) / 2
                        new_p = self.add_particle(new_pos[0], new_pos[1], new_vel)
                        if new_p:
                            new_p.life = (p.life + neighbor.life) / 2
                            # 标记连接新粒子
                            new_links.append((p.id, new_p.id))
                            new_links.append((neighbor.id, new_p.id))

        # 检查孤立点是否需要增殖
        for p_id, p in list(self.particles.items()):
            if not p.neighbors and p.on_fg and np.linalg.norm(p.vel) < 0.1:
                # 在随机方向上创建一个新粒子
                angle = random.uniform(0, 2 * np.pi)
                new_pos = p.pos + np.array([np.cos(angle), np.sin(angle)]) * config.MIN_LINK_LEN
                new_p = self.add_particle(new_pos[0], new_pos[1], np.zeros(2))
                if new_p:
                    p.vel = np.zeros(2)
                    new_links.append((p.id, new_p.id))

        # 执行拓扑更新
        for p1_id, p2_id in links_to_break:
            if p1_id in self.particles and p2_id in self.particles[p1_id].neighbors:
                del self.particles[p1_id].neighbors[p2_id]
            if p2_id in self.particles and p1_id in self.particles[p2_id].neighbors:
                del self.particles[p2_id].neighbors[p1_id]

        for p1_id, p2_id in new_links:
            if p1_id in self.particles and p2_id in self.particles:
                self.particles[p1_id].neighbors[p2_id] = self.particles[p2_id]
                self.particles[p2_id].neighbors[p1_id] = self.particles[p1_id]

    def apply_spring_forces(self):
        for p in self.particles.values():
            for neighbor in p.neighbors.values():
                # 防止重复计算
                if p.id < neighbor.id:
                    vec = p.pos - neighbor.pos
                    dist = np.linalg.norm(vec)
                    if dist > 1e-6:
                        # 弹簧力 F = -k * (dist - L0) * vec_normalized
                        force_magnitude = config.W_SPRING * (dist - config.MIN_LINK_LEN)
                        force_vec = force_magnitude * vec / dist
                        p.force -= force_vec
                        neighbor.force += force_vec

    def update(self):
        if not self.particles:
            return

        self.calcu_force_field()

        # 重置所有粒子的力，然后计算
        for p in self.particles.values():
            p.force = np.zeros(2, dtype=float)
            x, y = int(p.pos[1]), int(p.pos[0])  # NUMPY INDEXING (y, x)
            y = np.clip(y, 0, config.height - 1)
            x = np.clip(x, 0, config.width - 1)

            p.on_fg = self.foreground_mask[y, x] > 0

            # 从场中获取力
            force_from_field = np.array([self.field_gx[y, x], self.field_gy[y, x]], dtype=float)
            p.force += force_from_field

            # 计算阻尼力
            damping_force = -config.W_DAMPING * p.vel
            if p.on_fg:  # 只有在前景上才有强阻尼
                p.force += damping_force

        self.apply_spring_forces()  # 添加弹簧力

        # 更新生命并移动
        for p in list(self.particles.values()):
            p.update_life()
            p.move()

        self.update_lifecycle()
        self.proliferate()


# --- 可视化部分 ---
world = World()
world.initialize(config.INIT_PARTICLES)

fig, ax = plt.subplots(figsize=(10, 10))

# 预计算quiver的位置
skip = config.QUIVER_DENSITY
X, Y = np.meshgrid(np.arange(0, config.width, skip), np.arange(0, config.height, skip))


# 动画更新函数
def animate(frame):
    ax.clear()

    # 更新世界
    world.update()

    # 绘制背景图
    ax.imshow(config.img, cmap='gray', extent=(0, config.width, config.height, 0))

    # 绘制力场
    field_mag = np.sqrt(world.field_gx ** 2 + world.field_gy ** 2)
    # 用imshow绘制力场大小作为蒙版
    ax.imshow(field_mag, cmap='viridis', alpha=0.3, extent=(0, config.width, config.height, 0))
    # 绘制矢量箭头
    U = world.field_gx[Y, X]
    V = world.field_gy[Y, X]
    ax.quiver(X, Y, U, V, color='red', alpha=0.6, headwidth=2, headlength=3)

    # 绘制粒子和连杆
    if world.particles:
        positions = np.array([p.pos for p in world.particles.values()])
        lives = np.array([p.life for p in world.particles.values()])

        # 绘制连杆
        for p in world.particles.values():
            for neighbor in p.neighbors.values():
                if p.id < neighbor.id:  # 防止重复绘制
                    ax.plot([p.pos[0], neighbor.pos[0]], [p.pos[1], neighbor.pos[1]], 'w-', lw=1.5, alpha=0.8)

        # 绘制粒子 (颜色与生命值相关)
        sc = ax.scatter(positions[:, 0], positions[:, 1], c=lives, cmap='jet', vmin=0, vmax=config.MAX_LIFE, s=50,
                        edgecolors='w', zorder=10)

        if frame == 0:  # 第一次创建colorbar
            plt.colorbar(sc, ax=ax, label='Particle Life')

    ax.set_title(f"Frame: {frame + 1}/{config.SIMULATION_STEPS}, Particles: {len(world.particles)}")
    ax.set_xlim(0, config.width)
    ax.set_ylim(config.height, 0)
    ax.set_aspect('equal')


# 创建并保存动画
ani = FuncAnimation(fig, animate, frames=config.SIMULATION_STEPS, interval=50, repeat=False)
plt.show()
print("Generating animation... This may take a while.")
ani.save('particle_simulation.gif', writer='pillow', fps=20)
print("Animation saved as particle_simulation.gif")
