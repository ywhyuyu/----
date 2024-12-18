import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# 设置随机数种子以获得可重复的结果
np.random.seed(42)

# 定义目标函数
def objective_function(x):
    return x[0]**2 - x[1]**2

# PSO 参数设置
class PSO:
    def __init__(self, func, lb, ub, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        self.func = func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w      # 惯性权重
        self.c1 = c1    # 个人学习因子
        self.c2 = c2    # 社会学习因子
        self.dim = len(lb)
        # 初始化粒子位置和速度
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.num_particles, self.dim))
        self.V = np.random.uniform(low=-abs(self.ub - self.lb), high=abs(self.ub - self.lb), size=(self.num_particles, self.dim))
        # 初始化个体最佳和全局最佳
        self.pbest = self.X.copy()
        self.pbest_val = np.array([self.func(x) for x in self.X])
        self.gbest_idx = np.argmin(self.pbest_val)
        self.gbest = self.pbest[self.gbest_idx].copy()
        self.gbest_val = self.pbest_val[self.gbest_idx]
        # 记录历史
        self.history = []
        self.position_history = []
        self.all_position_history = []

    def optimize_step(self):
        for i in range(self.num_particles):
            # 更新速度
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive = self.c1 * r1 * (self.pbest[i] - self.X[i])
            social = self.c2 * r2 * (self.gbest - self.X[i])
            self.V[i] = self.w * self.V[i] + cognitive + social
            # 更新位置
            self.X[i] = self.X[i] + self.V[i]
            # 处理边界
            self.X[i] = np.clip(self.X[i], self.lb, self.ub)
            # 评估新位置
            fitness = self.func(self.X[i])
            # 更新个体最佳
            if fitness < self.pbest_val[i]:
                self.pbest[i] = self.X[i].copy()
                self.pbest_val[i] = fitness
                # 更新全局最佳
                if fitness < self.gbest_val:
                    self.gbest = self.pbest[i].copy()
                    self.gbest_val = fitness
        # 记录历史
        self.history.append(self.gbest_val)
        self.position_history.append(self.gbest.copy())
        self.all_position_history.append(self.X.copy())

    def optimize(self):
        for iter in range(self.max_iter):
            self.optimize_step()
            print(f"迭代 {iter+1}/{self.max_iter}, 最优值: {self.gbest_val}")
        return self.gbest, self.gbest_val

# 设置参数
lb = [-10, -10]
ub = [10, 10]
num_particles = 50
max_iter = 100
w = 0.5
c1 = 1.5
c2 = 1.5

# 创建 PSO 实例
pso = PSO(objective_function, lb, ub, num_particles=num_particles, max_iter=max_iter, w=w, c1=c1, c2=c2)

# 执行优化
best_position, best_value = pso.optimize()

# 打印最优结果
print(f"最优位置: x1 = {best_position[0]}, x2 = {best_position[1]}")
print(f"最小值: {best_value}")

# 创建三维图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 创建网格以绘制函数曲面
X = np.linspace(lb[0], ub[0], 400)
Y = np.linspace(lb[1], ub[1], 400)
X, Y = np.meshgrid(X, Y)
Z = objective_function([X, Y])

# 绘制函数曲面
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# 绘制全局最优轨迹
position_history = np.array(pso.position_history)
Z_history = objective_function(position_history.T)
ax.plot(position_history[:, 0], position_history[:, 1], Z_history, color='red', marker='o', label='Global optimal trajectory') #全局最优轨迹

# 标注最优点
ax.scatter(best_position[0], best_position[1], best_value, color='black', marker='*', s=100, label='Best of all') #最优点

# 设置标签和标题
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y = x1² - x2²')
ax.set_title('3D visualization of the particle swarm optimization (PSO) process') #粒子群优化（PSO）过程的三维可视化
ax.legend()

plt.show()

# 动画可视化
# 在此动画中，我们将展示每次迭代中所有粒子的移动
fig_anim = plt.figure(figsize=(12, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

# 绘制函数曲面
ax_anim.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# 初始化粒子点
scat = ax_anim.scatter([], [], [], color='blue', s=50, label='particle') #粒子
gbest_scatter = ax_anim.scatter([], [], [], color='red', marker='*', s=100, label='Global optimum') #全局最优

# 设置坐标轴范围
ax_anim.set_xlim(lb[0], ub[0])
ax_anim.set_ylim(lb[1], ub[1])
ax_anim.set_zlim(np.min(Z), np.max(Z))
ax_anim.set_xlabel('x1')
ax_anim.set_ylabel('x2')
ax_anim.set_zlabel('y = x1² - x2²')
ax_anim.set_title('Particle Swarm Optimization (PSO) process animation') #粒子群优化（PSO）过程动画
ax_anim.legend()

# 定义初始化动画函数
def init_anim():
    scat._offsets3d = ([], [], [])
    gbest_scatter._offsets3d = ([], [], [])
    return scat, gbest_scatter

# 定义动画更新函数
def animate_func(i):
    if i < len(pso.all_position_history):
        current_positions = pso.all_position_history[i]
        current_fitness = pso.func(current_positions.T)
        # 更新粒子位置
        scat._offsets3d = (current_positions[:, 0], current_positions[:, 1], current_fitness)
        # 更新全局最优位置
        current_gbest = pso.position_history[i]
        current_gbest_val = pso.func(current_gbest)
        gbest_scatter._offsets3d = ([current_gbest[0]], [current_gbest[1]], [current_gbest_val])
        # 更新标题
        ax_anim.set_title(f'Particle Swarm Optimization (PSO) process animation - iterate {i+1}')
    return scat, gbest_scatter

# 创建动画
ani = animation.FuncAnimation(fig_anim, animate_func, frames=max_iter, init_func=init_anim,
                              interval=200, blit=False, repeat=False)

plt.show()
