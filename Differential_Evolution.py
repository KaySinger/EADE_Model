import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# 定义列表用于存储
objective_values = []
mse_values = []

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, x_values, scale_factor):
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 定义非线性微分方程组
def equations(p, t, k):
    dpdt = np.zeros_like(p)
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * (p[1]**2)
    for i in range(2, 10):
        dpdt[i] = k[i-1] * (p[i-1]**2) - k[i] * (p[i]**2)
    dpdt[10] = k[9] * (p[9]**2)
    return dpdt

# 定义目标函数
def objective_global(k):
    initial_p = [10.0] + [0] * 10
    t = np.linspace(0, 200, 1000)
    # 求解微分方程
    sol = odeint(equations, initial_p, t, args=(k,))
    final_p = sol[-1, :] # 取最终浓度
    # 理想最终浓度
    ideal_p = [0] + list(target_p)
    # 计算误差
    sum_error = np.sum((final_p - ideal_p)**2)
    mse_error = sum_error / len(final_p)
    # 存储求得的误差
    objective_values.append(sum_error)
    mse_values.append(mse_error)

    return sum_error

# 定义目标函数
def objective_local(k):
    initial_p = [10.0] + [0] * 40
    t = np.linspace(0, 200, 1000)
    # 求解微分方程
    sol = odeint(equations, initial_p, t, args=(k,))
    final_p = sol[-1, :]  # 取最终浓度
    # 理想最终浓度
    ideal_p = [0] + list(target_p)
    # 计算误差
    sum_error = np.sum((final_p - ideal_p)**2)
    mse_error = sum_error / len(final_p)

    return sum_error

# 局部优化回调函数
def callback(xk):
    current_value = objective_local(xk)
    objective_values.append(current_value)
    if len(objective_values) > 1:
        change = np.abs(objective_values[-1] - objective_values[-2])
        print(f"迭代次数 {len(objective_values) - 1}: 变化 = {change}, 精度 = {objective_values[-1]}")

# 实现差分进化算法
class DifferentialEvolution:
    def __init__(self, func, bounds, mutation_factor=0.8, crossover_prob=0.7, population_size=100, seed=None):
        """
        初始化 DE 算法参数
        :param func: 目标函数
        :param bounds: 搜索空间边界，格式为 [(min, max), (min, max), ...]
        :param mutation_factor: 变异因子 (F)，默认为 0.8
        :param crossover_prob: 交叉概率 (CR)，默认为 0.9
        :param population_size: 种群大小，默认为 50
        :param seed: 随机种子，默认为 None
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.population_size = population_size
        self.num_dimensions = len(bounds)
        self.random_state = np.random.default_rng(seed)
        self.population = self._initialize_population()
        self.fitness = np.array([self.func(ind) for ind in self.population])

    def _initialize_population(self):
        """初始化种群"""
        return self.random_state.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.population_size, self.num_dimensions)
        )

    def _mutate_current_to_best(self, idx):
        """
        current-to-best/1 变异策略
        :param idx: 当前个体的索引
        :return: donor 向量
        """
        best_idx = np.argmin(self.fitness)  # 当前最优个体
        a, b = self._select_random_indices(idx, 2)  # 随机选择两个不同的个体
        donor = (self.population[idx] +
                 self.mutation_factor * (self.population[best_idx] - self.population[idx]) +
                 self.mutation_factor * (self.population[a] - self.population[b]))
        return donor

    def _select_random_indices(self, idx, num_samples):
        """
        随机选择不同于当前个体的索引
        :param idx: 当前个体的索引
        :param num_samples: 需要选择的样本数量
        :return: 选择的索引列表
        """
        indices = list(range(self.population_size))
        indices.remove(idx)
        return self.random_state.choice(indices, num_samples, replace=False)

    def _crossover(self, target, donor):
        """
        二项式交叉
        :param target: 目标向量
        :param donor: donor 向量
        :return: 试验向量
        """
        cross_points = self.random_state.random(self.num_dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.random_state.integers(0, self.num_dimensions)] = True
        trial = np.where(cross_points, donor, target)
        return trial

    def _clip_to_bounds(self, individual):
        """将个体限制在搜索空间范围内"""
        return np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])

    def solve(self, max_iter=1000, tol=1e-6):
        """
        优化求解过程
        :param max_iter: 最大迭代次数
        :param tol: 收敛精度（当相邻两次最优值变化小于 tol 时停止）
        :return: 最优解和对应目标值
        """
        best_fitness_history = []  # 记录每次迭代的最优目标值

        for iteration in range(max_iter):
            for idx in range(self.population_size):
                # 变异：current-to-best/1
                donor = self._mutate_current_to_best(idx)
                donor = self._clip_to_bounds(donor)

                # 交叉
                trial = self._crossover(self.population[idx], donor)

                # 选择
                trial_fitness = self.func(trial)
                if trial_fitness < self.fitness[idx]:
                    self.population[idx] = trial
                    self.fitness[idx] = trial_fitness

            # 记录当前迭代的最优值
            best_idx = np.argmin(self.fitness)
            current_best_fitness = self.fitness[best_idx]
            best_fitness_history.append(current_best_fitness)

            print(f"Iteration {iteration + 1}: Best Fitness = {current_best_fitness}")

            # 检查精度终止条件
            if current_best_fitness <= tol:
                print(f"Converged at generation {iteration} with precision {current_best_fitness:.6e}")
                break

        # 返回最终最优解和目标值
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx], best_fitness_history

def visualize_fitness():
    # 绘制目标函数和均方误差的收敛曲线
    plt.figure(figsize=(15, 8))
    plt.xlabel("Iteration")
    plt.ylabel("Objective_Fitness")
    plt.title("Objective Fitness Across Iterations")
    plt.plot(fitness_history, label='Objective_fitness', color='red')
    plt.legend()
    plt.grid(True)
    plt.show()

# 设置变量边界
bounds = np.array([(1.0, 10)] + [(0.01, 10)] * 9)

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=5.5, sigma=8, total_concentration=1.0, x_values=np.arange(1, 11), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 11)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 运行差分进化算法
objective_values.clear()
mse_values.clear()
result = DifferentialEvolution(objective_global, bounds, mutation_factor=0.8, crossover_prob=0.9, population_size=100, seed=42)
optimal_k, final_precision, fitness_history = result.solve(max_iter=1000, tol=1e-6)
print("全局优化得到的系数k:", {f'k{i}': c for i, c in enumerate(optimal_k, start=0)})
print("最终精度:", final_precision)

visualize_fitness()

# # 梯度优化，进一步提高精度
# print("开始梯度优化")
#
# objective_values.clear()
# mse_values.clear()
#
# result_final = minimize(objective_local, optimal_k, method='L-BFGS-B', bounds=bounds, callback=callback, tol=1e-8)
# optimal_k = result_final.x
# final_precision = result_final.fun
#
# print("反应系数K是:", {f"k{i}:": c for i, c in enumerate(optimal_k, start=0)})
# print("最终优化精度:", final_precision)

# 使用得到的系数求解
initial_p = [10.0] + [0] * 10
t = np.linspace(0, 200, 1000)
sol = odeint(equations, initial_p, t, args=(optimal_k,))

# 绘制理想稳态浓度曲线
plt.figure(figsize=(15, 8))
plt.xlabel("P-Species")
plt.ylabel("P-Concentrations")
plt.title("Ideal Concentrations and Actual Concentrations")
plt.xticks(range(len(x_values)), x_values, rotation=90)
final_concentrations = sol[-1, 1:]
plt.plot(range(len(x_values)), target_p, label = 'Ideal Concentrations', marker='o', linestyle='-', color='blue')
plt.plot(range(len(x_values)), final_concentrations, label = 'Actual Concentrations', marker='o', linestyle='-', color='red')
plt.grid(True)
plt.show()

# 绘图函数
plt.figure(figsize=(15, 8))
plt.plot(t, sol[:, 0], label='p0')
for i in range(1, 11):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P0-P10 Concentration over Time')
plt.grid(True)
plt.show()

# plt.figure(figsize=(15, 8))
# for i in range(11, 21):
#     plt.plot(t, sol[:, i], label=f'p{i}')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Concentration')
# plt.title('P11-P20 Concentration over Time')
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(15, 8))
# for i in range(21, 31):
#     plt.plot(t, sol[:, i], label=f'p{i}')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Concentration')
# plt.title('P21-P30 Concentration over Time')
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(15, 8))
# for i in range(31, 41):
#     plt.plot(t, sol[:, i], label=f'p{i}')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Concentration')
# plt.title('P31-P40 Concentration over Time')
# plt.grid(True)
# plt.show()