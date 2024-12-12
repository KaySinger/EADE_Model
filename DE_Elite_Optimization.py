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
    for i in range(2, 40):
        dpdt[i] = k[i-1] * (p[i-1]**2) - k[i] * (p[i]**2)
    dpdt[40] = k[39] * (p[39]**2)
    return dpdt

# 定义目标函数
def objective_global(k):
    initial_p = [10.0] + [0] * 40
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
    # 设立惩罚项，知道系数向着增大演变
    k_f = k[1:]
    penalty = np.sum(np.maximum(0, -np.diff(k_f))**2)
    alpha = 0.5

    return sum_error + alpha * penalty

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

# 定义局部优化函数，使用L-BFGS-B算法
def local_search(individual):
    result = minimize(objective_local, individual, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
    return result.x

# 实现差分进化算法
class DifferentialEvolutionSolver:
    def __init__(self, func, bounds, strategy='rand1bin', mutation=None, recombination=None, seed=None):
        self.func = func  # 优化目标函数
        self.bounds = np.array(bounds)
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.random_state = np.random.default_rng(seed)
        self.num_params = len(bounds)
        self.population_size = 10 * self.num_params  # 种群规模
        self.limits = np.array(bounds)
        self.population = self._init_population()
        self.fitness = np.array([self.func(ind) for ind in self.population])

    def _init_population(self):
        """随机初始化种群"""
        pop = self.random_state.uniform(self.limits[:, 0], self.limits[:, 1], size=(self.population_size, self.num_params))
        return pop

    def _select_samples(self, candidate_idx, num_samples):
        """从种群中选择不同于当前个体的样本"""
        indices = list(range(self.population_size))
        indices.remove(candidate_idx)
        chosen = self.random_state.choice(indices, num_samples, replace=False)
        return chosen

    def _rand1_bin(self, candidate_idx):
        """rand/1/bin策略的实现"""
        r1, r2, r3 = self._select_samples(candidate_idx, 3)  # 选择三个个体
        # 变异：生成 donor 向量
        donor = (self.population[r1] +
                 self.mutation * (self.population[r2] - self.population[r3]))
        # 限制 donor 在边界范围内
        donor = np.clip(donor, self.limits[:, 0], self.limits[:, 1])

        # 二进制交叉
        cross_points = self.random_state.random(self.num_params) < self.recombination
        if not np.any(cross_points):
            # 确保至少有一个维度发生交叉
            cross_points[self.random_state.integers(0, self.num_params)] = True

        # 生成试验向量 trial
        trial = np.where(cross_points, donor, self.population[candidate_idx])
        return trial

    def solve(self, max_iter=1000, tol=1e-8):
        """
        优化求解过程
        :param max_iter: 最大迭代次数
        :param tol: 收敛精度（当相邻两次最优值变化小于 tol 时停止）
        :return: 最优解和对应目标值
        """
        best_fitness_history = []  # 记录每次迭代的最优目标值
        iteration = 0  # 记录迭代次数

        # 初始和最终变异率和交叉率
        F_init = self.mutation
        CR_init = self.recombination
        F_final = 0.5
        CR_final = 0.5

        for iteration in range(max_iter):
            # 自适应调节变异率和交叉率
            F = F_init - (F_init - F_final) * iteration / max_iter
            CR = CR_init - (CR_init - CR_final) * iteration / max_iter
            self.mutation = F
            self.recombination = CR

            for idx in range(self.population_size):
                if self.strategy == 'rand1bin':
                    trial = self._rand1_bin(idx)
                else:
                    raise ValueError(f"策略 {self.strategy} 未实现")

                trial_fitness = self.func(trial)
                if trial_fitness < self.fitness[idx]:
                    self.population[idx] = trial
                    self.fitness[idx] = trial_fitness

            # 每50次迭代，进行一次局部优化
            if (iteration + 1) % 50 == 0:
                # 选择前50%的优秀个体
                sorted_indices = np.argsort(self.fitness)
                top_indices = sorted_indices[:int(self.population_size * 0.2)]
                for idx in top_indices:
                    try:
                        # 进行局部优化
                        result = minimize(self.func, self.population[idx], method='L-BFGS-B',
                                                                 bounds=self.bounds, options={'maxiter': 50})
                        optimized_ind = result.x
                        # 计算优化后的适应度
                        optimized_fitness = self.func(optimized_ind)
                        # 如果优化后的适应度更好，替换原来的个体
                        if optimized_fitness < self.fitness[idx]:
                            self.population[idx] = optimized_ind
                            self.fitness[idx] = optimized_fitness
                    except Exception as e:
                        print(f"局部优化失败: {e}")

            # 记录当前迭代的最优值
            best_idx = np.argmin(self.fitness)
            current_best_fitness = self.fitness[best_idx]
            best_fitness_history.append(current_best_fitness)

            print(f"Iteration {iteration + 1}: Objective Fitness = {current_best_fitness}")

            # 检查收敛条件
            if iteration > 0 and abs(best_fitness_history[-1] - best_fitness_history[-2]) < tol and current_best_fitness < tol:
                print(f"在第 {iteration} 次迭代时达到收敛精度：{tol}")
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
bounds = np.array([(0, 10)] * 40)

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=20.5, sigma=8, total_concentration=1.0, x_values=np.arange(1, 41), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 41)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 运行差分进化算法
objective_values.clear()
mse_values.clear()
result = DifferentialEvolutionSolver(objective_global, bounds=bounds, strategy='rand1bin', mutation=0.8, recombination=0.8, seed=42)
best_solution, best_fitness, fitness_history = result.solve(max_iter=1000, tol=1e-8)
print("全局优化得到的系数k:", {f'k{i}': c for i, c in enumerate(best_solution, start=0)})
print("最终精度:", best_fitness)

visualize_fitness()

# 梯度优化，进一步提高精度
print("开始梯度优化")

objective_values.clear()
mse_values.clear()

result_final = minimize(objective_local, best_solution, method='L-BFGS-B', bounds=bounds, callback=callback, tol=1e-8)
optimal_k = result_final.x
final_precision = result_final.fun

print("反应系数K是:", {f"k{i}:": c for i, c in enumerate(optimal_k, start=0)})
print("最终优化精度:", final_precision)

# 使用得到的系数求解
initial_p = [10.0] + [0] * 40
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

plt.figure(figsize=(15, 8))
for i in range(11, 21):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P11-P20 Concentration over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 8))
for i in range(21, 31):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P21-P30 Concentration over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 8))
for i in range(31, 41):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P31-P40 Concentration over Time')
plt.grid(True)
plt.show()