import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, x_values, scale_factor):
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 定义非线性微分方程组
def equations_1(p, t, k):
    dpdt = np.zeros_like(p)
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * p[1] ** 2
    for i in range(2, 40):
        dpdt[i] = k[i - 1] * p[i - 1] ** 2 - k[i] * p[i] ** 2
    dpdt[40] = k[39] * p[39] ** 2
    return dpdt

# 定义非线性微分方程组
def equations_2(p, t, k_values):
    dpdt = np.zeros_like(p)
    k = k_values[:40]
    k_inv = k_values[40:]
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * p[1] ** 2
    for i in range(2, 40):
        dpdt[i] = k[i - 1] * p[i - 1] ** 2 + k_inv[i - 1] * p[i + 1] - k_inv[i - 2] * p[i] - k[i] * p[i] ** 2
    dpdt[40] = k[39] * p[39] ** 2 - k_inv[38] * p[40]
    return dpdt


# 定义目标函数
def objective_global(k):
    initial_p = [10.0] + [0] * 40
    t = np.linspace(0, 500, 1000)
    # 求解微分方程
    sol = odeint(equations_1, initial_p, t, args=(k,))
    final_p = sol[-1, :] # 取最终浓度
    # 理想最终浓度
    ideal_p = [0] + list(target_p)
    # 计算误差
    sum_error = np.sum((final_p - ideal_p)**2)
    mse_error = sum_error / len(final_p)

    return mse_error

# 定义目标函数
def objective_local(k):
    initial_p = [10.0] + [0] * 40
    t = np.linspace(0, 500, 1000)
    # 求解微分方程
    sol = odeint(equations_1, initial_p, t, args=(k,))
    final_p = sol[-1, :] # 取最终浓度
    # 理想最终浓度
    ideal_p = [0] + list(target_p)
    # 计算误差
    sum_error = np.sum((final_p - ideal_p)**2)
    mse_error = sum_error / len(final_p)

    return mse_error

def jade(func, bounds, pop_size=None, max_gen=None, p=0.1, tol=None):
    dim = len(bounds)
    archive = []
    iteration_log = []
    F_mean, CR_mean = 0.6, 0.6  # 初始化平均参数

    # 用 LHS 初始化种群
    def latin_hypercube_sampling(bounds, pop_size):
        population = np.zeros((pop_size, len(bounds)))
        for i in range(len(bounds)):
            intervals = np.linspace(0, 1, pop_size + 1)[:-1]
            points = intervals + np.random.uniform(0, 1 / pop_size, pop_size)
            np.random.shuffle(points)
            lb, ub = bounds[i]
            population[:, i] = lb + points * (ub - lb)
        return population

    pop = latin_hypercube_sampling(bounds, pop_size)
    fitness = np.apply_along_axis(func, 1, pop)

    for gen in range(max_gen):
        F_values, CR_values = [], []
        new_pop = []

        # 检查精度终止条件
        best_val = np.min(fitness)
        iteration_log.append(best_val)
        if best_val <= tol:
            print(f"Converged at generation {gen} with precision {best_val:.6e}")
            break

        for i in range(pop_size):
            # 动态调整 F 和 CR
            F = np.clip(np.random.normal(F_mean, 0.1), 0, 1)
            CR = np.clip(np.random.normal(CR_mean, 0.1), 0, 1)
            F_values.append(F)
            CR_values.append(CR)

            # current-to-best/1 变异策略
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            a, b = pop[np.random.choice(pop_size, 2, replace=False)]
            mutant = pop[i] + F * (best - pop[i]) + F * (a - b)
            mutant = np.clip(mutant, [b[0] for b in bounds], [b[1] for b in bounds])

            # 二项交叉
            trial = np.array([mutant[j] if np.random.rand() < CR else pop[i][j] for j in range(dim)])
            trial_fitness = func(trial)

            # 贪心选择
            if trial_fitness < fitness[i]:
                new_pop.append(trial)
                fitness[i] = trial_fitness
                archive.append(pop[i])
            else:
                new_pop.append(pop[i])

        # 更新种群和外部存档
        pop = np.array(new_pop)
        archive = archive[-pop_size:]  # 限制存档大小

        # 更新 F 和 CR 的均值
        if F_values:
            F_mean = (1 - 0.1) * F_mean + 0.1 * np.mean(F_values)
        if CR_values:
            CR_mean = (1 - 0.1) * CR_mean + 0.1 * np.mean(CR_values)

        print(f"当前迭代次数{gen+1}, 迭代精度{np.min(fitness)}")

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx], iteration_log

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
bounds_1 = np.array([(0.5, 10)] + [(0.01, 15)] * 39)
bounds_2 = np.array([(0, None)] * 79)

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=20.5, sigma=8, total_concentration=1.0, x_values=np.arange(1, 41), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 41)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 运行差分进化算法
best_solution, best_fitness, fitness_history = jade(objective_global, bounds=bounds_1, pop_size=400, max_gen=1000, p=0.1, tol=1e-6)
print("全局优化得到的系数k:", {f'k{i}': c for i, c in enumerate(best_solution, start=0)})
print("最终精度:", best_fitness)

# 使用得到的系数求解
initial_p = [10.0] + [0] * 40
t = np.linspace(0, 500, 1000)
sol = odeint(equations_1, initial_p, t, args=(best_solution,))

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

visualize_fitness()

# 梯度优化，进一步提高精度
print("开始梯度优化")
k = np.zeros(40)
k_inv = np.zeros(39)
for i in range(39):
    k[i] = best_solution[i]
    k_inv[i] = (k[i+1] * target_p[i]**2) / target_p[i+1]
initial_condition = np.concatenate(k, k_inv)
result_final = minimize(objective_local, initial_condition, method='L-BFGS-B', bounds=bounds_2, tol=1e-8)
optimal_k = result_final.x
final_precision = result_final.fun

print("反应系数K是:", {f"k{i}:": c for i, c in enumerate(optimal_k, start=0)})
print("最终优化精度:", final_precision)

# 使用得到的系数求解
initial_p = [10.0] + [0] * 40
t = np.linspace(0, 500, 1000)
sol = odeint(equations_2, initial_p, t, args=(optimal_k,))

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