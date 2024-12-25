import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.stats import cauchy
from matplotlib import pyplot as plt

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
    # 设立惩罚项，知道系数向着增大演变
    k_f = k[1:]
    penalty = np.sum(np.maximum(0, -np.diff(k_f)) ** 2)
    alpha = 0.5

    return sum_error + alpha * penalty

# 定义差分进化算法
def DPADE(func, bounds=None, max_iter=None, NP=200, F=0.5, CR=0.9, alpha=1.0, beta=0.5, tol=None):
    dim = len(bounds)
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    # 初始化种群
    population = np.random.uniform(lb, ub, size=(NP, dim))
    fitness = np.array([func(ind) for ind in population])

    # 初始化控制参数 F 和 CR
    F = np.random.uniform(0.5, 0.8, NP)  # 初始化 F 为均匀分布
    CR = np.random.uniform(0.5, 0.9, NP)  # 初始化 CR 为均匀分布

    # 记录最佳个体
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]
    fitness_history = []

    # 动态调整种群规模
    NPG_max = int(0.45 * NP)
    NPG_min = 4

    for G in range(max_iter):
        # 计算变异概率 MR
        MR = 0.5 * (1 + np.cos(np.pi * (G / max_iter) ** alpha))

        # 动态调整优势种群规模
        NPG = int(NPG_max + (NPG_max - NPG_min) * (1 - (G / max_iter) ** 2))
        NPB = NP - NPG

        # 分类种群
        sorted_idx = np.argsort(fitness)
        NPG_idx = sorted_idx[:NPG]
        NPB_idx = sorted_idx[NPG:]

        # 检查精度终止条件
        best_val = np.min(fitness)
        if best_val <= tol:
            print(f"Converged at generation {G} with precision {best_val:.6e}")
            break

        for i in range(NP):
            # 选择变异策略
            if np.random.rand() < MR:
                # 全局搜索策略
                r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                mutant = population[r1] + F[i] * (population[r2] - population[r3])
            else:
                # 局部开发策略
                r1, r2 = np.random.choice(NPG_idx, 2, replace=False)
                r3 = np.random.choice(NPB_idx)
                mutant = population[r1] + F[i] * (population[r2] - population[r3])

            # 交叉操作
            trial = np.copy(population[i])
            j_rand = np.random.randint(0, dim)
            for j in range(dim):
                if np.random.rand() < CR[i] or j == j_rand:
                    trial[j] = mutant[j]

            # 边界处理
            trial = np.clip(trial, lb, ub)

            # 计算适应度
            f_trial = func(trial)

            # 选择操作
            if f_trial < fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                if f_trial < best_fitness:
                    best_solution = trial
                    best_fitness = f_trial

            # 更新控制参数
            for i in range(NP):
                if i in NPG_idx:
                    # 优势种群参数更新
                    F[i] = cauchy.rvs(loc=0.5, scale=0.1)  # 使用 scipy.stats.cauchy 生成柯西分布随机数
                    CR[i] = np.random.normal(0.5, 0.1)
                else:
                    # 劣势种群参数更新
                    F[i] = np.random.uniform(0.5, 0.8)
                    CR[i] = np.random.uniform(0.5, 0.9)

        # 邻域权重局部搜索
        for i in range(NP):
            neighbors = population[max(0, i - 2):min(NP, i + 3)]
            best_neighbor = neighbors[np.argmin([func(n) for n in neighbors])]
            D = best_neighbor - population[i]
            NL = beta * np.cos(np.pi / 2 * G / max_iter) * D
            trial = population[i] + NL
            trial = np.clip(trial, lb, ub)
            f_trial = func(trial)
            if f_trial < fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                if f_trial < best_fitness:
                    best_solution = trial
                    best_fitness = f_trial

        print(f"当前迭代次数{G + 1}, 迭代精度{np.min(fitness)}")
        fitness_history.append(np.min(fitness))

    return best_solution, best_fitness, fitness_history

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
bounds = np.array([(1, 2)] + [(0.01, 20)] * 39)

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=20.5, sigma=8, total_concentration=1.0, x_values=np.arange(1, 41), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 41)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 运行差分进化算法
best_solution, best_fitness, fitness_history = DPADE(objective_global, bounds=bounds, max_iter=1000, NP=400, F=0.5, CR=0.9, alpha=1.0, beta=0.5, tol=1e-6)
print("全局优化得到的系数k:", {f'k{i}': c for i, c in enumerate(best_solution, start=0)})
print("最终精度:", best_fitness)

visualize_fitness()

# 梯度优化，进一步提高精度
print("开始梯度优化")

result_final = minimize(objective_global, best_solution, method='L-BFGS-B', bounds=bounds, tol=1e-8)
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