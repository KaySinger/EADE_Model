import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, x_values, scale_factor):
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 定义非线性微分方程组
def equations(p, t, k_values):
    dpdt = np.zeros_like(p)
    k = k_values[:20]
    k_inv = k_values[20:]
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * (p[1] ** 2)
    for i in range(2, 20):
        dpdt[i] = k[i - 1] * (p[i - 1] ** 2) + k_inv[i - 1] * p[i + 1] - k_inv[i - 2] * p[i] - (k[i] * p[i] ** 2)
    dpdt[20] = (k[19] * p[19] ** 2) - k_inv[18] * p[20]
    return dpdt

# 定义目标函数
def objective_global(k):
    initial_p = [10.0] + [0] * 10
    t = np.linspace(0, 1000, 1000)
    sol = odeint(equations, initial_p, t, args=(k,))

    # 计算最终时间点的误差
    final_error = np.sum((sol[-1, 1:] - target_p) ** 2)

    # 计算浓度变化率
    concentration_change = np.diff(sol[:, 1:], axis=0)
    change_rate = np.sum(np.abs(concentration_change), axis=1)

    # 检查是否过早进入稳态
    t_threshold = 800
    if np.mean(change_rate[:t_threshold]) < 1e-6:  # 如果过早进入稳态
        final_error += 1e6  # 增加一个大的惩罚项

    return final_error

def local_search(func, individual, bounds):
    """局部搜索函数，使用 L-BFGS-B 方法对个体进行优化"""
    result = minimize(func, individual, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
    return result.x, result.fun

def DPADE(func, bounds, pop_size=None, max_gen=None, hist_size=200, tol=1e-6):
    dim = len(bounds)
    archive = []
    H = hist_size
    F_hist, CR_hist = [0.5] * H, [0.5] * H
    hist_idx = 0
    iteration_log = []

    # 设置参数p
    p_max, p_min = 0.15, 0.01

    # 初始化种群
    pop = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim))
    fitness = np.apply_along_axis(func, 1, pop)

    for gen in range(max_gen):
        F_values, CR_values = [], []
        S_F, S_CR = [], []
        new_pop = []

        # 参数p非线性下降
        p = p_min + 0.5 * (p_max - p_min) * (1 + np.cos(gen * np.pi / max_gen))

        # 检查精度终止条件
        best_val = np.min(fitness)
        iteration_log.append(best_val)
        if best_val <= tol:
            print(f"Converged at generation {gen} with precision {best_val:.6e}")
            break

        for i in range(pop_size):
            # 从成功历史记录中采样 F 和 CR
            hist_sample = np.random.randint(0, H)
            F = np.clip(np.random.standard_cauchy() * 0.1 + F_hist[hist_sample], 0, 1)
            CR = np.clip(np.random.normal(CR_hist[hist_sample], 0.1), 0, 1)
            F_values.append(F)
            CR_values.append(CR)

            # current-to-pbest/1 变异策略
            p_best_size = int(pop_size * p)
            p_best_indices = np.argsort(fitness)[:p_best_size]
            p_best_idx = np.random.choice(p_best_indices)
            p_best = pop[p_best_idx]
            a, b = pop[np.random.choice(pop_size, 2, replace=False)]
            mutant = pop[i] + F * (p_best - pop[i]) + F * (a - b)
            mutant = np.clip(mutant, [b[0] for b in bounds], [b[1] for b in bounds])

            # 二项交叉
            trial = np.array([mutant[j] if np.random.rand() < CR else pop[i][j] for j in range(dim)])
            trial_fitness = func(trial)

            # 贪心选择
            if trial_fitness < fitness[i]:
                new_pop.append(trial)
                fitness[i] = trial_fitness
                archive.append(pop[i])
                S_F.append(F)
                S_CR.append(CR)
            else:
                new_pop.append(pop[i])

        # 更新种群和外部存档
        pop = np.array(new_pop)
        if len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))

        # 更新成功历史记录
        if S_F and S_CR:
            weights = np.array([abs(fitness[i] - trial_fitness) for i in range(len(S_F))])
            weights /= np.sum(weights)
            F_hist[hist_idx] = np.sum(weights * np.array(S_F))
            CR_hist[hist_idx] = np.sum(weights * np.array(S_CR))
            hist_idx = (hist_idx + 1) % H

        # 停滞机制：对停滞次数达到阈值的个体进行局部优化（并行化）
        if (best_val < 0.1 or gen >= max_gen / 2) and gen % 100 == 0:
            elite_size = max(1, int(pop_size * 0.1))  # 前10%的精英个体
            elite_indices = np.argsort(fitness)[:elite_size]

            # 对精英个体进行局部搜索（并行化）
            results = Parallel(n_jobs=-1)(delayed(local_search)(func, pop[idx], bounds) for idx in elite_indices)
            for idx, (x_opt, f_opt) in zip(elite_indices, results):
                if f_opt < fitness[idx]:  # 仅当新解适应度更好时才更新
                    pop[idx] = x_opt
                    fitness[idx] = f_opt

        print(f"当前迭代次数{gen + 1}, 迭代精度{best_val}")

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
bounds = np.array([(2.0, 2.0)] + [(0, 5.0)] * 39 + [(0, 1)] * 39)

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=20.5, sigma=8, total_concentration=1.0, x_values=np.arange(1, 41), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 41)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 运行差分进化算法
best_solution, best_fitness, fitness_history = DPADE(objective_global, bounds, pop_size=200, max_gen=4000, hist_size=200, tol=1e-6)
print("全局优化得到的系数k:", {f'k{i}': c for i, c in enumerate(best_solution, start=0)})
print("全局优化的最终精度:", best_fitness)

best_solution = minimize(objective_global, best_solution, method='L-BFGS-B', bounds=bounds, tol=1e-8, options={'maxiter':1000})
optimal_k = best_solution.x
print("最终的系数k:", {f'k{i}': c for i, c in enumerate(optimal_k[:10], start=0)})
print("最终的系数k_inv:", {f'k{i}_inv': c for i, c in enumerate(optimal_k[10:], start=1)})
print("最终精度:", best_solution.fun)

# 使用得到的系数求解
initial_p = [10.0] + [0] * 40
t = np.linspace(0, 1000, 1000)
sol = odeint(equations, initial_p, t, args=(optimal_k,))

visualize_fitness()

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