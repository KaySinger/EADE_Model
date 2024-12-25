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
def equations(p, t, k):
    dpdt = np.zeros_like(p)
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * (p[1]**2)
    for i in range(2, 20):
        dpdt[i] = k[i-1] * (p[i-1]**2) - k[i] * (p[i]**2)
    dpdt[20] = k[19] * (p[19]**2)
    return dpdt

# 定义目标函数
def objective_global(k):
    initial_p = [10.0] + [0] * 20
    t = np.linspace(0, 200, 1000)
    # 求解微分方程
    sol = odeint(equations, initial_p, t, args=(k,))
    final_p = sol[-1, :] # 取最终浓度
    # 理想最终浓度
    ideal_p = [0] + list(target_p)
    # 计算误差
    sum_error = np.sum((final_p - ideal_p)**2)
    mse_error = sum_error / len(final_p)

    return sum_error


def shade(func, bounds, pop_size=None, max_gen=None, hist_size=10, tol=1e-6):
    dim = len(bounds)
    archive = []
    H = hist_size
    F_hist, CR_hist = [0.5] * H, [0.5] * H
    hist_idx = 0
    iteration_log = []  # 用于记录每代最优解

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

    # 初始化种群
    pop = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim))
    fitness = np.apply_along_axis(func, 1, pop)

    for gen in range(max_gen):
        F_values, CR_values = [], []
        S_F, S_CR = [], []
        new_pop = []

        # 检查精度终止条件
        best_val = np.min(fitness)
        iteration_log.append(best_val)
        if best_val <= tol:
            print(f"Converged at generation {gen} with precision {best_val:.6e}")
            break

        for i in range(pop_size):
            # 从成功历史记录中采样 F 和 CR
            hist_sample = np.random.randint(0, H)
            F = np.clip(np.random.normal(F_hist[hist_sample], 0.1), 0, 1)
            CR = np.clip(np.random.normal(CR_hist[hist_sample], 0.1), 0, 1)
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
                S_F.append(F)
                S_CR.append(CR)
            else:
                new_pop.append(pop[i])

        # 更新种群和外部存档
        pop = np.array(new_pop)
        archive = archive[-pop_size:]  # 限制存档大小

        # 更新成功历史记录
        if S_F and S_CR:
            F_hist[hist_idx] = np.mean(S_F)
            CR_hist[hist_idx] = np.mean(S_CR)
            hist_idx = (hist_idx + 1) % H

        print(f"当前迭代次数{gen + 1}, 迭代精度{np.min(fitness)}")

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
bounds = np.array([(0.5, 10.0)] + [(0.01, 15.0)] * 19)

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=10.5, sigma=8, total_concentration=1.0, x_values=np.arange(1, 21), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 21)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 运行差分进化算法
best_solution, best_fitness, fitness_history = shade(objective_global, bounds=bounds, pop_size=300, max_gen=1000, hist_size=10, tol=1e-6)
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
initial_p = [10.0] + [0] * 20
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