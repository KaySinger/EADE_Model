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
def equations_1(p, t, k):
    dpdt = np.zeros_like(p)
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * p[1] ** 2
    for i in range(2, 20):
        dpdt[i] = k[i - 1] * p[i - 1] ** 2 - k[i] * p[i] ** 2
    dpdt[20] = k[19] * p[19] ** 2
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
    initial_p = [10.0] + [0] * 20
    t = np.linspace(0, 200, 1000)
    # 求解微分方程
    sol = odeint(equations_1, initial_p, t, args=(k,))
    final_p = sol[-1, :] # 取最终浓度
    # 理想最终浓度
    ideal_p = [0] + list(target_p)
    # 计算误差
    sum_error = np.sum((final_p - ideal_p)**2)
    mse_error = sum_error / len(final_p)

    return sum_error

# 定义目标函数
def objective_local(k):
    initial_p = [10.0] + [0] * 20
    t = np.linspace(0, 200, 1000)
    # 求解微分方程
    sol = odeint(equations_1, initial_p, t, args=(k,))
    final_p = sol[-1, :] # 取最终浓度
    # 理想最终浓度
    ideal_p = [0] + list(target_p)
    # 计算误差
    sum_error = np.sum((final_p - ideal_p)**2)
    mse_error = sum_error / len(final_p)

    return mse_error


def parameter_adaptive_de(
        fobj, bounds, NP, max_iter=1000, tol=1e-6, F_init=0.5, CR_init=0.5,
        p_min=0.1, p_max=0.9, initial_scale=15, min_scale=5):
    """
    参数自适应差分进化算法，基于 SHADE-like 记忆机制和 current-to-pbest/1 策略。
    """
    dim = len(bounds)
    NP = initial_scale * NP
    min_NP = min_scale * NP
    pop = np.random.rand(NP, dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    fitness = np.array([fobj(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    best_fitness = fitness[best_idx]
    fitness_history = []

    archive = []
    mu_F, mu_CR = F_init, CR_init
    H = 5  # 历史记录大小
    M_CR = np.full(H, 0.5)  # CR 的历史记录
    M_F = np.full(H, 0.5)   # F 的历史记录
    k = 0  # 历史记录索引

    for gen in range(max_iter):
        S_CR, S_F = [], []
        p = p_max - (p_max - p_min) * gen / max_iter  # 线性调整 p
        prev_best_fitness = best_fitness

        for i in range(len(pop)):
            # 生成 F 和 CR
            R_i = np.random.randint(0, H)
            CR_i = np.clip(np.random.normal(M_CR[R_i], 0.1), 0, 1)
            F_i = np.clip(np.random.standard_cauchy() * 0.1 + M_F[R_i], 0, 1)

            # 选择 pbest 个体
            sorted_indices = np.argsort(fitness)
            top_p_indices = sorted_indices[:max(1, int(p * len(pop)))]
            pbest = pop[np.random.choice(top_p_indices)]

            # 变异：current-to-pbest/1
            candidates = [idx for idx in range(len(pop)) if idx != i]
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            mutant = pop[i] + F_i * (pbest - pop[i]) + F_i * (pop[r1] - pop[r2])

            # 交叉
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, pop[i])

            # 边界处理
            trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

            # 选择
            f_trial = fobj(trial)
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                S_CR.append(CR_i)
                S_F.append(F_i)
                archive.append(trial)
                if f_trial < best_fitness:
                    best = trial
                    best_fitness = f_trial

        # 更新 CR 和 F 的历史记录
        if S_CR and S_F:
            M_CR[k] = np.mean(S_CR)
            M_F[k] = np.sum(np.array(S_F) ** 2) / np.sum(S_F)  # Lehmer 平均值
            k = (k + 1) % H

        # 动态调整种群大小
        new_NP = max(int(min_NP + (initial_scale * NP - min_NP) * (1 - gen / max_iter)), min_NP)
        if new_NP < len(pop):
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices[:new_NP]]
            fitness = fitness[sorted_indices[:new_NP]]

        # 动态调整存档大小
        new_arc_size = int(new_NP * 1.0)  # arcRate = 1.0
        if len(archive) > new_arc_size:
            archive = archive[:new_arc_size]

        # 检查终止条件
        if best_fitness <= tol:
            print(f"Converged at generation {gen} with precision {best_fitness:.6e}")
            break

        print(f"当前迭代次数 {gen+1}, 迭代精度 {best_fitness}")
        fitness_history.append(best_fitness)

    return best, best_fitness, fitness_history

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
bounds_1 = np.array([(1, 2)] + [(0.01, 20)] * 19)
bounds_2 = np.array([(0, None)] * 79)

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=10.5, sigma=8, total_concentration=1.0, x_values=np.arange(1, 21), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 21)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 运行差分进化算法
best_solution, best_fitness, fitness_history = parameter_adaptive_de(objective_global, bounds=bounds_1, NP=20, max_iter=1000, tol=1e-6, F_init=0.5, CR_init=0.5,
        p_min=0.1, p_max=0.9, initial_scale=15, min_scale=5)
print("全局优化得到的系数k:", {f'k{i}': c for i, c in enumerate(best_solution, start=0)})
print("最终精度:", best_fitness)

# 使用得到的系数求解
initial_p = [10.0] + [0] * 20
t = np.linspace(0, 200, 1000)
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

# # 梯度优化，进一步提高精度
# print("开始梯度优化")
# k = np.zeros(40)
# k_inv = np.zeros(39)
# for i in range(39):
#     k[i] = best_solution[i]
#     k_inv[i] = (k[i+1] * target_p[i]**2) / target_p[i+1]
# initial_condition = np.concatenate(k, k_inv)
# result_final = minimize(objective_local, initial_condition, method='L-BFGS-B', bounds=bounds_2, tol=1e-8)
# optimal_k = result_final.x
# final_precision = result_final.fun
#
# print("反应系数K是:", {f"k{i}:": c for i, c in enumerate(optimal_k, start=0)})
# print("最终优化精度:", final_precision)
#
# # 使用得到的系数求解
# initial_p = [10.0] + [0] * 40
# t = np.linspace(0, 500, 1000)
# sol = odeint(equations_2, initial_p, t, args=(optimal_k,))

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