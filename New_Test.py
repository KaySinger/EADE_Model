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
    t = np.linspace(0, 1000, 1000)
    # 求解微分方程
    sol = odeint(equations, initial_p, t, args=(k,))
    final_p = sol[-1, :] # 取最终浓度
    # 理想最终浓度
    ideal_p = [0] + list(target_p)
    # 计算误差
    sum_error = np.sum((final_p - ideal_p)**2)

    return sum_error

def rand1bin(pop, F, i):
    r1, r2, r3 = np.random.choice(len(pop), 3, replace=False)
    return pop[r1] + F * (pop[r2] - pop[r3])

def rand2bin(pop, F, i):
    r1, r2, r3, r4 = np.random.choice(len(pop), 4, replace=False)
    return pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r1])


# 定义current-to-pbest变异策略
def de_current_to_pbest_1(X, F, i, pbest):
    NP, D = X.shape
    r1, r2 = np.random.choice(NP, 2, replace=False)
    return X[i] + F * (pbest - X[i]) + F * (X[r1] - X[r2])

def calculate_diversity(fitness):
    """
    计算种群多样性（适应度的标准差）
    """
    return np.std(fitness)

def select_mutation_strategy(pop, fitness, i, F, diversity_threshold=0.1):
    """
    根据种群多样性动态选择变异策略
    """
    diversity = calculate_diversity(fitness)
    if diversity < diversity_threshold:
        # 多样性低，使用 rand2bin
        return rand2bin(pop, F, i)
    else:
        # 多样性高，使用 rand1bin
        return rand1bin(pop, F, i)


def DPADE(func, bounds, pop_size=None, max_gen=None, hist_size=100, tol=1e-6):

    dim = len(bounds)
    archive = []
    H = hist_size
    F_hist, CR_hist = [0.5] * H, [0.5] * H
    hist_idx = 0
    iteration_log = []

    # 初始化种群
    pop = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim))
    fitness = np.apply_along_axis(func, 1, pop)

    # 控制参数自适应机制
    F_min, F_max = 0.4, 0.8  # 缩放因子 F 的范围
    CR_min, CR_max = 0.5, 0.9  # 交叉因子 CR 的范围
    st_max = 20  # 停滞代数的上限
    st_count = np.zeros(pop_size)  # 记录每个个体的停滞代数

    # 固定参数 p
    p_max = 0.1
    p_min = 0.02

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

        # 基于线性分布的参数自适应选择
        p = p_max - (p_max - p_min) * (gen / max_gen)

        # 种群分类
        fitness_sorted_indices = np.argsort(fitness)
        NPG_size = int(pop_size * 0.2)  # 优势种群固定为前20%
        NPG_indices = fitness_sorted_indices[:NPG_size]  # 优势种群
        NPB_indices = fitness_sorted_indices[NPG_size:]  # 劣势种群

        for i in range(pop_size):
            # 根据种群分类和停滞状态调整控制参数
            if i in NPG_indices:
                # 优势种群：使用 SHADE 的参数更新方式
                hist_sample = np.random.randint(0, H)
                F = np.clip(np.random.standard_cauchy() * 0.1 + F_hist[hist_sample], 0, 1)
                CR = np.clip(np.random.normal(CR_hist[hist_sample], 0.1), 0, 1)
            elif st_count[i] >= st_max:
                # 停滞个体：重新生成控制参数
                F = F_min + np.random.rand() * (F_max - F_min)
                CR = CR_min + np.random.rand() * (CR_max - CR_min)
                st_count[i] = 0  # 重置停滞计数器
            else:
                valid_indices = [idx % H for idx in NPG_indices]
                F_elite = np.mean([F_hist[idx] for idx in valid_indices])
                CR_elite = np.mean([CR_hist[idx] for idx in valid_indices])
                F = F_min + np.random.rand() * (F_elite - F_min)
                CR = CR_min + np.random.rand() * (CR_elite - CR_min)

            F_values.append(F)
            CR_values.append(CR)

            # 根据种群分类选择变异策略
            if i in NPG_indices:
                # 优势种群：使用 current-to-pbest 策略
                pbest_size = max(1, int(p * NPG_size))
                pbest_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(pbest_indices)
                pbest = pop[pbest_idx]
                mutant = de_current_to_pbest_1(pop, F, i, pbest)
            else:
                # 劣势种群：根据种群多样性动态选择变异策略
                mutant = select_mutation_strategy(pop, fitness, i, F)

            # 二项交叉
            trial = np.array([mutant[j] if np.random.rand() < CR else pop[i][j] for j in range(dim)])
            trial = np.clip(trial, [b[0] for b in bounds], [b[1] for b in bounds])
            trial_fitness = func(trial)

            # 贪心选择
            if trial_fitness < fitness[i]:
                new_pop.append(trial)
                fitness[i] = trial_fitness
                archive.append(pop[i])
                S_F.append(F)
                S_CR.append(CR)
                st_count[i] = 0  # 重置停滞计数器
            else:
                new_pop.append(pop[i])
                st_count[i] += 1  # 增加停滞计数器

        # 更新历史记忆
        if S_F and S_CR:
            F_hist[hist_idx] = np.mean(S_F) if np.mean(S_F) > 0 else F_hist[hist_idx]
            CR_hist[hist_idx] = np.mean(S_CR) if np.mean(S_CR) > 0 else CR_hist[hist_idx]
            hist_idx = (hist_idx + 1) % H

        # 更新种群
        pop = np.array(new_pop)

        # 每 0.1 * max_gen 次迭代对精英个体使用 L-BFGS-B 局部优化（并行化）
        if gen % int(0.1 * max_gen) == 0 and gen > 0:
            elite_size = max(1, int(pop_size * 0.1))  # 前10%的精英个体
            elite_indices = np.argsort(fitness)[:elite_size]

            # 并行化局部优化
            def local_optimize(idx):
                result = minimize(func, pop[idx], method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
                return idx, result.x, result.fun

            results = Parallel(n_jobs=-1)(delayed(local_optimize)(idx) for idx in elite_indices)

            # 更新精英个体（仅当新解适应度更低时）
            for idx, x_opt, f_opt in results:
                if f_opt < fitness[idx]:  # 仅当新解适应度更低时才更新
                    pop[idx] = x_opt
                    fitness[idx] = f_opt

        print(f"当前迭代次数{gen + 1}, 迭代精度{np.min(fitness)}")

    return pop[np.argmin(fitness)], np.min(fitness), iteration_log

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
bounds = np.array([(2.0, 2.0)] + [(0.01, 10.0)] * 39)

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=20.5, sigma=6, total_concentration=1.0, x_values=np.arange(1, 41), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 41)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 运行差分进化算法
best_solution, best_fitness, fitness_history = DPADE(objective_global, bounds=bounds, pop_size=400, max_gen=4000, hist_size=100, tol=1e-6)
print("全局优化得到的系数k:", {f'k{i}': c for i, c in enumerate(best_solution, start=0)})
print("最终精度:", best_fitness)

visualize_fitness()

# # 梯度优化，进一步提高精度
# print("开始梯度优化")
#
# result_final = minimize(objective_global, best_solution, method='L-BFGS-B', bounds=bounds, tol=1e-8)
# optimal_k = result_final.x
# final_precision = result_final.fun
#
# print("反应系数K是:", {f"k{i}:": c for i, c in enumerate(optimal_k, start=0)})
# print("最终优化精度:", final_precision)

# 使用得到的系数求解
initial_p = [10.0] + [0] * 40
t = np.linspace(0, 1000, 1000)
sol = odeint(equations, initial_p, t, args=(best_solution,))

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