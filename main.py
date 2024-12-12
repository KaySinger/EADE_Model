import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import curve_fit
import math, matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 记录迭代目标函数值和均方误差
objective_values = []
mse_values = []

# 保存每一代的种群
population_history = []

# 初始 mutation 和 recombination 参数
mutation_param = [1.0]  # 使用列表封装以便在回调函数中修改
recombination_param = [0.7]

# 定义微分方程
def equations(p, t, k):
    assert isinstance(k, (list, np.ndarray)), "k should be a list or numpy array"
    dpdt = np.zeros_like(p)
    dpdt[0] = -k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * p[1]**2
    for i in range(2, 10):
        dpdt[i] = k[i-1] * p[i-1]**2 - k[i] * p[i]**2
    dpdt[10] = k[9] * p[9]**2
    return dpdt

# 定义目标函数，鼓励 k 值呈递减趋势
def objective(k):
    # 初始条件
    assert isinstance(k, (list, np.ndarray)), "k should be a list or numpy array"
    initial_p = [1.0] + [0] * 10  # 初始浓度 P0 = 1
    t = np.linspace(0, 200, 1000)
    # 求解微分方程
    sol = odeint(equations, initial_p, t, args=(k,))
    final_p = sol[-1, :]
    # 目标浓度分布
    target_distribution = [0] + list(concentrations)
    # 计算 P1 到 P40 的误差
    error = np.sum((final_p - target_distribution) ** 2)

    mse = error / len(final_p)

    # 记录
    objective_values.append(error)
    mse_values.append(mse)

    return error

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, scale_factor):
    x_values = np.arange(1, 11)
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 局部搜索函数，应用于优秀个体
def local_search(individual):
    # 使用 L-BFGS-B 方法对个体进行局部优化
    result = minimize(
        objective,
        individual,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50}  # 限制局部优化的迭代次数
    )
    return result.x  # 返回优化后的个体

def de_callback(xk, convergence=None):
    global population_history, objective_values, mutation_param

    current_error = objective(xk)
    objective_values.append(current_error)
    population_history.append(np.copy(xk))

    if len(objective_values) > 1:
        change = np.abs(objective_values[-1] - objective_values[-2])
        print(f"迭代次数 {len(objective_values)}: 误差 = {current_error:.4f}, 变化 = {change:.4f}")
        # 动态调整变异率
        if change < 0.01:
            mutation_param[0] = max(0.5, mutation_param[0] - 0.1)

    # 每隔 10 次迭代，对部分优秀个体进行局部优化
    if len(objective_values) % 10 == 0:
        print("开始局部搜索...")
        current_population = population_history[-1]
        for i in range(len(current_population)):
            individual = current_population[i]
            # 确保个体是 numpy 数组
            if not isinstance(individual, np.ndarray):
                individual = np.array(individual)
            # 检查个体的长度
            assert len(individual) == 10, "个体长度不为10"
            # 选择前25%的优秀个体进行局部优化
            if objective(individual) < np.percentile(objective_values, 25):
                optimized_individual = local_search(individual)
                current_population[i] = optimized_individual
        # 更新 xk 为当前种群中的最优个体
        best_individual_index = np.argmin([objective(ind) for ind in current_population])
        xk[:] = current_population[best_individual_index]

def visualize_convergence():
    # 绘制目标函数和均方误差的收敛曲线
    plt.figure(figsize=(15, 8))
    plt.xlabel("Iteration")
    plt.ylabel("Function Convergence")
    plt.title("Mean and Objective Error Convergence")
    plt.plot(objective_values, label='Objective Value (Error)', color='blue')
    plt.plot(mse_values, label='Mean Value (Error)', color='blue')
    plt.grid(True)
    plt.show()

# 局部优化回调函数
def callback(xk):
    current_value = objective(xk)
    objective_values.append(current_value)
    if len(objective_values) > 1:
        change = np.abs(objective_values[-1] - objective_values[-2])
        print(f"迭代次数 {len(objective_values) - 1}: 变化 = {change}, 精度 = {objective_values[-1]:.4f}")


# 假设初始浓度分布
initial_p = np.zeros(11)
initial_p[0] = 1.0  # 初始浓度 P0 = 10

# 理想最终浓度
mu = 5.5
sigma = 5
scale_factor = 1
concentrations = simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
x_values = [f'P{i}' for i in range(1, 11)]
print("理想最终浓度:", {f"P{i}": c for i, c in enumerate(concentrations, start=1)})

# 定义约束
bounds = [(0.5, 10)] + [(0.01, 10)] * 9  # 对k0做出最小化限制，防止其过小

# 定义初始猜测
initial_guess = np.array([0.5, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])  # 替换为你的实际初始猜测值
population_size = 20  # 假设种群规模为15

# 生成初始化种群，将初始猜测值包含在内
init_population = [initial_guess] + [np.random.rand(len(bounds)) for _ in range(population_size - 1)]

# 使用差分进化算法来求解 k
print("开始全局优化")
objective_values.clear()
mse_values.clear()

result = differential_evolution(objective, bounds, strategy='rand1bin', mutation=mutation_param[0], recombination=recombination_param[0],
                                maxiter=1000, popsize=20, tol=1e-6, seed=42, init=np.array(init_population), callback=de_callback)
visualize_convergence()

# 优化后的 k 值
optimal_k = result.x
final_precision = result.fun

print("全局优化的反应系数K是:", {f"k{i}:": c for i, c in enumerate(optimal_k, start=0)})
print("全局优化的精度:", final_precision)

print("开始梯度优化")

objective_values.clear()
mse_values.clear()

result_final = minimize(objective, optimal_k, method='L-BFGS-B', bounds=bounds, callback=callback, tol=1e-8)
optimal_k = result_final.x
final_precision = result_final.fun

print("反应系数K是:", {f"k{i}:": c for i, c in enumerate(optimal_k, start=0)})
print("最终优化精度:", final_precision)

visualize_convergence()

t = np.linspace(0, 200, 1000)
sol = odeint(equations, initial_p, t, args=(optimal_k,))

# 绘制理想稳态浓度曲线
plt.figure(figsize=(15, 8))
plt.xlabel("P-Species")
plt.ylabel("P-Concentrations")
plt.title("Ideal Concentrations and Actual Concentrations")
plt.xticks(range(len(x_values)), x_values, rotation=90)
final_concentrations = sol[-1, 1:]
plt.plot(range(len(x_values)), concentrations, label = 'Ideal Concentrations', marker='o', linestyle='-', color='blue')
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
#
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
