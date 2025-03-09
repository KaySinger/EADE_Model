import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from matplotlib import pyplot as plt

objective_values = []

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, x_values, scale_factor):
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 定义非线性微分方程组
def equations(p, t, k_values):
    dpdt = np.zeros_like(p)
    k = k_values[:30]
    k_inv = k_values[30:]
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * (p[1] ** 2)
    for i in range(2, 30):
        dpdt[i] = k[i - 1] * (p[i - 1] ** 2) + k_inv[i - 1] * p[i + 1] - k_inv[i - 2] * p[i] - k[i] * (p[i] ** 2)
    dpdt[30] = k[29] * (p[29] ** 2) - k_inv[28] * p[30]
    return dpdt

# 修改后的目标函数
def objective_global(k):
    # 正向系数递增性惩罚项
    k_forward = k[1:30]
    penalty = 0.0
    # 计算所有相邻k的递减量，若k[i+1] < k[i]则施加惩罚
    for i in range(len(k_forward) - 1):
        if k_forward[i + 1] < k_forward[i]:
            penalty += (k_forward[i] - k_forward[i + 1]) ** 2  # 平方惩罚项
    penalty_weight = 1e6  # 惩罚权重（根据问题规模调整）
    total_penalty = penalty_weight * penalty

    initial_p = [10.0] + [0] * 30
    t = np.linspace(0, 1000, 1000)  # 时间范围仍到1000，确保包含900后的时间点
    # 求解微分方程
    sol = odeint(equations, initial_p, t, args=(k,))
    # 选取t>=900时的所有解（假设t=1000时有1000个点，索引900对应t=900）
    selected_sol = sol[900:, :]
    # 理想浓度
    ideal_p = np.array([0] + list(target_p))
    # 计算所有选中时间点的误差平方和
    sum_error = np.sum((selected_sol - ideal_p) ** 2)
    return sum_error + total_penalty

def callback(xk):
    current_value = objective_global(xk)
    objective_values.append(current_value)
    if len(objective_values) > 1:
        change = np.abs(objective_values[-1] - objective_values[-2])
        print(f"迭代次数 {len(objective_values) - 1}: 变化 = {change}, 精度 = {objective_values[-1]}")

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=15.5, sigma=6, total_concentration=1.0, x_values=np.arange(1, 31), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 31)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 设置变量边界
bounds = np.array([(1.0, 2.0)] + [(0, 2.0)] * 29 + [(0, 0.5)] * 29)

best_solution = {'k0': 1.0, 'k1': 0.22758393334777322, 'k2': 0.2277138184687888, 'k3': 0.2288582168655483, 'k4': 0.23173618966435, 'k5': 0.23315791460526783, 'k6': 0.23313662255812032, 'k7': 0.23350260502514966, 'k8': 0.2349243966420282, 'k9': 0.23492136118534132, 'k10': 0.23623321298415242, 'k11': 0.23648879867229158, 'k12': 0.2367482815787347, 'k13': 0.23774729945595088, 'k14': 0.23872801394833262, 'k15': 0.2449099045644161, 'k16': 0.2480447070368699, 'k17': 0.2487637876588262, 'k18': 0.2527987290583257, 'k19': 0.25854019361810204, 'k20': 0.25856108279599227, 'k21': 0.27048844432338126, 'k22': 0.27266205130561855, 'k23': 0.27554268284161954, 'k24': 0.28775747416350395, 'k25': 0.29579685093823543, 'k26': 0.3110792067760519, 'k27': 0.4364709572744968, 'k28': 0.8854138196607797, 'k29': 0.9774984721153273,
'k1_inv': 0.004836284943421216, 'k2_inv': 0.006861668346193606, 'k3_inv': 0.013022222973710678, 'k4_inv': 0.01783436745180811, 'k5_inv': 0.025525810061955957, 'k6_inv': 0.03506808903553856, 'k7_inv': 0.04634194204556016, 'k8_inv': 0.059581140434408264, 'k9_inv': 0.0743802753424991, 'k10_inv': 0.09240489502871806, 'k11_inv': 0.10706026299908807, 'k12_inv': 0.1224608213873686, 'k13_inv': 0.1408149577720301, 'k14_inv': 0.1506210566519239, 'k15_inv': 0.16534809361713312, 'k16_inv': 0.17029527321309748, 'k17_inv': 0.1700679189214335, 'k18_inv': 0.16949080455213208, 'k19_inv': 0.16409858451836193, 'k20_inv': 0.15142265666125676, 'k21_inv': 0.14000376920846538, 'k22_inv': 0.12948968206551598, 'k23_inv': 0.10268063717088546, 'k24_inv': 0.09139558940367372, 'k25_inv': 0.06714487349715381, 'k26_inv': 0.06436244308082222, 'k27_inv': 0.0749758546115172, 'k28_inv': 0.10250330723935244, 'k29_inv': 0.10052279983709524}
best_solution = list(best_solution.values())
print(best_solution)

initial_k = best_solution
result = minimize(objective_global, initial_k, method='L-BFGS-B', bounds=bounds, tol=1e-8, callback=callback)
optimal_k = result.x
print("最终的系数k:", {f'k{i}': c for i, c in enumerate(optimal_k[:30], start=0)})
print("最终的系数k_inv:", {f'k{i}_inv': c for i, c in enumerate(optimal_k[30:], start=1)})
print("最终精度:", result.fun)

# 使用得到的系数求解
initial_p = [10.0] + [0] * 30
t = np.linspace(0, 1000, 1000)
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