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
def equations(p, t, k_values):
    dpdt = np.zeros_like(p)
    k = k_values[:30]
    k_inv = k_values[30:]
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * p[1] ** 2
    for i in range(2, 30):
        dpdt[i] = k[i - 1] * p[i - 1] ** 2 + k_inv[i - 1] * p[i + 1] - k_inv[i - 2] * p[i] - k[i] * p[i] ** 2
    dpdt[30] = k[29] * p[29] ** 2 - k_inv[28] * p[30]
    return dpdt

# 定义目标函数
def objective_global(k):
    initial_p = [10.0] + [0] * 30
    t = np.linspace(0, 1000, 1000)
    # 求解微分方程
    sol = odeint(equations, initial_p, t, args=(k,))
    final_p = sol[-1, 1:] # 取最终浓度
    # 理想最终浓度
    ideal_p = list(target_p)
    # 计算误差
    sum_error = np.sum((final_p - ideal_p)**2)

    return sum_error

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=15.5, sigma=6, total_concentration=1.0, x_values=np.arange(1, 31), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 31)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

# 设置变量边界
bounds = np.array([(2.0, 2.0)] + [(0, 2.0)] * 58)

best_solution = {'k0': 2.0, 'k1': 0.027421182986195034, 'k2': 0.03118068764852096, 'k3': 0.02829685650986292, 'k4': 0.02372584207227388,
                 'k5': 0.019742267551924136, 'k6': 0.016505685005949698, 'k7': 0.014056151993950609, 'k8': 0.012248858413409508,
                 'k9': 0.010962554112336667, 'k10': 0.010109051829610707, 'k11': 0.00961676308202308, 'k12': 0.009451764971273443,
                 'k13': 0.009613110744971403, 'k14': 0.010120510300434173, 'k15': 0.011047139544184141, 'k16': 0.01250780120908035,
                 'k17': 0.014706559354856586, 'k18': 0.017971606282967345, 'k19': 0.02283781791823093, 'k20': 0.030194394987498065,
                 'k21': 0.0415430802205436, 'k22': 0.05949461238849267, 'k23': 0.08867277933118613, 'k24': 0.1377748438387926,
                 'k25': 0.22008031903760614, 'k26': 0.36332998816672485, 'k27': 0.6229028327675952, 'k28': 1.0493954119591689,
                 'k29': 1.589325697441975}
best_solution = list(best_solution.values())

back_solution = np.zeros(29)
for i in range(29):
    back_solution[i] = best_solution[i+1] * (target_p[i]**2) / target_p[i+1]

initial_k = list(best_solution) + list(back_solution)
result = minimize(objective_global, initial_k, method='L-BFGS-B', bounds=bounds, tol=1e-8, options={'maxiter':1000})
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