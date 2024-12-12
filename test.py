import numpy as np
import math
from scipy.integrate import odeint
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

# 记录迭代目标函数值和均方误差
objective_values = []
mse_values = []

# 定义微分方程
def equations(p, t, k):
    dpdt = np.zeros_like(p)
    dpdt[0] = -k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * p[1]**2
    for i in range(2, 40):
        dpdt[i] = k[i-1] * p[i-1]**2 - k[i] * p[i]**2
    dpdt[40] = k[39] * p[39]**2
    return dpdt

# 定义目标函数，鼓励 k 值呈递减趋势
def objective(k):
    # 初始条件
    initial_p = [10] + [0] * 40  # 初始浓度 P0 = 10
    t = np.linspace(0, 200, 1000)
    # 求解微分方程
    sol = odeint(equations, initial_p, t, args=(k,))
    final_p = sol[-1, 1:]
    # 目标浓度分布
    target_distribution = list(concentrations)
    # 计算 P1 到 P40 的误差
    error = np.sum((final_p - target_distribution) ** 2)

    mse = error / len(final_p)
    # 正则化法则
    # penalty = np.sum(np.maximum(0, -np.diff(k))**2)
    # alpha = 1.0

    # 记录
    objective_values.append(error)
    mse_values.append(mse)

    return error


def plot_concentration_combined(t, sol):
    # 创建一个包含 5 行 1 列的子图布局
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))  # figsize 为 (宽, 高)

    # 绘制 P0-P10 的浓度变化
    for i in range(11):
        axs[0].plot(t, sol[:, i], label=f'P{i}')
    axs[0].set_title('P0-P10 Concentration')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Concentration')
    axs[0].legend()  # 调整图例位置
    axs[0].grid(True)

    # 绘制 P11-P20 的浓度变化
    for i in range(11, 21):
        axs[1].plot(t, sol[:, i], label=f'P{i}')
    axs[1].set_title('P11-P20 Concentration')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Concentration')
    axs[1].legend()
    axs[1].grid(True)

    # 绘制 P21-P30 的浓度变化
    for i in range(21, 31):
        axs[2].plot(t, sol[:, i], label=f'P{i}')
    axs[2].set_title('P21-P30 Concentration')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Concentration')
    axs[2].legend()
    axs[2].grid(True)

    # 绘制 P31-P40 的浓度变化
    for i in range(31, 41):
        axs[3].plot(t, sol[:, i], label=f'P{i}')
    axs[3].set_title('P31-P40 Concentration')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Concentration')
    axs[3].legend()
    axs[3].grid(True)

    # 自动调整布局，避免重叠
    plt.tight_layout()
    plt.show()

# 参数拟合
pm = [math.log(2**(i+1)) for i in range(39)]

# 定义lnk = a * lnp ^ x的模型
def model(P, a, x):
    return a * P**x

def plot_k_lnp_curves(pm, optimal_k):
    diffs = np.diff(optimal_k)

    # 找到变化率最大的点作为分界点
    split_index = max(np.argmax(np.abs(diffs)) + 1, 5)

    # 分别拟合前后数据
    popt1, _ = curve_fit(model, pm[:split_index], optimal_k[:split_index], maxfev=1000)
    popt2, _ = curve_fit(model, pm[split_index:], optimal_k[split_index:], maxfev=1000)
    # 整体拟合
    popt_all, _ = curve_fit(model, pm, optimal_k, maxfev=1000)

    # 拟合得到的参数
    a1, x1 = popt1
    a2, x2 = popt2
    a_all, x_all = popt_all
    print(f"前半部分拟合参数: a = {a1}, x = {x1}")
    print(f"后半部分拟合参数: a = {a2}, x = {x2}")
    print(f"整体拟合参数: a = {a_all}, x = {x_all}")

    # 使用拟合参数绘制拟合曲线
    P_fit1 = np.linspace(min(pm[:split_index]), max(pm[:split_index]), 100)
    P_fit2 = np.linspace(min(pm[split_index:]), max(pm[split_index:]), 100)
    P_fit_all = np.linspace(min(pm), max(pm), 100)
    k_fit1 = model(P_fit1, *popt1)
    k_fit2 = model(P_fit2, *popt2)
    k_fit_all = model(P_fit_all, *popt_all)

    # 创建子图
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    # 绘制前半部分拟合
    axs[0].scatter(pm[:split_index], optimal_k[:split_index], label='Natural data')
    axs[0].plot(P_fit1, k_fit1, color='red', label=f'k = {a1:.2f} * ln(2^n)^{x1:.2f}')
    axs[0].set_xlabel('polymer')
    axs[0].set_ylabel('k')
    axs[0].legend()
    axs[0].set_title('front curve fitting')
    axs[0].grid(True)

    # 绘制后半部分拟合
    axs[1].scatter(pm[split_index:], optimal_k[split_index:], label='Natural data')
    axs[1].plot(P_fit2, k_fit2, color='blue', label=f'k = {a2:.2f} * ln(2^n)^{x2:.2f}')
    axs[1].set_xlabel('polymer')
    axs[1].set_ylabel('k')
    axs[1].legend()
    axs[1].set_title('behind curve fitting')
    axs[1].grid(True)

    # 绘制前后加在一起的拟合
    axs[2].scatter(pm, optimal_k, label='Natural data')
    axs[2].plot(P_fit1, k_fit1, color='red', label=f'front: k = {a1:.2f} * ln(2^n)^{x1:.2f}')
    axs[2].plot(P_fit2, k_fit2, color='blue', label=f'behind: k = {a2:.2f} * ln(2^n)^{x2:.2f}')
    axs[2].set_xlabel('polymer')
    axs[2].set_ylabel('k')
    axs[2].legend()
    axs[2].set_title('combined curve fitting')
    axs[2].grid(True)

    # 绘制整体拟合
    axs[3].scatter(pm, optimal_k, label='Natural data')
    axs[3].plot(P_fit_all, k_fit_all, color='green', label=f'all: k = {a_all:.2f} * ln(2^n)^{x_all:.2f}')
    axs[3].set_xlabel('polymer')
    axs[3].set_ylabel('k')
    axs[3].legend()
    axs[3].set_title('overall curve fitting')
    axs[3].grid(True)

    # 调整子图布局
    plt.tight_layout()
    plt.show()

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, scale_factor):
    x_values = np.arange(1, 41)
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 局部优化回调函数
def callback(xk):
    current_value = objective(xk)
    objective_values.append(current_value)
    if len(objective_values) > 1:
        change = np.abs(objective_values[-1] - objective_values[-2])
        print(f"迭代次数 {len(objective_values) - 1}: 变化 = {change}, 精度 = {objective_values[-1]}")

# 假设初始浓度分布
initial_p = np.zeros(41)
initial_p[0] = 10  # 初始浓度 P0 = 10

# 理想最终浓度
mu = 20.5
sigma = 8
scale_factor = 10
concentrations = simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
x_values = [f'P{i}' for i in range(1, 41)]
print("理想最终浓度:", {f"P{i}": c for i, c in enumerate(concentrations, start=1)})

# 定义约束
bounds = [(0.5, 10)] + [(0.01, None)] * 39  # 对k0做出最小化限制，防止其过小
constraints = {'type': 'ineq', 'fun': lambda k: k[1] - k[0]}

optimal_k = {'k0:': 1.2172602014196947, 'k1:': 0.1934035914469369, 'k2:': 0.25045540569692953,
             'k3:': 0.25144598438019355, 'k4:': 0.23206605044260536, 'k5:': 0.20560900900387588,
             'k6:': 0.18056311883156892, 'k7:': 0.15806090080216914, 'k8:': 0.1389178936322163,
             'k9:': 0.12331100791784368, 'k10:': 0.1113245694477911, 'k11:': 0.1014502045675814,
             'k12:': 0.09408618233319259, 'k13:': 0.08877483047633453, 'k14:': 0.08518023785617802,
             'k15:': 0.08322635591728118, 'k16:': 0.08286008338459744, 'k17:': 0.08405512800593015,
             'k18:': 0.08699439309974785, 'k19:': 0.0918542859140179, 'k20:': 0.09897876045584161,
             'k21:': 0.10890965894272928, 'k22:': 0.12243372523167667, 'k23:': 0.14067192393563266,
             'k24:': 0.16528153590796674, 'k25:': 0.1986573168511293, 'k26:': 0.24446727282340863,
             'k27:': 0.3076782892554301, 'k28:': 0.39696521235611604, 'k29:': 0.5241343759630537,
             'k30:': 0.7084928661748607, 'k31:': 0.9777065788883984, 'k32:': 1.4069943672684238,
             'k33:': 2.054376704026014, 'k34:': 3.0536493682198924, 'k35:': 4.629409240188316,
             'k36:': 7.096255777703563, 'k37:': 10.894731813797177, 'k38:': 16.999938415579383,
             'k39:': 23.99994486145106}
optimal_k = list(optimal_k.values())

result_final = minimize(objective, optimal_k, method='L-BFGS-B',bounds=bounds, tol=1e-8, callback=callback, options={'maxiter':1000})
optimal_k = result_final.x
final_precision = result_final.fun

print("反应系数K是:", {f"k{i}:": c for i, c in enumerate(optimal_k, start=0)})
print("最终优化精度:", final_precision)

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

plot_concentration_combined(t, sol)

plt.figure(figsize=(8, 15))
for i in range(41):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P0-P40 Concentration over Time')
plt.grid(True)
plt.show()

plot_k_lnp_curves(pm, optimal_k[1:])