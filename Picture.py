import numpy as np
import math
from scipy.integrate import odeint
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

# 定义非线性微分方程组
def equations(p, t, k_values):
    dpdt = np.zeros_like(p)
    k = k_values[:20]
    k_inv = k_values[20:]
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * (p[1] ** 2)
    for i in range(2, 20):
        dpdt[i] = k[i - 1] * (p[i - 1] ** 2) + k_inv[i - 1] * p[i + 1] - k_inv[i - 2] * p[i] - k[i] * (p[i] ** 2)
    dpdt[20] = k[19] * (p[19] ** 2) - k_inv[18] * p[20]
    return dpdt

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
    x_values = np.arange(1, 21)
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 假设初始浓度分布
initial_p = np.zeros(21)
initial_p[0] = 10  # 初始浓度 P0 = 10

# 理想最终浓度
mu = 10.5
sigma = 6
scale_factor = 10
concentrations = simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
x_values = [f'P{i}' for i in range(1, 21)]
print("理想最终浓度:", {f"P{i}": c for i, c in enumerate(concentrations, start=1)})

best_solution = {'k0': 1.115564485568162, 'k1': 0.4834945458141331, 'k2': 0.3221554930125466, 'k3': 0.28157713938698875, 'k4': 0.2522645002066048, 'k5': 0.16129907117085096, 'k6': 0.14444019410384334, 'k7': 0.14818632649216418, 'k8': 0.1356496000167094, 'k9': 0.12319960174447865, 'k10': 0.11600196382458658, 'k11': 0.09489492240026193, 'k12': 0.09628400224423128, 'k13': 0.08909084921413671, 'k14': 0.11186019623192459, 'k15': 0.1142878748579884, 'k16': 0.1351133984962806, 'k17': 0.18687123891011442, 'k18': 0.17834158237802444, 'k19': 0.19454081979790092}
k_inv = {'k1_inv': 0.07884944255869816, 'k2_inv': 0.0695160447797974, 'k3_inv': 0.07807724523254629, 'k4_inv': 0.0871300097295713, 'k5_inv': 0.0677329996018549, 'k6_inv': 0.07179053669406994, 'k7_inv': 0.08443867503527003, 'k8_inv': 0.08654127861039182, 'k9_inv': 0.08524609501640573, 'k10_inv': 0.08496210866639049, 'k11_inv': 0.07155118123278026, 'k12_inv': 0.07243527249845134, 'k13_inv': 0.06526034949553097, 'k14_inv': 0.07747673265271753, 'k15_inv': 0.07297231817729116, 'k16_inv': 0.07691626738154421, 'k17_inv': 0.09279987155577112, 'k18_inv': 0.07483917669417857, 'k19_inv': 0.06741287134221452}
best_solution = list(best_solution.values()) + list(k_inv.values())

initial_k = list(best_solution)

t = np.linspace(0, 1000, 1000)
sol = odeint(equations, initial_p, t, args=(initial_k,))

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

# plot_concentration_combined(t, sol)

plt.figure(figsize=(8, 15))
for i in range(21):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P0-P20 Concentration over Time')
plt.grid(True)
plt.show()

plot_k_lnp_curves(pm, best_solution[1:])