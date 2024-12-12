import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# 定义列表用于存储
objective_values = []
mse_values = []

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

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=20.5, sigma=8, total_concentration=1.0, x_values=np.arange(1, 41), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 41)]  # 定义图像横坐标
print("理想最终浓度", {f'P{i}': c for i, c in enumerate(target_p, start=1)})

optimal_k = {'k0:': 1.217397996991922, 'k1:': 0.19511163808747403, 'k2:': 0.25068586586014295,
             'k3:': 0.25104364112142775, 'k4:': 0.23442468263097737, 'k5:': 0.20738107718028331,
             'k6:': 0.1780033610584175, 'k7:': 0.15854670705324694, 'k8:': 0.13688858002009102,
             'k9:': 0.12356598036161148, 'k10:': 0.11101042393056586, 'k11:': 0.10145967315987127,
             'k12:': 0.09439475522207255, 'k13:': 0.08890531000079295, 'k14:': 0.08523642498283786,
             'k15:': 0.08327697311737166, 'k16:': 0.08285135781253303, 'k17:': 0.08405898390506536,
             'k18:': 0.08700368156452842, 'k19:': 0.09183722973711833, 'k20:': 0.09898890667231004,
             'k21:': 0.1089325285980028, 'k22:': 0.122452451514972, 'k23:': 0.14073080359716253,
             'k24:': 0.16555127474267142, 'k25:': 0.19928267893601728, 'k26:': 0.24448246068808613,
             'k27:': 0.3074684211835583, 'k28:': 0.3972085160626213, 'k29:': 0.5240700271519626,
             'k30:': 0.7184741029696702, 'k31:': 0.9712246884294911, 'k32:': 1.4108107191519483,
             'k33:': 2.0568669655833345, 'k34:': 3.054808826115762, 'k35:': 4.629927304803268,
             'k36:': 7.096424412569265, 'k37:': 10.89476300225377, 'k38:': 17.0, 'k39:': 24.0}
optimal_k = list(optimal_k.values())

# 使用得到的系数求解
initial_p = [10.0] + [0] * 40
t = np.linspace(0, 200, 1000)
sol = odeint(equations, initial_p, t, args=(optimal_k,))

final_precision = np.sum((target_p - sol[-1, 1:])**2)
print("最终收敛度", final_precision)

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