import torch
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
from matplotlib import pyplot as plt


# 模拟正态分布生成目标浓度
def simulate_normal_distribution(mu, sigma, total_concentration, x_values, scale_factor):
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 定义ODE系统
class ODESystem(torch.nn.Module):
    def __init__(self, k_values):
        super().__init__()
        self.k_forward = torch.nn.Parameter(k_values[:20])
        self.k_inverse = torch.nn.Parameter(k_values[20:])

    def forward(self, t, p):
        dpdt = torch.zeros_like(p)
        dpdt[0] = -self.k_forward[0] * p[0]
        dpdt[1] = self.k_forward[0] * p[0] + self.k_inverse[0] * p[2] - self.k_forward[1] * (p[1] ** 2)
        for i in range(2, 20):
            dpdt[i] = (
                    self.k_forward[i - 1] * (p[i - 1] ** 2) +
                    self.k_inverse[i - 1] * p[i + 1] -
                    self.k_inverse[i - 2] * p[i] -
                    self.k_forward[i] * (p[i] ** 2)
            )
        dpdt[20] = self.k_forward[19] * (p[19] ** 2) - self.k_inverse[18] * p[20]
        return dpdt


# 目标函数
def objective_global(k_tensor):
    k_tensor = torch.nn.functional.softplus(k_tensor)
    k_forward = k_tensor[:20]
    penalty = torch.sum(torch.relu(k_forward[:-1] - k_forward[1:]))
    total_penalty = 1e3 * penalty

    initial_p = torch.tensor([10.0] + [0.0] * 20, dtype=torch.float32)
    t_eval = torch.linspace(0, 1000, 1000)

    try:
        sol = odeint(
            ODESystem(k_tensor),
            initial_p,
            t_eval,
            method='rk4',
            rtol=1e-6,
            atol=1e-8
        )
    except:
        return torch.tensor(1e12, requires_grad=True)

    if torch.any(torch.isnan(sol)):
        print("检测到NaN！")
        return torch.tensor(1e12, requires_grad=True)

    steady_state = sol[900:, 1:]
    error = torch.mean((steady_state - target_p_tensor) ** 2)

    return error + total_penalty


# 初始化参数
best_solution = np.array([1.0, 0.0054, 0.0062, 0.0068, 0.0070, 0.0071, 0.0072, 0.0075, 0.0129, 0.0139,
                          0.0144, 0.0147, 0.0211, 0.0241, 0.0252, 0.0330, 0.0409, 0.0503, 0.0731, 0.0852,
                          0.0002, 3.6896e-05, 2.5627e-06, 2.0832e-06, 1.3370e-05, 5.8144e-05, 0.000219, 0.0037, 0.0045, 0.0048,
                          0.0044, 0.0083, 0.0087, 0.0067, 0.0082, 0.0080, 0.0066, 0.0099, 0.0099])  # 替换为DE优化结果
initial_k = torch.tensor(best_solution, dtype=torch.float32, requires_grad=True)

# 定义优化器
optimizer = optim.Adam([initial_k], lr=1e-5)

# 训练循环
for epoch in range(1000):
    optimizer.zero_grad()
    loss = objective_global(initial_k)

    if torch.isnan(loss):
        print("损失为NaN，跳过本轮迭代")
        continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_([initial_k], max_norm=1.0)
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 可视化损失曲线
plt.figure()
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 使用优化后的参数求解ODE
with torch.no_grad():
    final_k = torch.nn.functional.softplus(initial_k)  # 应用参数约束
    t_eval = torch.linspace(0, 1000, 1000)
    initial_p = torch.tensor([10.0] + [0.0]*20, dtype=torch.float32)
    sol = odeint(ODESystem(final_k), initial_p, t_eval, method='rk4')

# 转换为numpy数组用于绘图
t_np = t_eval.cpu().numpy()
sol_np = sol.cpu().numpy()

# ----------------------
# 1. 理想浓度与实际浓度对比
# ----------------------
plt.figure(figsize=(12, 6))
x_labels = [f'P{i}' for i in range(1, 21)]

# 理想浓度（添加零值使长度对齐）
ideal_padded = np.concatenate([[0], target_p])  # 添加P0=0

# 实际最终浓度
actual_final = sol_np[-1, :]

# 绘制对比曲线
plt.plot(ideal_padded, 'bo-', label='Ideal Concentrations')
plt.plot(actual_final, 'rx--', label='Actual Concentrations')

# 图表修饰
plt.xticks(range(21), [f'P{i}' for i in range(21)], rotation=45)
plt.xlabel('Chemical Species')
plt.ylabel('Concentration')
plt.title('Steady-State Concentration Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------
# 2. P0-P20浓度变化曲线
# ----------------------
plt.figure(figsize=(15, 8))

# 绘制P0-P20浓度曲线
for i in range(21):
    plt.plot(t_np, sol_np[:, i], label=f'P{i}')

# 图表修饰
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration Dynamics of P0-P20')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True)
plt.tight_layout()
plt.show()