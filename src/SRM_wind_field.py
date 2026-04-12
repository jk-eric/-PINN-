import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from scipy.signal import welch, csd

# -------------------------- 第一步：设定毕设风场核心参数 --------------------------
# 1. 风场物理特性参数（完全匹配你的毕设）
z_ref = 10.0  # 参考高度 (m)
U_ref = 30.0  # 参考高度平均风速 (m/s)
alpha = 0.16  # 地表粗糙度指数
Cy = 0.12  # 横风向相干衰减系数
Cz = 0.10  # 竖风向相干衰减系数
K_davenport = 0.004  # 调整地表阻力系数，提升湍流强度到0.10-0.25范围

# 2. 数值仿真参数（完全匹配你的毕设：400秒）
T = 400  # 总仿真时长 (s)
fs = 1.0  # 采样频率 (Hz)
N = int(T * fs)  # 总采样点数
df = 1.0 / T  # 频率分辨率 (Hz)

# 3. 空间网格设置（严格适配你的要求：yx方向，x默认相同=0，11×11）
grid_size = 11  # 11×11网格
x_fixed = 0.0  # x默认相同，固定为0
y_range = np.linspace(-50, 50, grid_size)  # y方向：-50m到50m（横风向）
z_range = np.linspace(50, 200, grid_size)  # z方向：50m到200m（低空分层）

# 生成yx方向的二维网格（indexing='ij'表示y优先）
Y, Z = np.meshgrid(y_range, z_range, indexing='ij')

# 直接将二维网格flatten为一维数组（y和z都是121个元素）
y = Y.flatten()
z = Z.flatten()

# 用np.full_like(y, x_fixed)生成x，确保x和y、z长度完全一致（都是121）
x = np.full_like(y, x_fixed)

# 拼接成空间网格坐标矩阵
space_grid = np.column_stack((x, y, z))
n_nodes = space_grid.shape[0]  # 空间节点总数：11×11=121

print(f"空间网格已生成：{grid_size}×{grid_size} = {n_nodes}个节点")
print(f"排列方式：yx方向（y优先），x固定为{x_fixed}m")
print(f"y范围：{y_range[0]:.1f}m ~ {y_range[-1]:.1f}m，z范围：{z_range[0]:.1f}m ~ {z_range[-1]:.1f}m")

# 4. 固定随机种子，保证数据集可复现
np.random.seed(42)


# -------------------------- 第二步：定义核心物理模型（彻底修复批量f和批量U10支持） --------------------------
def davenport_spectrum(f, U10):
    """
    彻底修复后的Davenport顺风向功率谱密度函数
    支持：
    1. f为单个值，U10为单个值 → 返回单个值
    2. f为数组，U10为单个值 → 返回和f长度一致的数组
    3. f为单个值，U10为数组 → 返回和U10长度一致的数组
    输入：f - 频率（单个值或数组），U10 - 10m参考高度平均风速（单个值或数组）
    输出：S_u - 功率谱密度（形状与输入匹配）
    """
    # 情况1：f和U10都是单个值
    if np.isscalar(f) and np.isscalar(U10):
        if f <= 1e-6:
            return 0.0
        x = 1200 * f / U10
        S_u = 4 * K_davenport * U10 ** 2 * x / (f * (1 + x ** 2) ** (4 / 3))
        return S_u

    # 情况2：f是数组，U10是单个值（功率谱验证用）
    elif not np.isscalar(f) and np.isscalar(U10):
        S_u = np.zeros_like(f, dtype=np.float64)
        mask = f > 1e-6
        f_valid = f[mask]
        x = 1200 * f_valid / U10
        S_u[mask] = 4 * K_davenport * U10 ** 2 * x / (f_valid * (1 + x ** 2) ** (4 / 3))
        return S_u

    # 情况3：f是单个值，U10是数组（Cholesky分解用）
    elif np.isscalar(f) and not np.isscalar(U10):
        S_u = np.zeros_like(U10, dtype=np.float64)
        if f <= 1e-6:
            return S_u
        x = 1200 * f / U10
        S_u = 4 * K_davenport * U10 ** 2 * x / (f * (1 + x ** 2) ** (4 / 3))
        return S_u

    # 情况4：f和U10都是数组（暂不需要，保留扩展性）
    else:
        raise ValueError("暂不支持f和U10同时为数组的情况")


def davenport_coherence(f, dy, dz, U_avg):
    """
    Davenport空间相干函数
    输入：f - 频率 (Hz)，dy - 横风向距离差 (m)，dz - 竖风向距离差 (m)，U_avg - 两节点平均风速 (m/s)
    输出：gamma - 相干系数 (无量纲，0-1)
    """
    numerator = f * np.sqrt((Cy * dy) ** 2 + (Cz * dz) ** 2)
    gamma = np.exp(-numerator / U_avg)
    return gamma


# -------------------------- 第三步：计算各节点平均风速（幂律风剖面） --------------------------
U_z = U_ref * (z / z_ref) ** alpha  # 每个节点所在高度的时均风速

# -------------------------- 第四步：构建频率向量 --------------------------
f = np.linspace(df, fs / 2, N // 2)
n_freq = len(f)

# -------------------------- 第五步：逐频率点构建互谱密度矩阵 + Cholesky分解 --------------------------
# 初始化Cholesky下三角矩阵：[节点数, 节点数, 频率数]
L = np.zeros((n_nodes, n_nodes, n_freq), dtype=np.complex128)

# 预计算所有节点对的距离差和平均风速（向量化优化，x固定为0，忽略dx）
dy_matrix = y[:, np.newaxis] - y[np.newaxis, :]
dz_matrix = z[:, np.newaxis] - z[np.newaxis, :]
U_avg_matrix = (U_z[:, np.newaxis] + U_z[np.newaxis, :]) / 2

for k in range(n_freq):
    f_k = f[k]

    # 1. 构建当前频率点的互谱密度矩阵(CSD矩阵)
    # 填充对角元：自功率谱（批量计算121个节点的自谱）
    S_diag = davenport_spectrum(f_k, U_z)
    S = np.diag(S_diag).astype(np.complex128)  # 先构建对角矩阵

    # 填充非对角元：互功率谱 = sqrt(自谱1*自谱2) * 相干函数（向量化优化）
    gamma_matrix = davenport_coherence(f_k, dy_matrix, dz_matrix, U_avg_matrix)
    S_offdiag = np.sqrt(S_diag[:, np.newaxis] * S_diag[np.newaxis, :]) * gamma_matrix
    # 只保留上三角非对角元，下三角用共轭对称填充（避免重复计算）
    upper_tri_idx = np.triu_indices(n_nodes, k=1)
    S[upper_tri_idx] = S_offdiag[upper_tri_idx]
    lower_tri_idx = np.tril_indices(n_nodes, k=-1)
    S[lower_tri_idx] = np.conj(S.T[lower_tri_idx])

    # 2. Cholesky下三角分解：S = L @ L^H
    S_reg = S + 1e-8 * np.eye(n_nodes)  # 加极小正则化项保证矩阵正定
    L[:, :, k] = cholesky(S_reg, lower=True)

    # 进度提示
    if (k + 1) % 50 == 0:
        print(f"Cholesky分解进度：{k + 1}/{n_freq}")

# -------------------------- 第六步：生成独立随机相位 --------------------------
phi = 2 * np.pi * np.random.rand(n_nodes, n_freq)

# -------------------------- 第七步：时域风速序列合成 --------------------------
u_pulse = np.zeros((n_nodes, N))
t = np.arange(N) / fs

for i in range(n_nodes):
    for k_idx in range(n_freq):
        f_k = f[k_idx]
        # 向量化求和，提升计算效率
        sum_term = np.sum(L[i, :i + 1, k_idx] * np.sqrt(2 * df) * np.exp(1j * phi[:i + 1, k_idx]))
        u_pulse[i, :] += np.real(sum_term * np.exp(1j * 2 * np.pi * f_k * t))

    # 进度提示
    if (i + 1) % 20 == 0:
        print(f"风速合成进度：{i + 1}/{n_nodes}")

# 叠加平均风速
U_wind = U_z[:, np.newaxis] + u_pulse


# -------------------------- 第八步：数据预处理 --------------------------
def preprocess_wind_data(U_wind):
    U_processed = U_wind.copy()
    n_nodes, N = U_processed.shape

    for i in range(n_nodes):
        mean_i = np.mean(U_processed[i, :])
        std_i = np.std(U_processed[i, :])
        lower = mean_i - 3 * std_i
        upper = mean_i + 3 * std_i
        mask = (U_processed[i, :] < lower) | (U_processed[i, :] > upper)
        U_processed[i, mask] = np.nan

        t_valid = t[~mask]
        U_valid = U_processed[i, ~mask]
        U_processed[i, :] = np.interp(t, t_valid, U_valid)

    U_mean = np.mean(U_processed, axis=1, keepdims=True)
    U_std = np.std(U_processed, axis=1, keepdims=True)
    U_norm = (U_processed - U_mean) / U_std

    return U_processed, U_norm, U_mean, U_std


U_processed, U_norm, U_mean, U_std = preprocess_wind_data(U_wind)

# -------------------------- 第九步：物理特性验证 --------------------------
print("\n" + "=" * 50)
print("SRM风场生成完成，开始物理特性验证...")
print("=" * 50)

# 1. 风剖面验证（取y=0的中心列，共11个不同高度的节点）
center_col_idx = np.where((y == 0))[0]  # y=0的所有节点索引
U_sim_mean_col = np.mean(U_processed[center_col_idx, :], axis=1)
U_theory_mean_col = U_z[center_col_idx]
error_wind_profile = np.abs((U_sim_mean_col - U_theory_mean_col) / U_theory_mean_col) * 100
print("\n【风剖面验证（y=0中心列，11个高度）】")
for i, idx in enumerate(center_col_idx):
    print(
        f"节点{idx} (z={z[idx]:.1f}m): 理论值={U_theory_mean_col[i]:.2f}m/s, 模拟值={U_sim_mean_col[i]:.2f}m/s, 相对误差={error_wind_profile[i]:.2f}%")

# 2. 湍流强度验证（取所有节点的平均值和中心列）
u_pulse_sim = U_processed - U_sim_mean_col[:, np.newaxis] if len(center_col_idx) == n_nodes else U_processed - np.mean(
    U_processed, axis=1, keepdims=True)
I_u_sim = np.std(u_pulse_sim, axis=1) / (
    U_sim_mean_col if len(center_col_idx) == n_nodes else np.mean(U_processed, axis=1))
print("\n【湍流强度验证】")
print(f"所有节点平均湍流强度: {np.mean(I_u_sim):.3f} (预设低空范围: 0.10-0.25)")
print(f"y=0中心列平均湍流强度: {np.mean(I_u_sim[center_col_idx]):.3f}")

# 3. 功率谱验证（取y=0、z=125m的中心节点）
center_node_idx = np.where((y == 0) & (np.isclose(z, 125.0, atol=1.0)))[0][0]
f_welch, S_u_sim = welch(U_processed[center_node_idx, :], fs=fs, nperseg=256, noverlap=128)
# 计算中心节点高度的理论平均风速
U_center_node = U_ref * (125.0 / z_ref) ** alpha
# 调用修复后的函数，传入数组f_welch和单个值U_center_node
S_u_theory = davenport_spectrum(f_welch, U_center_node)

# 4. 相干函数验证（取中心节点和y=10m、z=125m的相邻节点）
neighbor_node_idx = np.where((np.isclose(y, 10.0, atol=1.0)) & (np.isclose(z, 125.0, atol=1.0)))[0][0]


def calc_coherence(u1, u2, fs, nperseg=256, noverlap=128):
    """修复后的相干函数计算：用csd计算互谱密度"""
    f, Pxx = welch(u1, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f, Pyy = welch(u2, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f, Pxy = csd(u1, u2, fs=fs, nperseg=nperseg, noverlap=noverlap)
    gamma = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    return f, gamma


f_coh, gamma_sim = calc_coherence(U_processed[center_node_idx, :], U_processed[neighbor_node_idx, :], fs=fs)
dy_01 = y[center_node_idx] - y[neighbor_node_idx]
dz_01 = z[center_node_idx] - z[neighbor_node_idx]
U_avg_01 = (U_z[center_node_idx] + U_z[neighbor_node_idx]) / 2
gamma_theory = davenport_coherence(f_coh, dy_01, dz_01, U_avg_01)

# -------------------------- 第十步：可视化（适配yx方向网格） --------------------------
# -------------------------- 第十步：可视化（修复字体警告，完美显示负号、平方符号） --------------------------
# 修复matplotlib中文字体和负号显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 优先用微软雅黑，兼容性最好
plt.rcParams['axes.unicode_minus'] = False  # 强制使用ASCII负号，避免Unicode负号显示问题
plt.rcParams['mathtext.fontset'] = 'dejavusans'  # 数学公式用dejavusans字体，完美显示平方、指数等符号

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1：中心节点风速时程
axes[0, 0].plot(t, U_processed[center_node_idx, :], label='模拟风速', linewidth=0.8)
axes[0, 0].axhline(y=np.mean(U_processed[center_node_idx, :]), color='r', linestyle='--', label=f'平均风速: {np.mean(U_processed[center_node_idx, :]):.2f}m/s')
axes[0, 0].set_xlabel('时间 (s)', fontsize=12)
axes[0, 0].set_ylabel('风速 (m/s)', fontsize=12)
axes[0, 0].set_title(f'中心节点 (y=0m, z=125m) 顺风向风速时程', fontsize=14)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# 图2：yx方向网格风速云图（t=0时刻）
U_grid_t0 = U_processed[:, 0].reshape(grid_size, grid_size)
im = axes[0, 1].contourf(Y, Z, U_grid_t0, cmap='viridis', levels=20)
axes[0, 1].set_xlabel('y (m)', fontsize=12)
axes[0, 1].set_ylabel('z (m)', fontsize=12)
axes[0, 1].set_title(f'yx方向网格风速云图 (t=0s, x=0m)', fontsize=14)
cbar = plt.colorbar(im, ax=axes[0, 1])
cbar.set_label('风速 (m/s)', fontsize=12)

# 图3：功率谱验证
axes[1, 0].loglog(f_welch, S_u_sim, label='模拟功率谱', linewidth=0.8)
axes[1, 0].loglog(f_welch, S_u_theory, 'r--', label='Davenport理论谱', linewidth=2)
axes[1, 0].set_xlabel('频率 (Hz)', fontsize=12)
axes[1, 0].set_ylabel(r'功率谱密度 ($\mathregular{m^2/s}$)', fontsize=12)  # 用LaTeX语法写平方，完美显示
axes[1, 0].set_title('中心节点功率谱验证', fontsize=14)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3, which='both')
axes[1, 0].set_xlim([df, fs/2])

# 图4：相干函数验证
axes[1, 1].plot(f_coh, gamma_sim, label='模拟相干函数', linewidth=0.8)
axes[1, 1].plot(f_coh, gamma_theory**2, 'r--', label='Davenport理论相干函数', linewidth=2)
axes[1, 1].set_xlabel('频率 (Hz)', fontsize=12)
axes[1, 1].set_ylabel('相干系数 (无量纲)', fontsize=12)
axes[1, 1].set_title('中心节点与相邻节点空间相干函数验证', fontsize=14)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim([df, 0.2])

plt.tight_layout()
plt.savefig('SRM_yx11x11风场验证结果.png', dpi=300, bbox_inches='tight')
print("\n验证图表已保存为：SRM_yx11x11风场验证结果.png")
plt.show()

# -------------------------- 第十一步：保存风场数据（适配yx方向网格） --------------------------
np.savez('SRM_yx11x11风场数据.npz',
         space_grid=space_grid,
         Y=Y, Z=Z, x=x, y=y, z=z,  # 保存完整的网格坐标
         indexing='ij',  # 标记网格排列方式为yx方向
         t=t,
         U_processed=U_processed,
         U_norm=U_norm,
         U_mean=U_mean,
         U_std=U_std)
print("\nyx方向11×11网格风场数据已保存为：SRM_yx11x11风场数据.npz")
print("=" * 50)