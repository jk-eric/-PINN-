# ==========================
# 🔥 配置区
# ==========================
PLOT_RATIO = "ratio_10"
MODEL_FILE_NAME = f"fd_pinn_{PLOT_RATIO}_final_with_coh.pth"
TRAIN_HISTORY_FILE = f"train_history_{PLOT_RATIO}_final_with_coh.npy"
SAVE_FIGURES = True
DATASET_PATH = "wind_field_dataset.npz"

PLOT_MODE = "single_time"
PLOT_TIME_STEP = 200
FIG_SCALE = 1.0

# ==========================
# 库
# ==========================
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"


# ==========================
# 模型
# ==========================
class FDPINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_layers=5, hidden_neurons=200):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_neurons, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==========================
# 加载数据
# ==========================
def load_data_for_plot():
    data = np.load(DATASET_PATH, allow_pickle=True)
    dataset_dict = data["dataset_dict"].item()
    y_coords = data["y_coords"]
    z_coords = data["z_coords"]
    t_coords = data["t_coords"]
    ny, nz, nt = len(y_coords), len(z_coords), len(t_coords)
    total_space_points = ny * nz

    def norm_neg1_pos1(x, x_min, x_max):
        return 2 * (x - x_min) / (x_max - x_min) - 1

    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    t_min, t_max = t_coords.min(), t_coords.max()

    Y, Z, T = np.meshgrid(y_coords, z_coords, t_coords, indexing='ij')
    full_input = np.hstack([Y.flatten().reshape(-1, 1), Z.flatten().reshape(-1, 1), T.flatten().reshape(-1, 1)])
    full_input_norm = np.zeros_like(full_input)
    full_input_norm[:, 0] = norm_neg1_pos1(full_input[:, 0], y_min, y_max)
    full_input_norm[:, 1] = norm_neg1_pos1(full_input[:, 1], z_min, z_max)
    full_input_norm[:, 2] = norm_neg1_pos1(full_input[:, 2], t_min, t_max)

    full_output_real = dataset_dict["full"]["output"]
    output_mean = full_output_real.mean()
    output_std = full_output_real.std()

    train_space_idx = dataset_dict[PLOT_RATIO]["train_space_idx"]
    val_space_idx = np.setdiff1d(np.arange(total_space_points), train_space_idx)

    data_package = {
        "full_input_tensor": torch.tensor(full_input_norm, dtype=torch.float32).to(device),
        "full_output_real": full_output_real,
        "val_space_idx": val_space_idx,
        "train_space_idx": train_space_idx,  # 🔥 保存训练点索引用于画图
        "ny": ny, "nz": nz, "nt": nt,
        "norm_mean": output_mean, "norm_std": output_std,
        "y_coords": y_coords, "z_coords": z_coords
    }
    print(f"✅ 数据加载完成 | 工况: {PLOT_RATIO}")
    return data_package


# ==========================
# 预测（保留精度修复）
# ==========================
def predict(model_file, data_package):
    model = FDPINN().to(device)
    state = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    with torch.no_grad():
        pred_norm = model(data_package["full_input_tensor"]).cpu().numpy().ravel()

    pred = pred_norm * data_package["norm_std"] + data_package["norm_mean"]

    # 🔥 修复：向量化生成全量验证索引
    val_space_idx = data_package["val_space_idx"]
    nt = data_package["nt"]
    val_idx_full = (val_space_idx[:, np.newaxis] * nt + np.arange(nt)).ravel()

    val_pred = pred[val_idx_full]
    val_real = data_package["full_output_real"][val_idx_full]

    r2 = r2_score(val_real, val_pred)
    rmse = np.sqrt(mean_squared_error(val_real, val_pred))
    print(f"📊 精度 | R²={r2:.4f} | RMSE={rmse:.4f} m/s")

    ny, nz = data_package["ny"], data_package["nz"]
    U_pred = pred.reshape(ny, nz, nt)
    U_real = data_package["full_output_real"].reshape(ny, nz, nt)
    return U_pred, U_real, val_pred, val_real, r2, rmse


# ==========================
# 绘图（完整修复版：含图4+云图标点）
# ==========================
def plot_all_figures(U_pred, U_real, val_pred, val_real, r2, rmse, pkg):
    y_coords = pkg["y_coords"]
    z_coords = pkg["z_coords"]
    ny, nz, nt = pkg["ny"], pkg["nz"], pkg["nt"]
    train_space_idx = pkg["train_space_idx"]

    # 把训练点索引转换成(y,z)坐标
    train_y = y_coords[train_space_idx // nz]
    train_z = z_coords[train_space_idx % nz]

    if PLOT_MODE == "single_time":
        Ur = U_real[:, :, PLOT_TIME_STEP]
        Up = U_pred[:, :, PLOT_TIME_STEP]
        suffix = f"t={PLOT_TIME_STEP}"
    else:
        Ur = U_real.mean(axis=2)
        Up = U_pred.mean(axis=2)
        suffix = "时间平均"

    err = Up - Ur
    vmin, vmax = Ur.min(), Ur.max()
    emax = np.abs(err).max()

    # 图1 风场云图（叠加训练测点）
    fig1 = plt.figure(figsize=(22 * FIG_SCALE, 7 * FIG_SCALE))

    ax1 = plt.subplot(131)
    cf1 = ax1.contourf(y_coords, z_coords, Ur.T, 60, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(cf1, ax=ax1, label='风速 m/s')
    ax1.scatter(train_y, train_z, s=80, c='white', edgecolors='black', linewidths=2, marker='o', label='训练测点')
    ax1.set_title('(a) 真实风速场', weight='bold', pad=8)
    ax1.set_xlabel('y');
    ax1.set_ylabel('z')
    ax1.legend()

    ax2 = plt.subplot(132)
    cf2 = ax2.contourf(y_coords, z_coords, Up.T, 60, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(cf2, ax=ax2, label='风速 m/s')
    ax2.scatter(train_y, train_z, s=80, c='white', edgecolors='black', linewidths=2, marker='o', label='训练测点')
    ax2.set_title('(b) 预测风速场', weight='bold', pad=8)
    ax2.legend()

    ax3 = plt.subplot(133)
    cf3 = ax3.contourf(y_coords, z_coords, err.T, 60, cmap='coolwarm', vmin=-emax, vmax=emax)
    plt.colorbar(cf3, ax=ax3, label='误差 m/s')
    ax3.set_title('(c) 误差场', weight='bold', pad=8)

    fig1.suptitle(f'风速场对比 {suffix} | {PLOT_RATIO.replace("ratio_", "")}%测点', fontsize=16, weight='bold', y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGURES:
        plt.savefig(f'图1_风速场对比_{PLOT_RATIO}.png', dpi=300, bbox_inches='tight')
        print(f"✅ 图1 已保存")

    # 图2 残差（简化版：只散点图）
    fig2 = plt.figure(figsize=(10 * FIG_SCALE, 8 * FIG_SCALE))
    ax1 = plt.gca()
    ax1.scatter(val_real, val_pred, s=1, alpha=0.5, color='#1f77b4')
    mn, mx = min(val_real.min(), val_pred.min()), max(val_real.max(), val_pred.max())
    ax1.plot([mn, mx], [mn, mx], 'k--', lw=2, label='完美拟合')
    ax1.set_xlabel('真实风速 m/s')
    ax1.set_ylabel('预测风速 m/s')
    ax1.set_title('预测值 vs 真实值', weight='bold', pad=8)
    ax1.text(0.05, 0.95, f'$R^2={r2:.4f}$\nRMSE={rmse:.4f} m/s',
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
             verticalalignment='top')
    ax1.legend()
    ax1.grid(alpha=0.2)

    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'图2_残差分析_{PLOT_RATIO}.png', dpi=300, bbox_inches='tight')
        print(f"✅ 图2 已保存")

    # 图3 横向风速分布
    fig3 = plt.figure(figsize=(20 * FIG_SCALE, 7 * FIG_SCALE))
    z_heights = [60, 120, 180]
    z_idxs = [np.argmin(np.abs(z_coords - h)) for h in z_heights]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (z_idx, z_h) in enumerate(zip(z_idxs, z_heights)):
        real_lat = np.mean(U_real[:, z_idx, :], axis=1)
        pred_lat = np.mean(U_pred[:, z_idx, :], axis=1)

        plt.subplot(1, 3, i + 1)
        plt.plot(y_coords, real_lat, 'k--', lw=3, label='真实')
        plt.plot(y_coords, pred_lat, color=colors[i], lw=3, label='预测')
        plt.xlabel('y (m)')
        plt.ylabel('风速 (m/s)')
        plt.title(f'z={z_h}m', weight='bold')
        plt.legend()
        plt.grid(alpha=0.2)

    plt.suptitle('横向风速分布对比', fontsize=17, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGURES:
        plt.savefig(f'图3_横向风速对比_{PLOT_RATIO}.png', dpi=300, bbox_inches='tight')
        print(f"✅ 图3 已保存")

    # 🔥 图4 训练曲线（加回来了）
    if os.path.exists(TRAIN_HISTORY_FILE):
        hist = np.load(TRAIN_HISTORY_FILE, allow_pickle=True).item()
        fig4 = plt.figure(figsize=(20 * FIG_SCALE, 7 * FIG_SCALE))
        ax1 = plt.subplot(121)
        steps = np.arange(len(hist['total_loss']))
        ax1.semilogy(steps, hist['total_loss'], 'k', lw=1.5, label='总损失')
        ax1.semilogy(steps, hist['Lb'], '#1f77b4', alpha=0.7, label='监督损失')
        ax1.semilogy(steps, hist['Le'], '#ff7f0e', alpha=0.7, label='物理损失')
        ax1.set_xlabel('迭代步')
        ax1.set_ylabel('损失 (对数)')
        ax1.set_title('(a) 损失收敛曲线', weight='bold', pad=8)
        ax1.legend()
        ax1.grid(alpha=0.2)

        ax2 = plt.subplot(122)
        v_steps = np.arange(0, len(hist['total_loss']), 1000)[:len(hist['val_r2'])]
        ax2_r2 = ax2
        ax2_rmse = ax2.twinx()
        l1 = ax2_r2.plot(v_steps, hist['val_r2'], '#2ca02c', lw=2, label='R²')
        l2 = ax2_rmse.plot(v_steps, hist['val_rmse'], 'r', lw=2, ls='--', label='RMSE')
        ax2_r2.set_ylabel('$R^2$', color='#2ca02c')
        ax2_rmse.set_ylabel('RMSE', color='r')
        ax2_r2.tick_params(axis='y', labelcolor='#2ca02c')
        ax2_rmse.tick_params(axis='y', labelcolor='r')
        ax2.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='center right')
        ax2.set_xlabel('迭代步')
        ax2.set_title('(b) 验证精度', weight='bold', pad=8)
        ax2.grid(alpha=0.2)

        fig4.suptitle('训练曲线', fontsize=16, weight='bold', y=1.03)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if SAVE_FIGURES:
            plt.savefig(f'图4_训练曲线_{PLOT_RATIO}.png', dpi=300, bbox_inches='tight')
            print(f"✅ 图4 已保存")

    print("\n🎉 所有图绘制完成！")
    plt.show()
# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    pkg = load_data_for_plot()
    up, ur, vp, vr, r2, rmse = predict(MODEL_FILE_NAME, pkg)
    plot_all_figures(up, ur, vp, vr, r2, rmse, pkg)