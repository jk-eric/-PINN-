# 🔥 配置区
# ==========================
PLOT_RATIO = "ratio_10"
# 两个模型的文件路径
PINN_MODEL_FILE = f"fd_pinn_{PLOT_RATIO}_final_with_coh.pth"
MLP_MODEL_FILE = f"fd_mlp_{PLOT_RATIO}_final.pth"
# 两个模型的训练历史
PINN_HISTORY_FILE = f"train_history_{PLOT_RATIO}_final_with_coh.npy"
MLP_HISTORY_FILE = f"train_history_mlp_{PLOT_RATIO}_final.npy"

SAVE_FIGURES = True
DATASET_PATH = "wind_field_dataset.npz"
PLOT_TIME_STEP = 200  # 云图用的时间步
FIG_SCALE = 1.0

# ==========================
# 库 & 全局设置
# ==========================
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_squared_error

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 绘图全局设置（关闭交互，彻底不卡）
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.switch_backend('agg')


# ==========================
# 模型定义（和训练时完全一致）
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
# 数据加载 & 批量预测函数
# ==========================
def load_data():
    data = np.load(DATASET_PATH, allow_pickle=True)
    dataset_dict = data["dataset_dict"].item()
    y_coords = data["y_coords"]
    z_coords = data["z_coords"]
    t_coords = data["t_coords"]
    ny, nz, nt = len(y_coords), len(z_coords), len(t_coords)
    total_space_points = ny * nz

    def norm_neg1_pos1(x, x_min, x_max):
        return 2 * (x - x_min) / (x_max - x_min) - 1

    # 归一化参数
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    t_min, t_max = t_coords.min(), t_coords.max()
    full_output_real = dataset_dict["full"]["output"]
    output_mean = full_output_real.mean()
    output_std = full_output_real.std()

    # 生成全量输入
    Y, Z, T = np.meshgrid(y_coords, z_coords, t_coords, indexing='ij')
    full_input = np.hstack([Y.reshape(-1, 1), Z.reshape(-1, 1), T.reshape(-1, 1)])
    full_input_norm = np.zeros_like(full_input)
    full_input_norm[:, 0] = norm_neg1_pos1(full_input[:, 0], y_min, y_max)
    full_input_norm[:, 1] = norm_neg1_pos1(full_input[:, 1], z_min, z_max)
    full_input_norm[:, 2] = norm_neg1_pos1(full_input[:, 2], t_min, t_max)
    full_input_tensor = torch.tensor(full_input_norm, dtype=torch.float32).to(device)

    # 验证集索引
    train_space_idx = dataset_dict[PLOT_RATIO]["train_space_idx"]
    val_space_idx = np.setdiff1d(np.arange(total_space_points), train_space_idx)
    val_idx_full = (val_space_idx[:, np.newaxis] * nt + np.arange(nt)).ravel()

    data_package = {
        "full_input_tensor": full_input_tensor,
        "full_output_real": full_output_real,
        "val_idx_full": val_idx_full,
        "output_mean": output_mean,
        "output_std": output_std,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "train_y": y_coords[train_space_idx // nz],
        "train_z": z_coords[train_space_idx % nz],
        "ny": ny, "nz": nz, "nt": nt
    }
    print(f"✅ 数据加载完成 | 工况: {PLOT_RATIO}")
    return data_package


def predict_model(model_file, data_package):
    # 加载模型
    model = FDPINN().to(device)
    state = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    # 分批预测，防止显存爆掉
    with torch.no_grad():
        batch_size = 10000
        total_len = len(data_package["full_input_tensor"])
        pred_norm = np.zeros(total_len, dtype=np.float32)
        for i in range(0, total_len, batch_size):
            batch_input = data_package["full_input_tensor"][i:i + batch_size]
            pred_norm[i:i + batch_size] = model(batch_input).cpu().numpy().ravel()

    # 反归一化
    pred = pred_norm * data_package["output_std"] + data_package["output_mean"]
    # 验证集数据
    val_pred = pred[data_package["val_idx_full"]]
    val_real = data_package["full_output_real"][data_package["val_idx_full"]]
    # 计算精度
    r2 = r2_score(val_real, val_pred)
    rmse = np.sqrt(mean_squared_error(val_real, val_pred))
    # 重塑3D数组
    ny, nz, nt = data_package["ny"], data_package["nz"], data_package["nt"]
    U_pred = pred.reshape(ny, nz, nt)
    U_real = data_package["full_output_real"].reshape(ny, nz, nt)

    # 清理显存
    del model, batch_input
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"📊 {model_file.split('_')[1]} 精度 | R²={r2:.4f} | RMSE={rmse:.4f} m/s")
    return U_pred, U_real, val_pred, val_real, r2, rmse


# ==========================
# 主程序：加载数据+两个模型预测
# ==========================
if __name__ == "__main__":
    # 1. 加载数据
    pkg = load_data()
    ny, nz, nt = pkg["ny"], pkg["nz"], pkg["nt"]
    y_coords, z_coords = pkg["y_coords"], pkg["z_coords"]
    train_y, train_z = pkg["train_y"], pkg["train_z"]

    # 2. 两个模型预测
    print("\n正在预测PINN模型...")
    U_pinn, U_real_full, val_pinn, val_real, r2_pinn, rmse_pinn = predict_model(PINN_MODEL_FILE, pkg)
    print("\n正在预测MLP模型...")
    U_mlp, _, val_mlp, _, r2_mlp, rmse_mlp = predict_model(MLP_MODEL_FILE, pkg)

    # 3. 准备绘图数据
    # 云图用单时刻
    Ur = U_real_full[:, :, PLOT_TIME_STEP]
    Up_pinn = U_pinn[:, :, PLOT_TIME_STEP]
    Up_mlp = U_mlp[:, :, PLOT_TIME_STEP]
    err_pinn = Up_pinn - Ur
    err_mlp = Up_mlp - Ur
    vmin, vmax = Ur.min(), Ur.max()
    emax = max(np.abs(err_pinn).max(), np.abs(err_mlp).max())  # 统一误差色标

    # ==========================
    # 图1：风场云图对比（2行3列）
    # ==========================
    print("\n正在绘制图1：风场云图对比...")
    fig1 = plt.figure(figsize=(22 * FIG_SCALE, 14 * FIG_SCALE))
    # 第一行：PINN
    ax1 = plt.subplot(2, 3, 1)
    cf1 = ax1.contourf(y_coords, z_coords, Ur.T, 60, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(cf1, ax=ax1, label='风速 m/s')
    ax1.scatter(train_y, train_z, s=80, c='white', edgecolors='black', linewidths=2, marker='o', label='训练测点')
    ax1.set_title('真实风速场', weight='bold', pad=8, fontsize=14)
    ax1.set_xlabel('y (m)');
    ax1.set_ylabel('z (m)')
    ax1.legend()

    ax2 = plt.subplot(2, 3, 2)
    cf2 = ax2.contourf(y_coords, z_coords, Up_pinn.T, 60, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(cf2, ax=ax2, label='风速 m/s')
    ax2.scatter(train_y, train_z, s=80, c='white', edgecolors='black', linewidths=2, marker='o')
    ax2.set_title('PINN 预测风速场', weight='bold', pad=8, fontsize=14)
    ax2.set_xlabel('y (m)')

    ax3 = plt.subplot(2, 3, 3)
    cf3 = ax3.contourf(y_coords, z_coords, err_pinn.T, 60, cmap='coolwarm', vmin=-emax, vmax=emax)
    plt.colorbar(cf3, ax=ax3, label='误差 m/s')
    ax3.set_title('PINN 误差场', weight='bold', pad=8, fontsize=14)
    ax3.set_xlabel('y (m)')

    # 第二行：MLP
    ax4 = plt.subplot(2, 3, 4)
    cf4 = ax4.contourf(y_coords, z_coords, Ur.T, 60, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(cf4, ax=ax4, label='风速 m/s')
    ax4.scatter(train_y, train_z, s=80, c='white', edgecolors='black', linewidths=2, marker='o')
    ax4.set_title('真实风速场', weight='bold', pad=8, fontsize=14)
    ax4.set_xlabel('y (m)');
    ax4.set_ylabel('z (m)')

    ax5 = plt.subplot(2, 3, 5)
    cf5 = ax5.contourf(y_coords, z_coords, Up_mlp.T, 60, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(cf5, ax=ax5, label='风速 m/s')
    ax5.scatter(train_y, train_z, s=80, c='white', edgecolors='black', linewidths=2, marker='o')
    ax5.set_title('纯MLP 预测风速场', weight='bold', pad=8, fontsize=14)
    ax5.set_xlabel('y (m)')

    ax6 = plt.subplot(2, 3, 6)
    cf6 = ax6.contourf(y_coords, z_coords, err_mlp.T, 60, cmap='coolwarm', vmin=-emax, vmax=emax)
    plt.colorbar(cf6, ax=ax6, label='误差 m/s')
    ax6.set_title('纯MLP 误差场', weight='bold', pad=8, fontsize=14)
    ax6.set_xlabel('y (m)')

    fig1.suptitle(
        f'风速场对比 t={PLOT_TIME_STEP} | {PLOT_RATIO.replace("ratio_", "")}%测点\n(注：PINN R²={r2_pinn:.4f}, MLP R²={r2_mlp:.4f})',
        fontsize=18, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if SAVE_FIGURES:
        plt.savefig(f'对比图1_风场云图_{PLOT_RATIO}.png', dpi=300, bbox_inches='tight')
        print(f"✅ 对比图1 已保存 (看误差场就能看出区别！)")
    plt.close(fig1)

    # ==========================
    # 图2：残差散点图对比（1行2列）
    # ==========================
    print("\n正在绘制图2：残差散点图对比...")
    fig2 = plt.figure(figsize=(20 * FIG_SCALE, 8 * FIG_SCALE))
    # 统一坐标轴范围
    mn, mx = min(val_real.min(), val_pinn.min(), val_mlp.min()), max(val_real.max(), val_pinn.max(), val_mlp.max())

    # 左图：PINN
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(val_real, val_pinn, s=1, alpha=0.5, color='#1f77b4')
    ax1.plot([mn, mx], [mn, mx], 'k--', lw=2, label='完美拟合')
    ax1.set_xlabel('真实风速 m/s', fontsize=12)
    ax1.set_ylabel('预测风速 m/s', fontsize=12)
    ax1.set_title('PINN 预测值 vs 真实值', weight='bold', pad=8, fontsize=14)
    ax1.text(0.05, 0.95, f'$R^2={r2_pinn:.4f}$\nRMSE={rmse_pinn:.4f} m/s',
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
             verticalalignment='top')
    ax1.legend()
    ax1.grid(alpha=0.2)
    ax1.set_xlim(mn, mx)
    ax1.set_ylim(mn, mx)

    # 右图：MLP
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(val_real, val_mlp, s=1, alpha=0.5, color='#ff7f0e')
    ax2.plot([mn, mx], [mn, mx], 'k--', lw=2, label='完美拟合')
    ax2.set_xlabel('真实风速 m/s', fontsize=12)
    ax2.set_ylabel('预测风速 m/s', fontsize=12)
    ax2.set_title('纯MLP 预测值 vs 真实值', weight='bold', pad=8, fontsize=14)
    ax2.text(0.05, 0.95, f'$R^2={r2_mlp:.4f}$\nRMSE={rmse_mlp:.4f} m/s',
             transform=ax2.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
             verticalalignment='top')
    ax2.legend()
    ax2.grid(alpha=0.2)
    ax2.set_xlim(mn, mx)
    ax2.set_ylim(mn, mx)

    fig2.suptitle(f'残差分析对比 | {PLOT_RATIO.replace("ratio_", "")}%测点', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGURES:
        plt.savefig(f'对比图2_残差分析_{PLOT_RATIO}.png', dpi=300, bbox_inches='tight')
        print(f"✅ 对比图2 已保存")
    plt.close(fig2)

    # ==========================
    # 图3：多高度横向风速分布对比
    # ==========================
    print("\n正在绘制图3：横向风速分布对比...")
    fig3 = plt.figure(figsize=(20 * FIG_SCALE, 7 * FIG_SCALE))
    z_heights = [60, 120, 180]
    z_idxs = [np.argmin(np.abs(z_coords - h)) for h in z_heights]

    for i, (z_idx, z_h) in enumerate(zip(z_idxs, z_heights)):
        # 🔥 修复：用完整的3D数组算时间平均
        real_lat = np.mean(U_real_full[:, z_idx, :], axis=1)
        pinn_lat = np.mean(U_pinn[:, z_idx, :], axis=1)
        mlp_lat = np.mean(U_mlp[:, z_idx, :], axis=1)

        plt.subplot(1, 3, i + 1)
        plt.plot(y_coords, real_lat, 'k--', lw=3, label='真实值')
        plt.plot(y_coords, pinn_lat, color='#1f77b4', lw=3, label='PINN预测')
        plt.plot(y_coords, mlp_lat, color='#ff7f0e', lw=3, label='MLP预测')
        plt.xlabel('y (m)', fontsize=12)
        plt.ylabel('风速 (m/s)', fontsize=12)
        plt.title(f'高度 z={z_h}m', weight='bold', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.2)

    fig3.suptitle('横向风速分布对比（时间平均）', fontsize=17, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_FIGURES:
        plt.savefig(f'对比图3_横向风速对比_{PLOT_RATIO}.png', dpi=300, bbox_inches='tight')
        print(f"✅ 对比图3 已保存 (这里曲线区别会很明显！)")
    plt.close(fig3)

    # ==========================
    # 图4：训练收敛曲线对比
    # ==========================
    print("\n正在绘制图4：训练收敛曲线对比...")
    if os.path.exists(PINN_HISTORY_FILE) and os.path.exists(MLP_HISTORY_FILE):
        # 加载训练历史
        hist_pinn = np.load(PINN_HISTORY_FILE, allow_pickle=True).item()
        hist_mlp = np.load(MLP_HISTORY_FILE, allow_pickle=True).item()

        fig4 = plt.figure(figsize=(20 * FIG_SCALE, 7 * FIG_SCALE))
        # 左图：损失收敛曲线
        ax1 = plt.subplot(1, 2, 1)
        steps_pinn = np.arange(len(hist_pinn['total_loss']))
        steps_mlp = np.arange(len(hist_mlp['total_loss']))
        ax1.semilogy(steps_pinn, hist_pinn['total_loss'], '#1f77b4', lw=1.5, label='PINN 总损失')
        ax1.semilogy(steps_mlp, hist_mlp['total_loss'], '#ff7f0e', lw=1.5, label='MLP 总损失')
        ax1.set_xlabel('迭代步', fontsize=12)
        ax1.set_ylabel('损失值（对数坐标）', fontsize=12)
        ax1.set_title('损失收敛曲线对比', weight='bold', pad=8, fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.2)

        # 右图：验证精度对比
        ax2 = plt.subplot(1, 2, 2)
        v_steps_pinn = np.arange(0, len(hist_pinn['total_loss']), 1000)[:len(hist_pinn['val_r2'])]
        v_steps_mlp = np.arange(0, len(hist_mlp['total_loss']), 1000)[:len(hist_mlp['val_r2'])]

        ax2_r2 = ax2
        ax2_rmse = ax2.twinx()
        # R²曲线
        l1 = ax2_r2.plot(v_steps_pinn, hist_pinn['val_r2'], '#1f77b4', lw=2, label='PINN R²')
        l2 = ax2_r2.plot(v_steps_mlp, hist_mlp['val_r2'], '#ff7f0e', lw=2, label='MLP R²')
        # RMSE曲线
        l3 = ax2_rmse.plot(v_steps_pinn, hist_pinn['val_rmse'], '#1f77b4', lw=2, ls='--', label='PINN RMSE')
        l4 = ax2_rmse.plot(v_steps_mlp, hist_mlp['val_rmse'], '#ff7f0e', lw=2, ls='--', label='MLP RMSE')

        ax2_r2.set_ylabel('$R^2$', fontsize=12)
        ax2_rmse.set_ylabel('RMSE (m/s)', fontsize=12)
        ax2.set_xlabel('迭代步', fontsize=12)
        ax2.set_title('验证精度收敛对比', weight='bold', pad=8, fontsize=14)
        # 合并图例
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='center right')
        ax2.grid(alpha=0.2)

        fig4.suptitle('训练过程对比', fontsize=16, weight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if SAVE_FIGURES:
            plt.savefig(f'对比图4_训练曲线_{PLOT_RATIO}.png', dpi=300, bbox_inches='tight')
            print(f"✅ 对比图4 已保存")
        plt.close(fig4)
    else:
        print("⚠️  未找到训练历史文件，跳过图4绘制")

    print("\n🎉 所有对比图绘制完成！")
