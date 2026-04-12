import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0. 固定随机种子
# =========================
np.random.seed(20260305)

# =========================
# 1. 加载已生成的 SRM 风场数据 ✅ 修复这里
# =========================
# 你之前保存的是 npz 文件，正确加载方式：
data = np.load("SRM_yx11x11风场数据.npz", allow_pickle=True)
U_processed = data["U_processed"]  # 取出风速数组

# 你的数据维度是 (121, 400) → reshape 成 (11,11,400)
U_3d = U_processed.reshape(11, 11, 400)
ny, nz, nt = U_3d.shape
print(f"加载风场维度: Y={ny}个点, Z={nz}个点, 时间步={nt}个")

# 重构物理坐标（和你画图一致）
y = np.linspace(-50, 50, ny)
z = np.linspace(50, 200, nz)  # 你之前的高度是 50~200m ✅ 对齐
t = np.linspace(0, 399, nt)

# =========================
# 2. 生成全量时空坐标网格
# =========================
Y_grid, Z_grid, T_grid = np.meshgrid(y, z, t, indexing='ij')

Y_flat = Y_grid.flatten().reshape(-1, 1)
Z_flat = Z_grid.flatten().reshape(-1, 1)
T_flat = T_grid.flatten().reshape(-1, 1)

input_full = np.hstack([Y_flat, Z_flat, T_flat])
output_full = U_3d.flatten().reshape(-1, 1)

print(f"全量数据集: 输入shape={input_full.shape}, 输出shape={output_full.shape}")

# =========================
# 3. 按比例随机抽取稀疏测点
# =========================
total_space_points = ny * nz
train_ratios = [0.7, 0.5, 0.3, 0.1]
dataset_dict = {}

for ratio in train_ratios:
    n_train_points = int(np.round(total_space_points * ratio))
    print(f"\n===== {int(ratio * 100)}%测点工况: 抽取{n_train_points}个空间测点 =====")

    train_space_idx = np.random.choice(total_space_points, size=n_train_points, replace=False)
    dataset_dict[f"ratio_{int(ratio * 100)}"] = {"train_space_idx": train_space_idx}

    U_space_time = U_3d.reshape(total_space_points, nt)
    U_train = U_space_time[train_space_idx, :].flatten().reshape(-1, 1)

    input_space_time = input_full.reshape(total_space_points, nt, 3)
    input_train = input_space_time[train_space_idx, :, :].reshape(-1, 3)

    dataset_dict[f"ratio_{int(ratio * 100)}"]["input_train"] = input_train
    dataset_dict[f"ratio_{int(ratio * 100)}"]["output_train"] = U_train
    print(f"训练集: 输入shape={input_train.shape}, 输出shape={U_train.shape}")

dataset_dict["full"] = {"input": input_full, "output": output_full}

# =========================
# 4. 数据归一化
# =========================
input_mean = input_full.mean(axis=0)
input_std = input_full.std(axis=0)
input_std = np.where(input_std < 1e-6, 1e-6, input_std)

output_mean = output_full.mean()
output_std = output_full.std()
output_std = output_std if output_std > 1e-6 else 1e-6

norm_params = {
    "input_mean": input_mean,
    "input_std": input_std,
    "output_mean": output_mean,
    "output_std": output_std
}

for key in dataset_dict.keys():
    if key == "full":
        dataset_dict[key]["input_norm"] = (dataset_dict[key]["input"] - input_mean) / input_std
        dataset_dict[key]["output_norm"] = (dataset_dict[key]["output"] - output_mean) / output_std
    else:
        dataset_dict[key]["input_train_norm"] = (dataset_dict[key]["input_train"] - input_mean) / input_std
        dataset_dict[key]["output_train_norm"] = (dataset_dict[key]["output_train"] - output_mean) / output_std

print("\n===== 归一化参数 =====")
print(f"输入特征均值: {input_mean}, 标准差: {input_std}")
print(f"输出风速均值: {output_mean:.2f} m/s, 标准差: {output_std:.2f} m/s")

print(f"\n===== 数据校验 =====")
for key in dataset_dict.keys():
    if key == "full":
        has_nan = np.isnan(dataset_dict[key]["input_norm"]).any()
        print(f"全量数据集 输入是否含NaN: {has_nan}")
    else:
        has_nan = np.isnan(dataset_dict[key]["input_train_norm"]).any()
        print(f"{key}工况 训练输入是否含NaN: {has_nan}")

# =========================
# 5. 保存数据集 ✅ 修复路径
# =========================
np.savez("wind_field_dataset.npz",
         dataset_dict=dataset_dict,
         norm_params=norm_params,
         y_coords=y,
         z_coords=z,
         t_coords=t)
print("\n✅ 数据集已保存为 wind_field_dataset.npz")

# =========================
# 6. 可视化验证
# =========================
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(12, 8))
y_mesh, z_mesh = np.meshgrid(y, z, indexing='ij')
y_all = y_mesh.flatten()
z_all = z_mesh.flatten()

for i, ratio in enumerate(train_ratios):
    ax = plt.subplot(2, 2, i + 1)
    ax.scatter(y_all, z_all, c='lightgray', s=60, label='全量测点')
    train_idx = dataset_dict[f"ratio_{int(ratio * 100)}"]["train_space_idx"]
    ax.scatter(y_all[train_idx], z_all[train_idx], c='red', s=80, label='训练测点')
    ax.set_title(f"{int(ratio * 100)}% 测点工况 (共{len(train_idx)}个测点)", fontsize=12)
    ax.set_xlabel("Y Position (m)", fontsize=10)
    ax.set_ylabel("Z Height (m)", fontsize=10)
    ax.set_xlim(-55, 55)
    ax.set_ylim(45, 205)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

plt.suptitle("训练测点空间分布验证", fontsize=16, y=0.98)
plt.tight_layout()
plt.show()