import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 🔧 【一键配置区】所有需要改的都在这！下面逻辑代码不用动
# ==========================================
# 1. 随机种子（保证结果可复现，和你论文一致）
SEED = 20260305

# 2. 原始SRM风场数据配置
RAW_DATA_PATH = "SRM_yx11x11风场数据.npz"  # 改成你实际的文件名（建议改成英文）
RAW_DATA_KEY = "U_processed"  # 改成你npz文件里实际的风速数组键名

# 3. 风场物理坐标配置（和你论文/画图一致）
Y_RANGE = (-50, 50)  # 横向坐标范围 (m)
Z_RANGE = (50, 200)  # 高度坐标范围 (m)
T_RANGE = (0, 399)  # 时间步范围
GRID_SHAPE = (11, 11)  # 空间网格点数 (Y方向, Z方向)
NT = 400  # 总时间步数

# 4. 训练测点比例配置（只改这里，就能生成不同比例的数据集）
TRAIN_RATIOS = [0.7, 0.5, 0.3, 0.05]  # 你论文里的4个比例：70%,50%,30%,10%
# 如果你想加其他比例，直接在这加就行，比如：TRAIN_RATIOS = [0.7, 0.5, 0.3, 0.1, 0.05]

# 5. 最终保存的数据集文件名
SAVE_DATA_PATH = "wind_field_dataset.npz"

# 6. 可视化字体配置
FONT_FAMILY = ["SimHei", "DejaVu Sans"]  # 优先用黑体显示中文，找不到用默认
# ==========================================
# 🔧 【配置区结束】下面的逻辑代码不用动
# ==========================================

# =========================
# 0. 固定随机种子
# =========================
np.random.seed(SEED)

# =========================
# 1. 加载已生成的 SRM 风场数据
# =========================
print(f"正在加载原始数据: {RAW_DATA_PATH} ...")
data = np.load(RAW_DATA_PATH, allow_pickle=True)

# 【安全检查】确保npz文件里有需要的键
assert RAW_DATA_KEY in data.files, f"错误：npz文件里找不到键 '{RAW_DATA_KEY}'！现有键: {data.files}"
U_processed = data[RAW_DATA_KEY]

# 数据reshape
U_3d = U_processed.reshape(*GRID_SHAPE, NT)
ny, nz, nt = U_3d.shape
print(f"✅ 加载风场维度: Y={ny}个点, Z={nz}个点, 时间步={nt}个")

# 重构物理坐标
y = np.linspace(*Y_RANGE, ny)
z = np.linspace(*Z_RANGE, nz)
t = np.linspace(*T_RANGE, nt)

# =========================
# 2. 生成全量时空坐标网格
# =========================
Y_grid, Z_grid, T_grid = np.meshgrid(y, z, t, indexing='ij')

Y_flat = Y_grid.flatten().reshape(-1, 1)
Z_flat = Z_grid.flatten().reshape(-1, 1)
T_flat = T_grid.flatten().reshape(-1, 1)

input_full = np.hstack([Y_flat, Z_flat, T_flat])
output_full = U_3d.flatten().reshape(-1, 1)

print(f"✅ 全量数据集: 输入shape={input_full.shape}, 输出shape={output_full.shape}")

# =========================
# 3. 按比例随机抽取稀疏测点
# =========================
total_space_points = ny * nz
dataset_dict = {}

for ratio in TRAIN_RATIOS:
    n_train_points = int(np.round(total_space_points * ratio))
    ratio_name = f"ratio_{int(ratio * 100)}"
    print(f"\n===== 正在处理 {int(ratio * 100)}%测点工况: 抽取{n_train_points}个空间测点 =====")

    # 随机抽取空间测点
    train_space_idx = np.random.choice(total_space_points, size=n_train_points, replace=False)
    dataset_dict[ratio_name] = {"train_space_idx": train_space_idx}

    # 提取训练集数据
    U_space_time = U_3d.reshape(total_space_points, nt)
    U_train = U_space_time[train_space_idx, :].flatten().reshape(-1, 1)

    input_space_time = input_full.reshape(total_space_points, nt, 3)
    input_train = input_space_time[train_space_idx, :, :].reshape(-1, 3)

    dataset_dict[ratio_name]["input_train"] = input_train
    dataset_dict[ratio_name]["output_train"] = U_train
    print(f"✅ {ratio_name}训练集: 输入shape={input_train.shape}, 输出shape={U_train.shape}")

# 保存全量数据集
dataset_dict["full"] = {"input": input_full, "output": output_full}

# =========================
# 4. 数据归一化
# =========================
print("\n===== 正在进行数据归一化 =====")
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

# 对所有数据集进行归一化
for key in dataset_dict.keys():
    if key == "full":
        dataset_dict[key]["input_norm"] = (dataset_dict[key]["input"] - input_mean) / input_std
        dataset_dict[key]["output_norm"] = (dataset_dict[key]["output"] - output_mean) / output_std
    else:
        dataset_dict[key]["input_train_norm"] = (dataset_dict[key]["input_train"] - input_mean) / input_std
        dataset_dict[key]["output_train_norm"] = (dataset_dict[key]["output_train"] - output_mean) / output_std

print(f"✅ 输入特征均值: {input_mean}, 标准差: {input_std}")
print(f"✅ 输出风速均值: {output_mean:.2f} m/s, 标准差: {output_std:.2f} m/s")

# 数据校验（检查NaN）
print(f"\n===== 数据校验 =====")
for key in dataset_dict.keys():
    if key == "full":
        has_nan = np.isnan(dataset_dict[key]["input_norm"]).any()
    else:
        has_nan = np.isnan(dataset_dict[key]["input_train_norm"]).any()
    print(f"✅ {key}工况 输入是否含NaN: {has_nan}")

# =========================
# 5. 保存数据集
# =========================
np.savez(SAVE_DATA_PATH,
         dataset_dict=dataset_dict,
         norm_params=norm_params,
         y_coords=y,
         z_coords=z,
         t_coords=t)
print(f"\n🎉 数据集已保存为: {SAVE_DATA_PATH}")

# =========================
# 6. 可视化验证
# =========================
print("\n===== 正在生成可视化验证图 =====")
# 字体设置（带fallback）
try:
    plt.rcParams["font.family"] = FONT_FAMILY
except:
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    print("⚠️ 未找到SimHei字体，中文可能显示为方框，不影响代码运行")
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(12, 8))
y_mesh, z_mesh = np.meshgrid(y, z, indexing='ij')
y_all = y_mesh.flatten()
z_all = z_mesh.flatten()

for i, ratio in enumerate(TRAIN_RATIOS):
    ratio_name = f"ratio_{int(ratio * 100)}"
    ax = plt.subplot(2, 2, i + 1)

    # 画全量测点
    ax.scatter(y_all, z_all, c='lightgray', s=60, label='全量测点')
    # 画训练测点
    train_idx = dataset_dict[ratio_name]["train_space_idx"]
    ax.scatter(y_all[train_idx], z_all[train_idx], c='red', s=80, label='训练测点')

    ax.set_title(f"{int(ratio * 100)}% 测点工况 (共{len(train_idx)}个测点)", fontsize=12)
    ax.set_xlabel("Y Position (m)", fontsize=10)
    ax.set_ylabel("Z Height (m)", fontsize=10)
    ax.set_xlim(Y_RANGE[0] - 5, Y_RANGE[1] + 5)
    ax.set_ylim(Z_RANGE[0] - 5, Z_RANGE[1] + 5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

plt.suptitle("训练测点空间分布验证", fontsize=16, y=0.98)
plt.tight_layout()
plt.show()
print("🎉 可视化验证图已生成！")
