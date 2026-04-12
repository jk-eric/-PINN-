# ==========================
# 1. 基础库导入 & 环境配置
# ==========================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import r2_score, mean_squared_error

# 设备选择：优先CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用训练设备: {device}")

# 固定随机种子
SEED = 20260305
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================
# 2. 核心参数定义（完全对应论文）
# ==========================
# 物理参数（论文公式2.2直接复用）
U_ref = 30.0
z_ref = 10.0
alpha = 0.16
C_y, C_z = 10.0, 10.0  # 论文相干函数衰减因子
dt = 1.0
fs = 1 / dt
N_t = 400
f_feature = fs / N_t  # 风场特征频率，对应论文公式2.2的f

# 网络参数
INPUT_DIM = 3
OUTPUT_DIM = 1
HIDDEN_LAYERS = 5
HIDDEN_NEURONS = 200

# 训练参数
TRAIN_RATIO = "ratio_10"
TRAIN_STEPS =100000  # 用之前的权重继续跑3万步即可，不用从头跑
LR_INIT = 1e-3        # 微调学习率，适配新增的相干损失
WEIGHT_Le = 1       # 论文物理损失总权重
WEIGHT_Lb = 1         # 监督损失权重
GRAD_CLIP = 1.0

# 断点续训参数
SAVE_STEP = 10000
CHECKPOINT_PATH = f"fd_pinn_{TRAIN_RATIO}_checkpoint"
RESUME_TRAINING = True  # 开启续训，用你之前的final模型继续跑

# ==========================
# 3. FD-PINN 网络结构（完全不变，兼容原有权重）
# ==========================
class FDPINN(nn.Module):
    def __init__(self):
        super(FDPINN, self).__init__()
        layers = []
        layers.append(nn.Linear(INPUT_DIM, HIDDEN_NEURONS))
        layers.append(nn.Tanh())
        for _ in range(HIDDEN_LAYERS):
            layers.append(nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(HIDDEN_NEURONS, OUTPUT_DIM))
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.model(x)

# ==========================
# 4. 数据集加载（完全不变，兼容原有数据集）
# ==========================
def load_dataset(data_path="wind_field_dataset.npz"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件 {data_path} 不存在！")

    data = np.load(data_path, allow_pickle=True)
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
    full_input = np.hstack([Y.flatten().reshape(-1, 1),
                            Z.flatten().reshape(-1, 1),
                            T.flatten().reshape(-1, 1)])
    full_input_norm = np.zeros_like(full_input)
    full_input_norm[:, 0] = norm_neg1_pos1(full_input[:, 0], y_min, y_max)
    full_input_norm[:, 1] = norm_neg1_pos1(full_input[:, 1], z_min, z_max)
    full_input_norm[:, 2] = norm_neg1_pos1(full_input[:, 2], t_min, t_max)

    full_output_real = dataset_dict["full"]["output"]
    output_mean = full_output_real.mean()
    output_std = full_output_real.std()
    full_output_norm = (full_output_real - output_mean) / output_std

    train_space_idx = dataset_dict[TRAIN_RATIO]["train_space_idx"]
    input_space_time = full_input.reshape(total_space_points, nt, 3)
    output_space_time_norm = full_output_norm.reshape(total_space_points, nt)

    train_input = input_space_time[train_space_idx, :, :].reshape(-1, 3)
    train_input_norm = np.zeros_like(train_input)
    train_input_norm[:, 0] = norm_neg1_pos1(train_input[:, 0], y_min, y_max)
    train_input_norm[:, 1] = norm_neg1_pos1(train_input[:, 1], z_min, z_max)
    train_input_norm[:, 2] = norm_neg1_pos1(train_input[:, 2], t_min, t_max)
    train_output_norm = output_space_time_norm[train_space_idx, :].flatten().reshape(-1, 1)

    val_space_idx = np.setdiff1d(np.arange(total_space_points), train_space_idx)
    val_input = input_space_time[val_space_idx, :, :].reshape(-1, 3)
    val_input_norm = np.zeros_like(val_input)
    val_input_norm[:, 0] = norm_neg1_pos1(val_input[:, 0], y_min, y_max)
    val_input_norm[:, 1] = norm_neg1_pos1(val_input[:, 1], z_min, z_max)
    val_input_norm[:, 2] = norm_neg1_pos1(val_input[:, 2], t_min, t_max)
    val_output_real = full_output_real.reshape(total_space_points, nt)[val_space_idx, :].flatten()

    train_input_tensor = torch.tensor(train_input_norm, dtype=torch.float32, requires_grad=True).to(device)
    train_output_tensor = torch.tensor(train_output_norm, dtype=torch.float32).to(device)
    full_input_tensor = torch.tensor(full_input_norm, dtype=torch.float32, requires_grad=True).to(device)
    val_input_tensor = torch.tensor(val_input_norm, dtype=torch.float32).to(device)

    # 预计算坐标网格，用于相干函数计算
    Y_grid, Z_grid = np.meshgrid(y_coords, z_coords, indexing='ij')
    Y_tensor = torch.tensor(Y_grid, dtype=torch.float32).to(device)
    Z_tensor = torch.tensor(Z_grid, dtype=torch.float32).to(device)

    data_dict = {
        "train_input": train_input_tensor,
        "train_output": train_output_tensor,
        "full_input": full_input_tensor,
        "val_input": val_input_tensor,
        "val_output_real": val_output_real,
        "norm_params": {
            "output_mean": output_mean,
            "output_std": output_std,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max,
            "t_min": t_min, "t_max": t_max
        },
        "y_coords": y_coords,
        "z_coords": z_coords,
        "Y_tensor": Y_tensor,
        "Z_tensor": Z_tensor,
        "ny": ny, "nz": nz, "nt": nt
    }
    print(f"总空间测点: {total_space_points}个 | 训练测点: {len(train_space_idx)}个 | 验证测点: {len(val_space_idx)}个")
    return data_dict

# ==========================
# 5. 损失函数（100%贴合论文公式）
# ==========================
def cal_total_loss(model, data_dict):
    ny, nz, nt = data_dict["ny"], data_dict["nz"], data_dict["nt"]
    norm_params = data_dict["norm_params"]
    Y_tensor = data_dict["Y_tensor"]
    Z_tensor = data_dict["Z_tensor"]

    # --------------------------
    # 1. 监督损失 Lb（完全不变）
    # --------------------------
    U_pred_train = model(data_dict["train_input"])
    Lb = torch.mean(torch.square(U_pred_train - data_dict["train_output"]))

    # --------------------------
    # 2. 物理损失 Le（完全对应论文公式2.6，包含2项）
    # --------------------------
    # 全量预测结果反归一化到真实风速
    U_pred_full = model(data_dict["full_input"])
    U_mean = torch.tensor(norm_params["output_mean"], dtype=torch.float32).to(device)
    U_std = torch.tensor(norm_params["output_std"], dtype=torch.float32).to(device)
    U_pred_real = U_pred_full * U_std + U_mean
    U_pred_3d = U_pred_real.reshape(ny, nz, nt)  # 形状 [ny, nz, nt]

    # --- 第1项：风廓线垂向规律损失（对应论文i=1）
    U_pred_avg = torch.mean(U_pred_3d, dim=2)  # 时间平均风速 [ny, nz]
    z_tensor = torch.tensor(data_dict["z_coords"], dtype=torch.float32).to(device)
    U_theory = U_ref * torch.pow(z_tensor / z_ref, alpha)  # 理论风廓线 [nz]
    eps1 = U_pred_avg - U_theory.unsqueeze(0)  # 风廓线残差

    # --- 第2项：Davenport相干函数损失（完全对应论文公式2.2，i=2）
    # 1. 计算y方向相邻点对的理论相干函数与实际相关系数
    # 相邻点坐标差
    delta_y = Y_tensor[1:, :] - Y_tensor[:-1, :]  # [ny-1, nz]
    delta_z = Z_tensor[1:, :] - Z_tensor[:-1, :]  # [ny-1, nz]
    # 两点平均风速
    U_avg_y = (U_pred_avg[1:, :] + U_pred_avg[:-1, :]) / 2  # [ny-1, nz]
    # 论文公式2.2：计算理论相干函数
    dist_y = torch.sqrt(C_y**2 * delta_y**2 + C_z**2 * delta_z**2)
    gamma_theory_y = torch.exp( - f_feature * dist_y / U_avg_y )
    # 计算实际预测风速的相关系数
    U_j_y = U_pred_3d[1:, :, :]  # [ny-1, nz, nt]
    U_k_y = U_pred_3d[:-1, :, :]  # [ny-1, nz, nt]
    # 去均值
    U_j_y_center = U_j_y - torch.mean(U_j_y, dim=2, keepdim=True)
    U_k_y_center = U_k_y - torch.mean(U_k_y, dim=2, keepdim=True)
    # 计算相关系数
    cov_y = torch.mean(U_j_y_center * U_k_y_center, dim=2)
    std_j_y = torch.std(U_j_y, dim=2)
    std_k_y = torch.std(U_k_y, dim=2)
    gamma_pred_y = cov_y / (std_j_y * std_k_y + 1e-8)  # 加小值避免除零
    # 相干函数残差
    eps2_y = gamma_pred_y - gamma_theory_y

    # 2. 计算z方向相邻点对的理论相干函数与实际相关系数
    delta_y_z = Y_tensor[:, 1:] - Y_tensor[:, :-1]  # [ny, nz-1]
    delta_z_z = Z_tensor[:, 1:] - Z_tensor[:, :-1]  # [ny, nz-1]
    U_avg_z = (U_pred_avg[:, 1:] + U_pred_avg[:, :-1]) / 2  # [ny, nz-1]
    dist_z = torch.sqrt(C_y**2 * delta_y_z**2 + C_z**2 * delta_z_z**2)
    gamma_theory_z = torch.exp( - f_feature * dist_z / U_avg_z )
    # 实际相关系数
    U_j_z = U_pred_3d[:, 1:, :]  # [ny, nz-1, nt]
    U_k_z = U_pred_3d[:, :-1, :]  # [ny, nz-1, nt]
    U_j_z_center = U_j_z - torch.mean(U_j_z, dim=2, keepdim=True)
    U_k_z_center = U_k_z - torch.mean(U_k_z, dim=2, keepdim=True)
    cov_z = torch.mean(U_j_z_center * U_k_z_center, dim=2)
    std_j_z = torch.std(U_j_z, dim=2)
    std_k_z = torch.std(U_k_z, dim=2)
    gamma_pred_z = cov_z / (std_j_z * std_k_z + 1e-8)
    eps2_z = gamma_pred_z - gamma_theory_z

    # --- 合并2项残差，计算最终Le（完全对应论文公式2.6）
    eps_all = torch.cat([eps1.flatten(), eps2_y.flatten(), eps2_z.flatten()])
    Ne = len(eps_all)
    Le = (1 / Ne) * torch.sum(torch.square(eps_all))

    # --------------------------
    # 总损失
    # --------------------------
    LT = WEIGHT_Le * Le + WEIGHT_Lb * Lb
    return LT, Lb, Le

# ==========================
# 6. 验证集评估（完全不变）
# ==========================
def evaluate_model(model, data_dict):
    model.eval()
    with torch.no_grad():
        U_pred_norm = model(data_dict["val_input"])
        norm_params = data_dict["norm_params"]
        U_pred_real = U_pred_norm.cpu().numpy() * norm_params["output_std"] + norm_params["output_mean"]
        U_pred_real = U_pred_real.flatten()
        r2 = r2_score(data_dict["val_output_real"], U_pred_real)
        rmse = np.sqrt(mean_squared_error(data_dict["val_output_real"], U_pred_real))
    model.train()
    return r2, rmse

# ==========================
# 7. 断点保存/加载（完全不变，兼容原有断点）
# ==========================
def save_checkpoint(model, optimizer, scheduler, history, step):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history
    }
    torch.save(checkpoint, f"{CHECKPOINT_PATH}_step{step}.pth")
    print(f"✅ 断点已保存：{CHECKPOINT_PATH}_step{step}.pth")

def load_checkpoint(model, optimizer, scheduler):
    latest_step = 0
    latest_checkpoint = None
    for filename in os.listdir("."):
            if filename.startswith(CHECKPOINT_PATH) and filename.endswith(".pth"):
                try:
                    step = int(filename.split("_step")[-1].split(".pth")[0])
                    if step > latest_step:
                        latest_step = step
                        latest_checkpoint = filename
                except:
                    continue

    if latest_checkpoint and RESUME_TRAINING:
        print(f"🔄 发现断点文件：{latest_checkpoint}，正在加载...")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"] + 1
        history = checkpoint["history"]
        print(f"✅ 断点加载成功！从第 {start_step} 步继续训练。")
        return model, optimizer, scheduler, start_step, history
    else:
        print("🆕 未发现断点文件，从头开始训练。")
        return model, optimizer, scheduler, 0, {
            "total_loss": [], "Lb": [], "Le": [],
            "val_r2": [], "val_rmse": []
        }

# ==========================
# 8. 训练主循环（🔥 加了时间记录和保存）
# ==========================
def train_model():
    model = FDPINN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.5)
    data_dict = load_dataset()

    model, optimizer, scheduler, start_step, history = load_checkpoint(model, optimizer, scheduler)

    # 🔥 修改1：记录训练开始时间
    start_time = time.time()
    print(f"\n开始训练 | 总步数: {TRAIN_STEPS} | 初始学习率: {LR_INIT}")
    print(f"当前工况: {TRAIN_RATIO} 测点 | 断点续训: {'开启' if RESUME_TRAINING else '关闭'}")
    print(f"✅ 已加载Davenport相干函数物理约束，完全对应论文公式")

    for step in range(start_step, TRAIN_STEPS):
        optimizer.zero_grad()
        LT, Lb, Le = cal_total_loss(model, data_dict)
        LT.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        history["total_loss"].append(LT.item())
        history["Lb"].append(Lb.item())
        history["Le"].append(Le.item())

        if step % 1000 == 0:
            r2, rmse = evaluate_model(model, data_dict)
            history["val_r2"].append(r2)
            history["val_rmse"].append(rmse)
            elapsed_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Step [{step:6d}/{TRAIN_STEPS}] | 总损失: {LT.item():.6f} | Lb: {Lb.item():.6f} | Le: {Le.item():.6f}")
            print(
                f"               | 耗时: {elapsed_time:.2f}s | 学习率: {current_lr:.2e} | R²: {r2:.4f} | RMSE: {rmse:.4f} m/s\n")

        if step % SAVE_STEP == 0 and step != 0:
            save_checkpoint(model, optimizer, scheduler, history, step)

    # 🔥 修改2：计算总训练时间
    end_time = time.time()
    total_train_time_sec = end_time - start_time
    total_train_time_min = total_train_time_sec / 60

    # 🔥 修改3：把时间信息保存到history字典里
    history["total_train_time_sec"] = total_train_time_sec
    history["total_train_time_min"] = total_train_time_min

    # 保存模型和历史
    torch.save(model.state_dict(), f"fd_pinn_{TRAIN_RATIO}_final_with_coh.pth")
    np.save(f"train_history_{TRAIN_RATIO}_final_with_coh.npy", history)
    print(f"✅ 训练完成！最终模型已保存为 fd_pinn_{TRAIN_RATIO}_final_with_coh.pth")

    # 🔥 修改4：打印最终结果时，同时打印总训练时间
    final_r2, final_rmse = evaluate_model(model, data_dict)
    print(f"\n" + "="*50)
    print(f"===== 【最终结果】 =====")
    print(f"总训练步数: {TRAIN_STEPS}")
    print(f"总训练时间: {total_train_time_sec:.2f} 秒 | {total_train_time_min:.2f} 分钟")
    print(f"验证集 R²: {final_r2:.4f}")
    print(f"验证集 RMSE: {final_rmse:.4f} m/s")
    print(f"="*50)

    return model, history

# ==========================
# 9. 主程序入口
# ==========================
if __name__ == "__main__":
    train_model()