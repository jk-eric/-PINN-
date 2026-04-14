# FD-PINN-3D-Wind-Field-Prediction

基于物理信息神经网络 (PINN) 的三维风场预测。
## 项目简介
 本项目为毕业设计，针对相关论文进行代码复现，即利用物理信息神经网络PNN实现三维风场预测。

 本项目添加了对比普通MLP数据驱动模型效果，验证物理约束对预测精度的提升作用。
 
论文原文题目：**A novel frequency-domain physics-informed neural network for accurate prediction of 3D spatio-temporal wind fields in wind turbine applications**

连接：https://doi.org/10.1016/j.apenergy.2025.125526

## 📁 项目文件结构

```text
FD-PINN-3D-Wind-Field-Prediction/
├── we5                            # 在we=5下不同数据集图片结果
├── ratio_10                       # 在10%数据集下we取不同权重图片结果
├── src/
│   ├── SRM_wind_field.py          # 生成仿真风场数据
│   ├── SRM_wind_field_plot.py     # 绘制仿真风场图
│   ├── data_set.py                # 数据集加载与预处理
│   ├── pinn.py                    # PINN(PNN)模型训练与预测
│   ├── pure_mlp.py                # 纯MLP基线模型（无物理约束）
│   ├── pinn_plot.py               # 绘制PNN预测结果图
│   └── contrast_plot.py           # PNN vs 纯MLP 对比图
├── README.md               # 本文件
└── requirements.txt        # 依赖包
```
---
## 🛠️ 环境说明 (Environment)

为了确保项目能够正常运行，建议在以下环境中进行配置：

* **Python 版本**:  `3.14` 
* **计算设备**: 
    * **CPU**: 支持基础训练与推理。
    * **GPU (强烈推荐)**: 若要进行大规模训练，请安装支持 CUDA 的 PyTorch。
    * *安装命令参考*: 请访问 [PyTorch 官网](https://pytorch.org/) 获取适合你显卡驱动的安装指令。
---
## 🛠️ 具体步骤

### 1. 安装依赖包
在项目根目录下打开终端，执行以下命令安装必要的库：
```bash
pip install -r requirements.txt
```
###  2. 项目运行步骤
 
2.1 生成仿真风场数据
 
- 运行 [src/SRM_wind_field.py](src/SRM_wind_field.py) ，生成项目所需的三维风场原始数据。

- （可选）运行 [src/SRM_wind_field_plot.py](src/SRM_wind_field_plot.py)，可视化生成的原始风场。
 
2.2 数据集预处理
 
- 运行 [src/data_set.py](src/data_set.py)，完成数据加载、归一化与训练/测试集划分。选择测点百分比
 
2.3 模型训练
 
- 物理信息神经网络（带物理约束）：运行[src/pinn.py]( src/pinn.py)

- 纯MLP基线模型（无物理约束）：运行 [src/pure_mlp.py](src/pure_mlp.py)
 
2.4 结果可视化
 
- PNN单独结果图：运行 [src/pinn_plot.py](src/pinn_plot.py)

- 模型效果对比图：运行 [src/contrast_plot.py](src/contrast_plot.py)
