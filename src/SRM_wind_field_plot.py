import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===================== 加载数据 =====================
data = np.load('SRM_yx11x11风场数据.npz', allow_pickle=True)
y_old = data['y']
z_old = data['z']
t_old = data['t']
U_processed = data['U_processed']

x_unique = np.unique(y_old)
y_unique = t_old
z_unique = np.unique(z_old)

U_3d = U_processed.reshape(len(x_unique), len(z_unique), len(y_unique))

# ===================== 步长越小越密 =====================
slice_step = 4
# ======================================================

y_slice_idx = np.arange(0, len(y_unique), slice_step)
y_slice_pos = y_unique[y_slice_idx]

vmin = np.floor(U_3d.min())
vmax = np.ceil(U_3d.max())
cmap = plt.cm.jet

plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(14, 8), dpi=120)
ax = fig.add_subplot(111, projection='3d')

X_mesh, Z_mesh = np.meshgrid(x_unique, z_unique)

for idx, y_idx in enumerate(y_slice_idx):
    U_slice = U_3d[:, :, y_idx].T
    current_y = y_slice_pos[idx]

    ax.plot_surface(
        X_mesh, np.full_like(X_mesh, current_y), Z_mesh,
        rstride=1, cstride=1,
        facecolors=cmap((U_slice - vmin) / (vmax - vmin)),
        shade=False,
        antialiased=True,
        linewidth=0,
        alpha=1.0
    )

ax.set_xlabel(r'$y$ (m)', fontsize=14, labelpad=15)
ax.set_ylabel(r'$t$ (s)', fontsize=14, labelpad=15)
ax.set_zlabel(r'$z$ (m)', fontsize=14, labelpad=15)

ax.set_xlim(x_unique[0], x_unique[-1])
ax.set_ylim(y_unique[0], y_unique[-1])
ax.set_zlim(z_unique[0], z_unique[-1])

ax.tick_params(labelsize=12)
ax.view_init(elev=25, azim=-130)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.12)
cbar.set_label(r'$U$ (m/s)', fontsize=14, labelpad=10)

plt.figtext(0.5, 0.02,
            '采用 SRM 数值模拟得到的风场示意图',
            ha='center', fontsize=16, fontproperties='SimHei')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('SRM_wind_field.png', dpi=300, bbox_inches='tight')
plt.show()