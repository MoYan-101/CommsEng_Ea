import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# ===== 数据读取 =====
df = pd.read_csv("DATA_VIS_YEAR.csv")          # ← 修改为你的 CSV 路径
df_plot = df[["year", "Temperature (°C)",
              "Methanol selectivity (%)",
              "CO2 conversion efficiency (%)"]].dropna()
df_plot.columns = ["year", "temperature",
                   "methanol_selectivity", "co2_conversion"]
df_plot["combo_score"] = (df_plot["methanol_selectivity"]
                          + df_plot["co2_conversion"])

# ===== 最大点定位 =====
max_idx = df_plot["combo_score"].idxmax()
x_max = df_plot.loc[max_idx, "year"]
y_max = df_plot.loc[max_idx, "methanol_selectivity"]
z_max = df_plot.loc[max_idx, "co2_conversion"]
t_max = df_plot.loc[max_idx, "temperature"]

# ===== 点大小缩放 =====
base = 15
size_scaled = df_plot["methanol_selectivity"] + base
size_scaled = (size_scaled / size_scaled.max()) * 100

# ===== 每年最大轨迹点 =====
selectivity_max = df_plot.loc[df_plot.groupby("year")
                                         ["methanol_selectivity"].idxmax()]
conversion_max  = df_plot.loc[df_plot.groupby("year")
                                         ["co2_conversion"].idxmax()]

# ===== 全局字体 & 样式（全部加粗） =====
mpl.rcParams.update({
    "font.size": 13,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.4,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ===== 布局：两行（legend | 图+色条） =====
fig = plt.figure(figsize=(12, 10))
gs  = GridSpec(2, 1, height_ratios=[0.10, 0.90], figure=fig)
gs_body = gs[1].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.1)

# -- 3D 图 ---------------------------------------------------------------
ax3d = fig.add_subplot(gs_body[0], projection="3d")
# ====== NEW: 把相机拉近一点 ======
ax3d.dist = 8.5            # default≈10，数值越小越近；自己微调
# ax3d.set_proj_type("persp", focal_length=0.4)

sc3d = ax3d.scatter(df_plot["year"], df_plot["methanol_selectivity"],
                    df_plot["co2_conversion"],
                    c=df_plot["temperature"], s=size_scaled,
                    cmap="plasma", alpha=0.7, edgecolors="k",
                    linewidth=0.3, marker="o")

line_sel = ax3d.plot(selectivity_max["year"],
                     selectivity_max["methanol_selectivity"],
                     selectivity_max["co2_conversion"],
                     color="#1f77b4", linewidth=2)[0]

line_conv = ax3d.plot(conversion_max["year"],
                      conversion_max["methanol_selectivity"],
                      conversion_max["co2_conversion"],
                      color="#d62728", linestyle="--", linewidth=2)[0]

max_marker = ax3d.plot([x_max], [y_max], [z_max],
                       marker="+", markersize=12,
                       color="black", markeredgewidth=2, linestyle='None')[0]

ax3d.text(x_max + 0.5, y_max, z_max,
          f"T = {t_max:.1f} °C\nMS+CCE = {y_max + z_max:.1f}",
          fontsize=10, color="blue")

ax3d.set_xlabel("Year")
ax3d.set_ylabel("Methanol Selectivity (%)")
ax3d.set_zlabel("CO₂ Conversion Efficiency (%)")
ax3d.grid(False)
# ─── ❶ 让 y / z 轴都从 0 开始 ───
y_pad = 2
z_pad = 2
ax3d.set_ylim(0, df_plot["methanol_selectivity"].max() + y_pad)
ax3d.set_zlim(0, df_plot["co2_conversion"].max() + z_pad)

# 加粗刻度
plt.setp(ax3d.get_xticklabels() +
         ax3d.get_yticklabels() +
         ax3d.get_zticklabels(), fontweight="bold")


# -- 竖直 colorbar ------------------------------------------------------
ax_cbar = fig.add_subplot(gs_body[1])
cbar = fig.colorbar(sc3d, cax=ax_cbar, orientation="vertical")
cbar.set_label("Temperature (°C)", rotation=90,
               labelpad=12, fontweight="bold")
# 加粗刻度
for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight("bold")
    tick.set_rotation(90)

# -- legend（顶部单独一行） --------------------------------------------
# ── Legend ───────────────────────────────────────────────
ax_leg = fig.add_subplot(gs[0])
ax_leg.axis("off")

# 把 ax_leg 这块横向缩到 0.85，并居中
pos = ax_leg.get_position()                       # [x0, y0, w, h]
new_width = 0.75
ax_leg.set_position([0.5 - new_width / 2,         # x0 → (1-0.85)/2 = 0.075
                     pos.y0,                      # y0 保持不变
                     new_width,                  # w  = 0.85
                     pos.height])                # h  原高度

ax_leg.legend(handles=[line_sel, line_conv, max_marker],
                 labels=["Max Methanol Selectivity",
                         "Max CO₂ Conversion",
                         "Max Combined Score"],
                 loc="center", ncol=3, frameon=False, fontsize=12)


plt.tight_layout()
plt.savefig("final_autolayout_3d_only.jpg", dpi=700, bbox_inches="tight", pad_inches=0)
plt.show()
