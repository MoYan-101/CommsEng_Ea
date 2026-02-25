import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== 读取数据 & 提取字段 =====
df = pd.read_csv("DATA_VIS_YEAR.csv")
df_plot = df[[
    "year",
    "Temperature (°C)",
    "Methanol selectivity (%)",
    "CO2 conversion efficiency (%)"
]].dropna()
df_plot.columns = ["year", "temperature", "methanol_selectivity", "co2_conversion"]

# ===== 在最大选择性样本中找出 CO₂ 转化率最高的那个点 =====
max_selectivity = df_plot["methanol_selectivity"].max()
subset = df_plot[df_plot["methanol_selectivity"] == max_selectivity]
max_idx = subset["co2_conversion"].idxmax()

x_max = df_plot.loc[max_idx, "year"]
y_max = df_plot.loc[max_idx, "co2_conversion"]
t_max = df_plot.loc[max_idx, "temperature"]
s_max = df_plot.loc[max_idx, "methanol_selectivity"]

# ===== 点大小缩放并加最小值限制 =====
base = 15  # 所有点至少有 5 的视觉大小
size_scaled = (df_plot["methanol_selectivity"] + base)
size_scaled = (size_scaled / size_scaled.max()) * 200


# ===== 设置风格 =====
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["axes.edgecolor"] = "#333333"
mpl.rcParams["axes.linewidth"] = 1.2

# ===== 绘图准备 =====
fig, ax = plt.subplots(figsize=(12, 8))
ax.grid(False)

# ===== 散点图 =====
sc = ax.scatter(
    df_plot["year"],
    df_plot["co2_conversion"],
    c=df_plot["temperature"],
    s=size_scaled,
    cmap="plasma",
    alpha=0.6,
    edgecolors="none",
    marker="o"
)

# ===== 每年最大选择性和转化率点，构造折线 =====
selectivity_max_points = df_plot.loc[df_plot.groupby("year")["methanol_selectivity"].idxmax()]
conversion_max_points = df_plot.loc[df_plot.groupby("year")["co2_conversion"].idxmax()]

# 计算滑动平均或直接连线
ax.plot(selectivity_max_points["year"], selectivity_max_points["co2_conversion"],
        color="#1f77b4", linestyle="-", linewidth=2, label="Max Methanol Selectivity")

ax.plot(conversion_max_points["year"], conversion_max_points["co2_conversion"],
        color="#d62728", linestyle="--", linewidth=2, label="Max CO₂ Conversion")


# ===== 添加最大点标记和虚线（不加入图例）=====
ax.plot(x_max, y_max, marker='+', markersize=12, color='black', markeredgewidth=2)
ax.axhline(y=y_max, color='gray', linestyle='--', linewidth=1, zorder=0)
ax.axvline(x=x_max, color='gray', linestyle='--', linewidth=1, zorder=0)

# ===== 添加最大点说明文本 =====
label_text = f"T = {t_max:.1f} °C\nMS = {s_max:.1f} %\nCCE = {y_max:.1f} %"
ax.text(x_max + 0.2, y_max + 0.15, label_text, fontsize=8, va='bottom', ha='left', color='blue')

# ===== 图例、坐标轴和颜色条 =====
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("CO₂ Conversion Efficiency (%)", fontsize=14)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Temperature (°C)", fontsize=12)

ax.legend(loc="upper left", fontsize=14, frameon=True)

# ===== 自适应纵坐标范围 =====
y_margin = (df_plot["co2_conversion"].max() - df_plot["co2_conversion"].min()) * 0.015
ax.set_ylim(df_plot["co2_conversion"].min() - y_margin, df_plot["co2_conversion"].max() + y_margin)

# ===== 显示完整边框 =====
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color("#333333")
    ax.spines[spine].set_linewidth(1.2)

plt.tight_layout()
plt.savefig("vis_by_year.jpg", dpi=700)
plt.show()
