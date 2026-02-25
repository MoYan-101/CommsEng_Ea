import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 常量 ----------
CSV_PATH      = "Test_0611_cleansed.csv"   # ← CSV 文件
LN_BIN_STEP   = 0.25                       # ln bin 宽
MAX_BINS      = 120                        # 横轴最大 bins
CMAP_NAME     = "YlGnBu_r"                 # Nature 蓝绿渐变

LABEL_FS      = 14   # 轴标签字号
TICK_FS       = 8    # 坐标刻度字号
STEP_LABEL    = 3    # “显示一个 → 跳 STEP_LABEL-1 个”
OFFSET        = 1    # 从第 OFFSET+1 个刻度开始显示

# ===== 1. 读取并展开 =====
cols = ["Promoter 1", "Promoter 2",
        "Promoter 1 ratio (Promoter 1:Cu)",
        "Promoter 2 ratio (Promoter 2:Cu)"]
df = pd.read_csv(CSV_PATH, usecols=cols)

long_df = pd.concat([
    df.rename(columns={"Promoter 1": "promoter",
                       "Promoter 1 ratio (Promoter 1:Cu)": "ratio"})[["promoter", "ratio"]],
    df.rename(columns={"Promoter 2": "promoter",
                       "Promoter 2 ratio (Promoter 2:Cu)": "ratio"})[["promoter", "ratio"]]
])

long_df = long_df[
    long_df["promoter"].notna() &
    long_df["ratio"].notna() &
    (long_df["ratio"] > 0)
].copy()

long_df["ln_ratio"] = np.log(long_df["ratio"].astype(float))

# ===== 2. 自适应 ln 分箱 =====
ln_min, ln_max = long_df["ln_ratio"].min(), long_df["ln_ratio"].max()
bin_step = LN_BIN_STEP
while (ln_max - ln_min) / bin_step > MAX_BINS:
    bin_step *= 2

bins = np.arange(np.floor(ln_min/bin_step)*bin_step,
                 np.ceil (ln_max/bin_step)*bin_step + bin_step,
                 bin_step)

bin_labels = [f"({bins[i]:.1f},{bins[i+1]:.1f})"
              for i in range(len(bins)-1)]
long_df["ln_bin"] = pd.cut(long_df["ln_ratio"], bins=bins,
                           labels=bin_labels, include_lowest=True)

# ===== 3. 百分比矩阵 =====
pivot_pct = (long_df.groupby(["promoter", "ln_bin"])
             .size().unstack(fill_value=0).astype(float)
             .pipe(lambda tbl: tbl.div(tbl.sum(axis=1), axis=0))
             .fillna(0))

# ===== 4. 绘图 =====
n_rows = len(pivot_pct)
plt.figure(figsize=(8, min(0.23*n_rows + 1.5, 9)))

sns.set(style="white",
        rc={"axes.edgecolor": "black", "axes.linewidth": 1.2},
        font_scale=1.05)

ax = sns.heatmap(
    pivot_pct,
    cmap=sns.color_palette(CMAP_NAME, as_cmap=True),
    vmin=0, vmax=1,
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Fraction of samples'},
    xticklabels=True, yticklabels=True, annot=False
)

# —— 横轴标签：隔 STEP_LABEL 显示，带 OFFSET ——
full_labels = pivot_pct.columns.tolist()
display = [
    lbl if ((i - OFFSET) % STEP_LABEL == 0) else ""
    for i, lbl in enumerate(full_labels)
]
ax.set_xticklabels(display, rotation=30, ha='right', fontsize=TICK_FS)

# —— 纵轴刻度字号统一 ——
ax.set_yticklabels(ax.get_yticklabels(), fontsize=TICK_FS)

# 轴标签
ax.set_xlabel("Promoter/Cu Ratio (ln bins)", fontsize=LABEL_FS)
ax.set_ylabel("Promoter", fontsize=LABEL_FS)

# 无标题
ax.set_title("")

# 色条并加框线
cbar = ax.collections[0].colorbar
cbar.outline.set_visible(True)
cbar.outline.set_linewidth(1.2)
cbar.outline.set_edgecolor("black")
# ↓ 这里加 labelrotation=90 （负值 -90 也可，方向相同）
cbar.ax.tick_params(labelsize=TICK_FS+3, labelrotation=90)

cbar.set_label("Fraction of samples", fontsize=LABEL_FS)

plt.tight_layout(pad=0.35)
plt.savefig("promoter_ratio_heatmap_ln.jpg", dpi=700, bbox_inches="tight")
plt.show()
