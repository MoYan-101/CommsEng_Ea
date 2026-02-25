"""
utils.py

包含所有绘图函数 & 一些辅助:
1) correlation_heatmap (含普通 & onehot)
2) 训练可视化: loss_curve, scatter(MAE/MSE), residual, feature_importance, etc.
3) 原始数据分析(kde, scatter, boxplot)
4) 推理可视化(2D Heatmap + ConfusionMatrix)
5) 混淆矩阵中在每个三角形内显示数值 + colorbar范围扩展 + 保持正方形布局.

已去掉K-Fold, 保留注释.
"""

from __future__ import annotations
import os
import re
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import PolyCollection
from matplotlib.artist import Artist
from matplotlib.projections.polar import PolarAxes
import pandas as pd
import math
from matplotlib.patches import Patch, Polygon, Rectangle
from sklearn.metrics import r2_score
from matplotlib.ticker import MaxNLocator, FormatStrFormatter  # 如果文件顶部已经导入可省略
import matplotlib.ticker as ticker
import shap  # type: ignore[reportMissingImports]
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[reportUnusedImport]
from matplotlib.lines import Line2D
# from matplotlib import colors
from scipy.ndimage import zoom
import warnings
from typing import cast
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global font settings for all plots.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

#save pt
def _resolve_run_id(run_id: str | None) -> str | None:
    rid = run_id if run_id not in (None, "") else os.environ.get("RUN_ID")
    if rid is None:
        return None
    rid = str(rid).strip()
    if not rid or rid.lower() in {"none", "null"}:
        return None
    return rid

def get_run_id(config: dict | None = None) -> str | None:
    rid = os.environ.get("RUN_ID")
    if rid:
        return rid
    if config:
        cfg_rid = config.get("run_id") or config.get("data", {}).get("run_id")
        if cfg_rid:
            return str(cfg_rid).strip()
    return None

def get_model_dir(csv_name: str, model_type: str, run_id: str | None = None) -> str:
    """统一返回  ./models/<csv_name>[/<run_id>]/<model_type>  目录"""
    rid = _resolve_run_id(run_id)
    parts = ["./models", csv_name]
    if rid:
        parts.append(rid)
    parts.append(model_type)
    return os.path.join(*parts)

def get_root_model_dir(csv_name: str, run_id: str | None = None) -> str:
    """返回  ./models/<csv_name>[/<run_id>]  根目录（metadata / 每类模型子目录）"""
    rid = _resolve_run_id(run_id)
    parts = ["./models", csv_name]
    if rid:
        parts.append(rid)
    return os.path.join(*parts)

def get_postprocess_dir(csv_name: str, run_id: str | None = None, *parts: str) -> str:
    rid = _resolve_run_id(run_id)
    base = ["postprocessing", csv_name]
    if rid:
        base.append(rid)
    base.extend(parts)
    return os.path.join(*base)

def get_eval_dir(csv_name: str, run_id: str | None = None, *parts: str) -> str:
    rid = _resolve_run_id(run_id)
    base = ["evaluation", "figures", csv_name]
    if rid:
        base.append(rid)
    base.extend(parts)
    return os.path.join(*base)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def ensure_dir_for_file(filepath):
    dir_ = os.path.dirname(filepath)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

def normalize_data(data, vmin, vmax):
    """归一化数据到 [0,1] 范围"""
    return (data - vmin) / (vmax - vmin) if vmax > vmin else data

def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

# --------------- correlation ---------------
def short_label(s: str) -> str:
    """
    根据下划线将字符串分割，取最后一段，并做以下处理：
      - 若最后一段全是大写 (如 "CO", "OH")，则原样返回
      - 特定化学符号进行特例转换（如 "cu(oh)2" -> "Cu(OH)2"）
      - 其余情况：首字母大写，其他部分保持原状
    """
    special_chemicals = {
        # --- 单纯化学式：\mathrm 保持直立 ---
        "cu":  r"$\mathrm{Cu}$",
        "cu(oh)2": r"$\mathrm{Cu(OH)_{2}}$",
        "cuxo":    r"$\mathrm{Cu_{X}O}$",
        "cu2s":    r"$\mathrm{Cu_{2}S}$",
        "cu2(oh)2co3": r"$\mathrm{Cu_{2}(OH)_{2}CO_{3}}$",
        "c2+": r"$\mathrm{C_{2+}}$",
        "c1":  r"$\mathrm{C_{1}}$",
        "h2":  r"$\mathrm{H_{2}}$",

        # --- 长文本也放在 \mathrm{}，空格用 '\ ' ---
        "catalyst surface area (m2/g) (ln scale)":
            r"$\mathrm{Catalyst\ surface\ area\ (m^{2}/g)\ (LN\ scale)}$",

        "h2/co2 ratio (-)":
            r"$\mathrm{H_{2}/CO_{2}\ ratio\ (-)}$",

        "ch3oh (g/kg·h) (ln scale)":
            r"$\mathrm{STY\_CH_{3}OH\ (g/kg\!\cdot\!h)\ (LN\ scale)}$",

        "co2 conversion efficiency (%)":
            r"$\mathrm{CO_{2}\ conversion\ efficiency\ (\%)}$"
}


    s = str(s)
    parts = s.split('_')
    last_part = parts[-1]  # 取最后一段

    # 若最后一段是空字符串
    if not last_part:
        return s  # 避免空标签

    # 先检查是否属于特例化学符号
    lower_last_part = last_part.lower()  # 转小写匹配
    if lower_last_part in special_chemicals:
        return special_chemicals[lower_last_part]

    # 如果最后一段全是大写 (含数字/符号不影响 isupper，只要字母全大写即可)
    if last_part.isupper():
        return last_part

    # 否则，仅将首字母转大写，其余部分保持原状
    return last_part[0].upper() + last_part[1:]

def only_positive_formatter(x, pos):
    """
    自定义 Formatter:
      - 当 x<=0 时，返回空字符串 => 不显示刻度文字
      - 当 x>0 时，显示 x 的浮点数(保留两位小数或自行调整)
    """
    if x <= 0:
        return ""
    else:
        return f"{x:.2f}"

# ------------------------------------------------------------------------------
# Correlation‑Network Heatmap  (feature × feature, MIC / distance‑corr)
# ------------------------------------------------------------------------------

def plot_mic_network_heatmap(feature_df: pd.DataFrame,
                             filename: str,
                             method: str = "mic",
                             dpi: int = 700) -> None:
    """
    Parameters
    ----------
    feature_df : pd.DataFrame
    filename   : str
    method     : {"mic", "distance"}
        - "mic"      → Maximal Information Coefficient
        - "distance" → distance‑correlation（dcor）
    dpi        : int
        dpi default  → 700。
    """

    # ------------------------------------------------------------------
    # 0. 内部工具
    # ------------------------------------------------------------------
    def _load_style() -> None:
        try:
            plt.style.use("chartlab.mplstyle")
        except Exception:
            pass
        plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["xtick.major.size"] = 0
        plt.rcParams["ytick.major.size"] = 0

    def _gradient_color(min_v: float, max_v: float,
                        palette: list[str], v: float) -> str:
        """线性插色，返回 HEX"""
        if max_v == min_v:
            return palette[len(palette) // 2]
        t = np.clip((v - min_v) / (max_v - min_v), 0, 1)
        i = int(t * (len(palette) - 1))
        c0 = mcolors.to_rgb(palette[i])
        c1 = mcolors.to_rgb(palette[min(i + 1, len(palette) - 1)])
        blend = tuple((1 - (t % 1)) * s + (t % 1) * e for s, e in zip(c0, c1))
        return mcolors.to_hex(blend)

    def _mic(x: np.ndarray, y: np.ndarray) -> float:
        """MIC；若 minepy 不可用则退化为 distance‑correlation / Pearson"""
        try:
            from minepy import MINE  # type: ignore[reportMissingImports]
            mine = MINE(alpha=0.6, c=15)
            mine.compute_score(x, y)
            return float(mine.mic())
        except Exception:
            try:
                import dcor  # type: ignore[reportMissingImports]
                return float(dcor.distance_correlation(x, y))
            except Exception:
                val = float(np.corrcoef(x, y)[0, 1])
                return 0.0 if np.isnan(val) else abs(val)

    def _corr_matrix(df: pd.DataFrame, _method: str = "mic") -> np.ndarray:
        """计算 n×n 非线性相关矩阵（MIC 或 distance‑corr）"""
        n = df.shape[1]
        mat = np.eye(n)
        to_num = lambda s: s.to_numpy(float) if pd.api.types.is_numeric_dtype(s) \
                           else s.astype("category").cat.codes.to_numpy(float)
        for i in range(n):
            xi = to_num(df.iloc[:, i])
            for j in range(i + 1, n):
                xj = to_num(df.iloc[:, j])
                m  = min(len(xi), len(xj))
                if _method == "mic":
                    score = _mic(xi[:m], xj[:m])
                else:
                    # distance‑correlation（若 dcor 不在则用 |Pearson|）
                    try:
                        import dcor  # type: ignore[reportMissingImports]
                        score = abs(dcor.distance_correlation(xi[:m], xj[:m]))
                    except Exception:
                        score = abs(np.corrcoef(xi[:m], xj[:m])[0, 1])
                mat[i, j] = mat[j, i] = np.clip(score, 0, 1)
        return mat

    # ------------------------------------------------------------------
    # 1. 准备数据与配色
    # ------------------------------------------------------------------
    _load_style()
    palette = ["#515a85", "#c0627a"]

    C         = _corr_matrix(feature_df, method)
    feat_full = feature_df.columns.to_list()
    feat_lbl = []
    for c in feat_full:
        lbl = short_label(c)
        feat_lbl.append(lbl)
    n         = len(feat_full)

    # ------------------------------------------------------------------
    # 2. 绘图
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(8, .55 * n) + 3,  # ← 右侧多留一些空
                                    max(6, .55 * n)),
                           dpi=dpi)

    # 方块（对角含 1，主对角线以上画一次即可）
    for i in range(n):
        for j in range(i, n):
            s   = C[i, j]
            col = _gradient_color(0, 1, palette, s)
            # 背景白框
            ax.add_patch(mpatches.Rectangle((n - i - 1, j), 1, 1,
                                            edgecolor="#999999", facecolor="#ffffff",
                                            linewidth=.25))
            # 比例方块
            ax.add_patch(mpatches.Rectangle((n - i - 0.5 - s / 2, j + 0.5 - s / 2),
                                            s, s,
                                            edgecolor="#999999", facecolor=col,
                                            linewidth=.5))

        # ---------------------------- 轴标签 ----------------------------
        # 只保留右侧纵向标签；旋转 30°
        for k, lab in enumerate(feat_lbl):
            ax.text(n + 0.6,  # 稍微往右挪一点 0.5 → 0.6
                    n - 0.5 - k,
                    lab,
                    ha="left",  # 让文字从左起
                    va="center",
                    fontsize=10)

    # 颜色条
    # cmap  = mcolors.LinearSegmentedColormap.from_list("mic_cmap", palette)
    # cb_ax = fig.add_axes([0.09, 0.15, 0.03, 0.25])
    # cb    = plt.colorbar(cm.ScalarMappable(cmap=cmap,
    #                                        norm=mcolors.Normalize(0, 1)),
    #                      cax=cb_ax)
    # cb.ax.set_title(f"{method.upper()}\nCorr", fontsize=9, pad=8)

        # -------------------- 颜色条：放到左下角 --------------------
        # ① 位置：把原来的 [0.89, 0.15, 0.03, 0.25]
        #    改成左下角一个小条；示例占据图宽 25%、高 3%
        cb_ax = fig.add_axes((0.15, 0.25, 0.25, 0.03))  # [left, bottom, width, height]

        # ② 水平 colorbar
        cb = plt.colorbar(
            cm.ScalarMappable(
                cmap=mcolors.LinearSegmentedColormap.from_list("mic_cmap", palette),
                norm=mcolors.Normalize(0, 1)
            ),
            cax=cb_ax,
            orientation="horizontal"
        )

        # ③ 把刻度文字都旋转 45 °
        #    – 标题
        # cb.set_label(f"{method.upper()} Corr", labelpad=4,
        #              fontsize=9, ha="left", va="center")  # (把 ha 调成 'left' 看起来更顺)
        cb.set_label(f"{method.upper()} Corr", fontsize=11)  # (把 ha 调成 'left' 看起来更顺)
        #    – 刻度
        for t in cb.ax.get_xticklabels():
            t.set_rotation(45)
            t.set_horizontalalignment("right")  # 让文字贴着刻度略微往内收
            t.set_fontsize(11)

        # ④ 额外：去掉 colorbar 的外框使更简洁
        for spine in cb.ax.spines.values():
            spine.set_visible(False)

    # 细节 & 保存
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.xaxis.tick_top();    ax.yaxis.tick_right()
    ax.axis("equal")
    ax.set_xlim(-2, n + 1.5)
    ax.set_ylim(-1, n + 1)
    ax.set_title("Correlation Network Heatmap",
                 fontsize=14, pad=18)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    # 去掉整张图的外框线
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(rect=(0, 0, 0.88, 1))
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"[plot_mic_network_heatmap] → {filename}")


# --------------- 训练可视化: Loss, scatter, residual, etc. ---------------
def plot_loss_curve(train_losses, val_losses, filename):
    ensure_dir_for_file(filename)
    plt.figure()
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # plt.title("Training/Validation Loss")
    plt.savefig(filename, dpi=700, format='jpg')
    plt.close()


def plot_joint_scatter_with_marginals(y_true, y_pred, y_labels=None, filename="joint_scatter_with_marginals.jpg"):
    """
    Create joint scatter plots with marginal gradient-filled KDE curves for each output dimension.

    For each output dimension:
      - Display a scatter plot of true vs. predicted values, colored by 2D kernel density.
      - Draw a reference line (dashed red) for perfect prediction (True = Predicted).
      - Add marginal plots above and to the right showing the KDE of true and predicted values.
        The marginal plots are filled with a gradient color that reflects the local KDE value using
        a segment-wise PolyCollection:
           * Top marginal: using the 'Blues' colormap for true values, colored by the KDE density.
           * Right marginal: using the 'Reds' colormap for predicted values, colored by the KDE density.
      - Annotate the main plot with the R² score.

    Parameters:
      y_true : numpy array of true values, shape (N, out_dim)
      y_pred : numpy array of predicted values, shape (N, out_dim)
      y_labels : Optional list of labels for each output dimension.
      filename : Name of the file to save the plot.
    """
    ensure_dir_for_file(filename)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2-dimensional arrays, shaped as (N, out_dim)")

    _, out_dim = y_pred.shape
    fig, axes = plt.subplots(1, out_dim, figsize=(5.3 * out_dim, 5.15), squeeze=False)

    # Define colormaps for main and marginal plots.
    cmap_main = cm.get_cmap('cividis')
    cmap_x = cm.get_cmap('Blues')
    cmap_y = cm.get_cmap('Reds')

    for i in range(out_dim):
        # Extract data for current dimension.
        x = y_true[:, i]
        y = y_pred[:, i]
        ax = axes[0, i]

        # Compute R² score.
        r2_val = r2_score(x, y)

        # Main scatter plot: colored by 2D kernel density.
        xy = np.vstack([x, y])
        kde_xy = gaussian_kde(xy)
        density_xy = kde_xy(xy)
        sc = ax.scatter(x, y, c=density_xy, cmap=cmap_main, alpha=0.5, edgecolor='none')

        # Draw perfect prediction line using data min and max.
        min_val = float(np.min([np.min(x), np.min(y)]))
        max_val = float(np.max([np.max(x), np.max(y)]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

        # Set axis labels and title，字体统一设置为 fontsize=10
        if y_labels and i < len(y_labels):
            # ax.set_title(f"{y_labels[i]} (MAE)", fontsize=16)
            ax.set_xlabel(f"True {y_labels[i]}", fontsize=17)
            ax.set_ylabel(f"Predicted {y_labels[i]}", fontsize=17)

        else:
            # ax.set_title(f"Out {i} (MAE)", fontsize=16)
            ax.set_xlabel("True Value", fontsize=17)
            ax.set_ylabel("Predicted Value", fontsize=17)
        # ★ 这一行放在 set_xlabel / set_ylabel 之后即可
        ax.tick_params(axis="both", labelsize=17)
        # Annotate R² in the main plot.
        ax.text(0.05, 0.95, f"R² = {r2_val:.3f}", transform=ax.transAxes,
                fontsize=16, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        # Hide the main plot's top and right borders.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Get main axis data coordinate range.
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()

        # Add marginal axes for KDE plots.
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", size="20%", pad=0, sharex=ax)
        ax_histy = divider.append_axes("right", size="20%", pad=0, sharey=ax)
        # Force marginal axes to use the same coordinate range as the main plot.
        ax_histx.set_xlim(current_xlim)
        ax_histy.set_ylim(current_ylim)

        # --- Top marginal (x): gradient-filled KDE for true values ---
        x_vals = np.linspace(current_xlim[0], current_xlim[1], 200)
        if len(x) > 1:
            kde_x = gaussian_kde(x)
            kde_x_vals = kde_x(x_vals)
        else:
            kde_x_vals = np.zeros_like(x_vals)
        segments_x = []
        colors_x = []
        # Use KDE values (density) for color mapping.
        norm_kde_x = mcolors.Normalize(vmin=np.min(kde_x_vals), vmax=np.max(kde_x_vals))
        for j in range(len(x_vals) - 1):
            x0, x1 = x_vals[j], x_vals[j + 1]
            d0, d1 = kde_x_vals[j], kde_x_vals[j + 1]
            segments_x.append([[x0, 0], [x0, d0], [x1, d1], [x1, 0]])
            mid_density = 0.5 * (d0 + d1)
            colors_x.append(cmap_x(norm_kde_x(mid_density)))
        pc_x = PolyCollection(segments_x, facecolors=colors_x, edgecolors='none', alpha=0.8)
        ax_histx.plot(x_vals, kde_x_vals, color='darkblue', linewidth=1.2, alpha=0.5)
        ax_histx.add_collection(pc_x)
        ax_histx.set_ylim(0, np.max(kde_x_vals))
        ax_histx.axis('off')

        # --- Right marginal (y): gradient-filled KDE for predicted values ---
        y_vals = np.linspace(current_ylim[0], current_ylim[1], 200)
        if len(y) > 1:
            kde_y_obj = gaussian_kde(y)
            kde_y_vals = kde_y_obj(y_vals)
        else:
            kde_y_vals = np.zeros_like(y_vals)
        segments_y = []
        colors_y = []
        # Use KDE values (density) for color mapping.
        norm_kde_y = mcolors.Normalize(vmin=np.min(kde_y_vals), vmax=np.max(kde_y_vals))
        for j in range(len(y_vals) - 1):
            y0, y1 = y_vals[j], y_vals[j + 1]
            d0, d1 = kde_y_vals[j], kde_y_vals[j + 1]
            segments_y.append([[0, y0], [d0, y0], [d1, y1], [0, y1]])
            mid_density = 0.5 * (d0 + d1)
            colors_y.append(cmap_y(norm_kde_y(mid_density)))
        pc_y = PolyCollection(segments_y, facecolors=colors_y, edgecolors='none', alpha=0.8)
        ax_histy.plot(kde_y_vals, y_vals, color='darkred', linewidth=1.2, alpha=0.5)
        ax_histy.add_collection(pc_y)
        ax_histy.set_xlim(0, np.max(kde_y_vals))
        ax_histy.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()


class MyScalarFormatter(ticker.ScalarFormatter):
    def __init__(self, useMathText=True):
        super().__init__(useMathText=useMathText)
        # 这里也可以在外部调用 set_powerlimits((0,0)) 来强制科学计数法

    def _set_format(self):
        # 关键：只显示一位小数
        self.format = '%.1f'



# --------------- 原始数据分析 ---------------
def plot_kde_distribution(df, columns, filename):
    """
    Draw KDE plots for each column in `columns`, up to 4 per row, wrapping to new rows
    as needed. Figure size adapts to the grid dimensions.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        List of column names to plot.
    filename : str
        Path to save the figure.
    """
    ensure_dir_for_file(filename)

    n = len(columns)
    if n == 0:
        raise ValueError("No columns provided to plot.")

    # determine grid: max 4 columns per row
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    # let figure size grow with grid (approx 4" per subplot)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)
    axes = np.ravel(axes)

    for i, col in enumerate(columns):
        ax = axes[i]
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}' not in df", ha='center', va='center')
            ax.set_axis_off()
            continue

        # basic KDE
        sns.kdeplot(df[col], ax=ax, fill=False, color="black",
                    clip=(df[col].min(), df[col].max()))

        lines = ax.get_lines()
        if not lines:
            ax.set_title(f"No Data for {col}")
            ax.set_axis_off()
            continue

        line = lines[-1]
        x_plot, y_plot = line.get_xdata(), line.get_ydata()
        idxsort = np.argsort(x_plot)
        x_plot, y_plot = x_plot[idxsort], y_plot[idxsort]

        # build gradient fill under curve
        vmin, vmax = float(np.min(x_plot)), float(np.max(x_plot))
        cmap = cm.get_cmap("coolwarm")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for j in range(len(x_plot) - 1):
            x0, x1 = x_plot[j], x_plot[j + 1]
            y0, y1 = y_plot[j], y_plot[j + 1]
            color = cmap(norm((x0 + x1) * 0.5))
            verts = np.array([[x0, 0], [x0, y0], [x1, y1], [x1, 0]])
            poly = PolyCollection([verts],
                                  facecolors=[color],
                                  edgecolor='none',
                                  alpha=0.6)
            ax.add_collection(poly)

        # labels (slightly larger for data analysis plots)
        label_size = 14
        tick_size = 11
        cb_label_size = 12
        cb_tick_size = 10
        offset_size = 10

        ax.set_xlabel(col, fontsize=label_size)
        ax.set_ylabel("Density", fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.set_xlim(vmin, vmax)

        # y-axis formatting
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        fmt = MyScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(offset_size)

        # colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("Value", fontsize=cb_label_size)
        cb.ax.tick_params(labelsize=cb_tick_size)

    # hide any unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_kde_distribution] => {filename}")

#####################################################
# 自定义的 onehot合并函数
#####################################################
import copy

def merge_onehot_shap(shap_data, onehot_groups, case_map=None):
    """
    将同一类别的 one-hot dummy 列合并成单列，并返回新的 shap_data dict。
    - shap_data: 由 train.py 保存、visualization.py 读入的 dict
                 必须含 "shap_values", "X_full", "x_col_names"
    - onehot_groups: [[7,8,9], [10,11], ...]  每个子列表是一组 dummy 的全局列号
    - case_map:  {lower_name: OriginalName}  ——想还原大小写时传入
    """
    shap_values = shap_data["shap_values"]
    X_full      = shap_data["X_full"]
    col_names   = shap_data["x_col_names"]

    # ------------- 1) 统一成 list -------------
    shap_is_list = isinstance(shap_values, list)
    shap_values = shap_values if shap_is_list else [shap_values]

    # ------------- 2) 建立“保留列”索引 -------------
    flat_oh = {i for g in onehot_groups for i in g}
    keep_idx = [i for i in range(len(col_names)) if i not in flat_oh]

    # ------------- 3) 构造新列名 -------------
    new_col_names = [col_names[i] for i in keep_idx]
    for g in onehot_groups:
        pref = col_names[g[0]].split('_')[0]      # 取前缀作为类别名
        if case_map is not None:
            pref = case_map.get(pref.lower(), pref)
        new_col_names.append(pref)

    # ------------- 4) 合并 SHAP & X_full -------------
    new_shap_list, new_data = [], []
    for sv in shap_values:                        # sv: (n_samples, n_features)
        parts = [sv[:, keep_idx]]
        for g in onehot_groups:
            parts.append(sv[:, g].sum(axis=1, keepdims=True))
        new_shap_list.append(np.hstack(parts))

    if X_full is not None:
        parts_d = [X_full[:, keep_idx]]
        for g in onehot_groups:
            # 取 argmax 的列下标作为类别标识；也可以改成类别字符串
            chosen = (X_full[:, g].argmax(axis=1)).reshape(-1, 1)
            parts_d.append(chosen)
        new_data = np.hstack(parts_d)
    else:
        new_data = None

    # ------------- 5) 封装并返回 -------------
    new_sd = copy.deepcopy(shap_data)
    new_sd["shap_values"] = new_shap_list if shap_is_list else new_shap_list[0]
    new_sd["X_full"]      = new_data
    new_sd["x_col_names"] = new_col_names
    return new_sd

#####################################################
# 1) 自定义的 plot_shap_importance 函数
#####################################################
def plot_shap_importance(
    shap_data,
    output_path,
    top_n_features=15,
    plot_width=12,
    plot_height=8
):
    """
    绘制自定义的 SHAP 特征重要性条形图：
      - 通过计算 mean(|SHAP|) 得到特征重要性
      - 仅展示前 top_n_features
      - 以这 top_n_features 的平均值作为阈值：大于均值 → 蓝色，小于等于均值 → 红色
      - 在图中用阴影和虚线标示该阈值
      - 适配多输出情况（shap_values 为 list）

    shap_data 必须包含:
        "shap_values": array 或 list<array> 形状 (n_samples, n_features)
        "X_full":      形状 (n_samples, n_features)，这里不一定要用到，只要列数对得上即可
        "x_col_names": 特征名列表 (长度 n_features)
        "y_col_names": 输出名列表 (若多输出，对应 shap_values 的每个输出)
    """
    ensure_dir_for_file(os.path.join(output_path, "dummy.txt"))  # 确保目录存在

    shap_values = shap_data["shap_values"]
    X_full = np.asarray(shap_data["X_full"])
    x_col_names = shap_data["x_col_names"]
    y_col_names = shap_data["y_col_names"]

    # short_label 处理
    x_col_names = [short_label(col) for col in x_col_names]
    y_col_names = [short_label(y) for y in y_col_names]

    # 若 shap_values 不是 list，转为单输出的 list 方便处理
    multi_output = True
    if not isinstance(shap_values, list):
        shap_values = [shap_values]
        multi_output = False

    # 对每个输出分别绘图
    for idx, sv in enumerate(shap_values):
        sv_arr = np.asarray(sv)
        # 计算 mean(|SHAP|) 作为特征重要性
        # sv 形状 (n_samples, n_features)
        mean_abs_shap = np.mean(np.abs(sv_arr), axis=0)  # (n_features, )

        # 取 top_n_features
        sorted_idx = np.argsort(mean_abs_shap)[::-1]  # 降序
        top_idx = sorted_idx[:top_n_features]
        top_imps = mean_abs_shap[top_idx]
        top_feats = [x_col_names[i] for i in top_idx]

        # 计算阈值(均值)
        threshold = top_imps.mean()

        # 颜色：大于均值→蓝色，小于等于均值→红色
        colors = ["blue" if imp > threshold else "red" for imp in top_imps]

        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        ax.barh(range(len(top_imps)), top_imps, align='center', color=colors)
        ax.set_yticks(range(len(top_imps)))
        ax.set_yticklabels(top_feats, fontsize=10)
        ax.invert_yaxis()

        # X 轴标签与标题
        ax.set_xlabel("Mean(|SHAP|)", fontsize=12)
        if multi_output:
            out_label = y_col_names[idx] if idx < len(y_col_names) else f"Output{idx}"
            safe_out_label = safe_filename(out_label)  # 对输出标签进行过滤处理
            out_name = f"shap_importance_{safe_out_label}.jpg"
            # 多输出时，根据 idx 对应 y_col_names
            # ax.set_title(f"Mean |SHAP| (Top-{top_n_features}) - {out_label}",
            #              fontsize=14)
        else:
            # ax.set_title(f"Mean |SHAP| (Top-{top_n_features})", fontsize=14)
            out_name = "shap_importance.jpg"

        # 画阴影和竖线
        ax.axvspan(0, threshold, facecolor='lightgray', alpha=0.5)
        ax.axvline(threshold, color='gray', linestyle='dashed', linewidth=2)

        # 图例
        legend_e = [
            Patch(facecolor="blue", label="Above Mean"),
            Patch(facecolor="red", label="Below/Equal Mean")
        ]
        ax.legend(handles=legend_e, loc="lower right", fontsize=12)

        # 显示外框
        for spine in ax.spines.values():
            spine.set_visible(True)

        plt.tight_layout()
        save_path = os.path.join(output_path, out_name)
        plt.savefig(save_path, dpi=700)
        plt.close()
        print(f"[INFO] SHAP importance (custom) saved => {save_path}")


#####################################################
# 2) 自定义的 plot_shap_beeswarm 函数 (加外框)
#####################################################
def plot_shap_beeswarm(
    shap_data,
    output_path,
    top_n_features=15,
    plot_width=12,
    plot_height=8
):
    """
    使用 shap.summary_plot(..., plot_type='dot'/默认) 绘制 beeswarm，
    但在绘图后手动设置外部边框可见。

    参数:
    -------
    shap_data : dict
        包含 "shap_values", "X_full", "x_col_names", "y_col_names"
    top_n_features : int
        最多展示多少个特征
    plot_width, plot_height : float
        控制图像大小
    """
    ensure_dir_for_file(os.path.join(output_path, "dummy.txt"))  # 确保目录存在

    shap_values = shap_data["shap_values"]
    X_full = shap_data["X_full"]
    x_col_names = shap_data["x_col_names"]
    y_col_names = shap_data["y_col_names"]

    # 对特征名、输出名进行 short_label 处理
    x_col_names = [short_label(col) for col in x_col_names]
    y_col_names = [short_label(y) for y in y_col_names]

    # 判断多输出
    multi_output = True
    if not isinstance(shap_values, list):
        shap_values = [shap_values]
        multi_output = False

    for idx, sv in enumerate(shap_values):
        if multi_output:
            out_label = y_col_names[idx] if idx < len(y_col_names) else f"Output{idx}"
            safe_out_label = safe_filename(out_label)  # 对输出标签进行过滤处理
            out_name = f"shap_beeswarm_{safe_out_label}.jpg"
        else:
            out_name = "shap_beeswarm.jpg"

        sv_arr = np.asarray(sv)
        X_full_arr = np.asarray(X_full)
        # 调用 shap.summary_plot 生成 beeswarm
        shap.summary_plot(
            sv_arr,
            features=X_full_arr,
            feature_names=x_col_names,
            show=False,
            max_display=top_n_features,
            plot_size=(plot_width, plot_height)
        )
        # 这时会创建/切换到 shap 的默认 figure/axes
        ax = plt.gca()

        # 使外部边框可见
        for spine in ax.spines.values():
            spine.set_visible(True)

        plt.tight_layout()
        save_path = os.path.join(output_path, out_name)

        plt.savefig(save_path, dpi=700)
        plt.close()
        print(f"[INFO] SHAP beeswarm saved => {save_path}")

STYLE_COLORS = ["#24345C", "#279DE1", "#36CDCB", "#FF7F4C"]


def plot_shap_importance_multi_output(
    shap_data,
    output_path,
    top_n_features=14,
    plot_width=12,
    plot_height=8
):
    """
    绘制多输出(MIMO)模型的 SHAP 特征重要性堆叠条形图:
      - 对每个输出 separately 计算 mean(|SHAP|)，得到 (n_features, ) 各输出
      - 合并 => shape (n_features, M)，每行是特征, 每列是输出
      - 求行和 => 排序 => 取 Top-N
      - 水平堆叠条形图: 每个特征一行，总长度=各输出贡献之和
      - 不同输出用不同颜色，图例显示输出名称

    参数
    ----
    shap_data : dict
        {
          "shap_values": list of arrays or array
                        若是 list，表示多输出: shap_values[i].shape=(n_samples, n_features)
          "x_col_names": list of feature names
          "y_col_names": list of output names (多输出)
          ...
        }
    output_path : str
        图片保存的路径 (文件夹 + 文件名). 函数会在内部创建目录.
    top_n_features : int
        只显示贡献和最高的前 N 个特征
    plot_width, plot_height : float
        图像大小
    """

    ensure_dir_for_file(output_path)

    shap_values = shap_data["shap_values"]  # 可能是 list，也可能是单 array
    x_col_names = shap_data["x_col_names"]
    y_col_names = shap_data.get("y_col_names", None)  # 可能没有

    # 若 shap_values 不是 list => 转成 list，以统一处理多输出
    if not isinstance(shap_values, list):
        shap_values = [shap_values]
    n_outputs = len(shap_values)

    # 如果输出数 > 颜色数，就重复颜色或自行处理
    if n_outputs > len(STYLE_COLORS):
        color_palette = STYLE_COLORS * (n_outputs // len(STYLE_COLORS) + 1)
    else:
        color_palette = STYLE_COLORS[:n_outputs]

    # 简化特征名
    x_col_names = [short_label(f) for f in x_col_names]
    # 若无 y_col_names，就用 "Output1", "Output2"...
    if not y_col_names or len(y_col_names) < n_outputs:
        y_col_names = [f"Output{i+1}" for i in range(n_outputs)]
    else:
        y_col_names = [short_label(y) for y in y_col_names]

    # ============ 1) 计算 mean(|SHAP|) for each output ============
    # shap_values[i] shape = (n_samples, n_features)
    # mean_abs_shap[i, :] => shape=(n_features,)
    # 最终得到 shap_matrix shape=(n_features, n_outputs)
    n_features = len(x_col_names)
    shap_matrix = np.zeros((n_features, n_outputs), dtype=np.float64)

    for i in range(n_outputs):
        sv_i = np.asarray(shap_values[i])  # (n_samples, n_features)
        mean_abs_i = np.mean(np.abs(sv_i), axis=0)  # (n_features,)
        shap_matrix[:, i] = mean_abs_i

    # ============ 2) 求行和 => 选 Top-N 特征 ============
    sum_importances = np.sum(shap_matrix, axis=1)  # (n_features,)
    # 按 sum_importances 降序排序
    sorted_idx = np.argsort(sum_importances)[::-1]
    top_idx = sorted_idx[:top_n_features]

    # ============ 3) 堆叠条形图 (Stacked Barh) ============
    # 取出 top_n_features
    top_features = [x_col_names[i] for i in top_idx]
    # shap_matrix_top shape=(top_n, n_outputs)
    shap_matrix_top = shap_matrix[top_idx, :]
    # sum_top shape=(top_n,)
    sum_top = sum_importances[top_idx]

    # 依据 sum_top 降序 => top_idx 可能已是
    # 这里把 shap_matrix_top row 也按 sum_top 排序
    # sum_top 已经是降序
    # 只需 invert yaxis 后, 就是从上往下
    # or we can keep as is, no problem

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    # y 的刻度: 0..top_n-1
    # 画 barh, each bar 堆叠 n_outputs segment
    for rank in range(top_n_features):
        f_idx = top_idx[rank]
        # segments = shap_matrix_top[rank, :]
        segments = shap_matrix[f_idx, :]
        left_acc = 0.0
        for i in range(n_outputs):
            val_i = segments[i]
            ax.barh(
                y=rank,
                width=val_i,
                left=left_acc,
                color=color_palette[i],  # 顶刊配色
                alpha=0.7
            )
            left_acc += val_i

    # y ticks => top_features
    ax.set_yticks(np.arange(top_n_features))
    ax.set_yticklabels([top_features[i] for i in range(top_n_features)], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Sum of mean(|SHAP|) across outputs", fontsize=12)
    # ax.set_title(f"Multi-output SHAP Feature Importance (Top-{top_n_features})", fontsize=14)

    # x axis => up to max of sum_top
    ax.set_xlim(0, np.max(sum_top)*1.05)

    # ============ 4) 图例: n_outputs => y_col_names + color = color_palette[i] ============
    legend_patches = []
    for i in range(n_outputs):
        patch = Patch(facecolor=color_palette[i], label=y_col_names[i], alpha=0.7)
        legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc="lower right", fontsize=10, frameon=False)

    # 美化
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=700)
    plt.close()
    print(f"[plot_shap_importance_multi_output] => {output_path}")


# ---------------------- 新增 combined SHAP 函数 ---------------------
def plot_shap_combined(
    shap_data,
    output_path,
    top_n_features=15,
    plot_width=12,
    plot_height=8
):
    """
    在同一张图中绘制：
      - 共享的左侧纵坐标（特征）
      - 底部 x 轴：beeswarm 图（展示各样本的 SHAP 值分布），
        并调用 invert_yaxis() 使得视觉上最高的特征在最上。
      - 上部 x 轴：水平条形图，绘制 mean(|SHAP|)（带透明度、阈值阴影和虚线），
        数据顺序反转，使得条形图顺序与 beeswarm 图一致（最高的特征在上）。
      - 右侧坐标轴隐藏
    输出：保存为 JPG 格式，dpi=700。

    参数
    ----------
    shap_data : dict
        必须包含：
          - "shap_values": np.array，形状 (n_samples, n_features)
          - "X_full":      np.array，形状 (n_samples, n_features)
          - "x_col_names": 特征名列表，长度 = n_features
        可选：
          - "y_col_names": 若多输出则自行处理，这里示例仅单输出
    output_path : str
        图片保存的文件夹路径
    top_n_features : int
        仅绘制前 top_n_features 个最重要的特征
    plot_width, plot_height : float
        图像宽度和高度（单位英寸）
    """
    ensure_dir_for_file(os.path.join(output_path, "dummy.txt"))
    # 读取数据
    shap_values = shap_data["shap_values"]
    X_full = shap_data["X_full"]
    x_col_names = shap_data["x_col_names"]

    # 简化特征名
    x_col_names = [short_label(col) for col in x_col_names]

    # 如果 shap_values 不是 list，则转换成 list（示例假设单输出）
    if not isinstance(shap_values, list):
        shap_values = [shap_values]
        multi_output = False
    else:
        multi_output = True

    for idx, sv in enumerate(shap_values):
        sv_arr = np.asarray(sv)
        # -------------------------------
        # 1. 计算每个特征的平均绝对 SHAP 值
        # -------------------------------
        mean_abs_shap = np.squeeze(np.mean(np.abs(sv_arr), axis=0))  # (n_features,)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]             # 降序排序：第一项最高
        top_idx = sorted_idx[:top_n_features]
        top_idx = [int(i) for i in top_idx]  # 转换为标准整型

        top_imps = mean_abs_shap[top_idx]
        top_feats = [x_col_names[i] for i in top_idx]

        # 重新排列 SHAP 值和原始数据，使得 beeswarm 图按同一顺序显示
        sv_sorted = sv_arr[:, top_idx]
        X_sorted = X_full[:, top_idx]
        feat_sorted = top_feats[:]  # 复制

        # 构造 Explanation 对象（新版 SHAP 推荐用法）
        explanation = shap.Explanation(
            values=sv_sorted,
            base_values=None,
            data=X_sorted,
            feature_names=feat_sorted
        )

        # 阈值（平均值）以及设定颜色：大于阈值为蓝，小于等于为红
        threshold = top_imps.mean()
        colors = ["blue" if imp > threshold else "red" for imp in top_imps]

        # -------------------------------
        # 2. 创建单一坐标系（单轴）
        # -------------------------------
        fig, ax_bottom = plt.subplots(figsize=(plot_width, plot_height), dpi=700)
        try:
            plt.sca(ax_bottom)
            shap.summary_plot(
                sv_sorted,
                features=X_sorted,
                feature_names=feat_sorted,
                max_display=top_n_features,
                show=False,
                plot_size=None,
                plot_type="dot",
                sort=False
            )
        except TypeError:
            plt.sca(ax_bottom)
            shap.summary_plot(
                sv_sorted,
                features=X_sorted,
                feature_names=feat_sorted,
                max_display=top_n_features,
                show=False,
                plot_type="dot",
                sort=False
            )
        ax_bottom.set_xlabel("SHAP Value", fontsize=12)
        ax_bottom.spines['right'].set_visible(False)
        # 为确保 beeswarm 图的顺序是从上到下最高的特征在上
        # ax_bottom.invert_yaxis()

        # -------------------------------
        # 3. 添加上部横坐标（通过 twiny），绘制条形图
        # -------------------------------
        ax_top = ax_bottom.twiny()  # 共享 y 轴
        ax_top.set_ylim(ax_bottom.get_ylim())
        y_labels = [t.get_text() for t in ax_bottom.get_yticklabels()]
        y_ticks = ax_bottom.get_yticks()
        imp_map = {feat_sorted[i]: float(top_imps[i]) for i in range(len(top_imps))}
        bar_imps = [imp_map.get(lbl, 0.0) for lbl in y_labels]
        bar_colors = ["blue" if v > threshold else "red" for v in bar_imps]
        ax_top.barh(
            y=y_ticks,
            width=bar_imps,
            color=bar_colors,
            alpha=0.15,
            align='center'
        )
        ax_top.axvspan(0, threshold, facecolor='lightgray', alpha=0.3)
        ax_top.axvline(threshold, color='gray', linestyle='dashed', linewidth=2)
        ax_top.xaxis.set_label_position('top')
        ax_top.xaxis.tick_top()
        ax_top.set_xlabel("Feature Importance", fontsize=12)
        ax_top.set_yticks(y_ticks)
        ax_bottom.set_yticks(y_ticks)
        ax_bottom.set_yticklabels(y_labels)
        # 注意：不再调用 invert_yaxis()在 ax_top 上

        # 添加图例（基于条形图所用颜色）
        legend_e = [
            Patch(facecolor=to_rgba("blue", alpha=0.15), label="Above Mean"),
            Patch(facecolor=to_rgba("red", alpha=0.15), label="Below/Equal Mean")
        ]
        ax_top.legend(handles=legend_e, loc="lower right", fontsize=12)

        # -------------------------------
        # 4. 添加整体标题与保存图片
        # -------------------------------
        if multi_output:
            out_label = shap_data["y_col_names"][idx] if idx < len(shap_data["y_col_names"]) else f"Output{idx}"
            safe_out_label = safe_filename(out_label)
            out_file = f"shap_combined_{safe_out_label}.jpg"
            # fig.suptitle(f"SHAP Combined Plot - {out_label}", fontsize=16)
        else:
            out_file = "shap_combined.jpg"
            # fig.suptitle("SHAP Combined Plot", fontsize=16)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        save_path = os.path.join(output_path, out_file)
        plt.savefig(save_path, dpi=700, format='jpg')
        plt.close(fig)
        print(f"[INFO] SHAP combined figure saved => {save_path}")
# ============ 主要函数 ============
# ------------------------------------------------------------------
# -----------------------------------------------------------------
def plot_local_shap_force(shap_data, sample_index, output_path,
                          top_n_features=8, outputID=0,
                          pos_color="#d9534f", neg_color="#1766b5",
                          bar_height=1.3, dpi=700):
    """
    Pure‑Matplotlib local SHAP force‑plot:
    • Top‑N + Other, no feature names
    • Baseline & f(x) arrows
    • Arrow shapes follow shap/_force_matplotlib.py (v0.44)
    """

    # ========== 1. 数据 ==========
    sv = shap_data["shap_values"]
    vals = sv[outputID][sample_index] if isinstance(sv, list) else sv[sample_index]
    vals = np.nan_to_num(np.asarray(vals, float))

    bv = shap_data.get("base_values", 0.)
    base = float(bv[sample_index]) if isinstance(bv, (list, np.ndarray, pd.Series)) else float(bv)

    # ========== 2. Top‑N + Other ==========
    idx_sorted = np.argsort(np.abs(vals))[::-1]
    top_idx, rest_idx = idx_sorted[:top_n_features], idx_sorted[top_n_features:]
    top_vals = vals[top_idx]
    if rest_idx.size:
        top_vals = np.append(top_vals, vals[rest_idx].sum())

    # Top 部分再按 |v| 降序，Other 留最后
    if rest_idx.size:
        sort_part = np.argsort(np.abs(top_vals[:-1]))[::-1]
        top_vals = np.concatenate([top_vals[:-1][sort_part], top_vals[-1:]])

    pos_vals = top_vals[top_vals > 0]
    neg_vals = top_vals[top_vals < 0]

    # ========== 3. 箭头参数 ==========
    total_neg, total_pos = neg_vals.sum(), pos_vals.sum()
    x_span = abs(total_neg) + total_pos
    head_len_const = max(x_span / 200.0, 0.02)   # pixels in data‑coords

    def head_len(v):
        """避免小箭头整个被头占满；官方做法近似如此"""
        h = min(abs(v) * 0.4, head_len_const)
        return max(h, 0.3 * abs(v))              # 保证 <= 0.7*|v|

    # ========== 4. 绘图 ==========
    fig, ax = plt.subplots(figsize=(13, 1.8), dpi=dpi)

    # ---- 4A. 负侧：从 0 往左累积（反序，以保证靠 baseline 的最后画） ----
    p = 0.0
    for v in sorted(neg_vals, key=abs):          # 从小 |v| → 大 |v|
        h = head_len(v)
        # 矩形
        rect_w = abs(v) - h
        rect_start = p - rect_w
        ax.add_patch(Rectangle((rect_start, -bar_height/2),
                               rect_w, bar_height,
                               color=neg_color, lw=0))
        # 三角
        tri = [[rect_start,  bar_height/2],
               [rect_start - h, 0],
               [rect_start, -bar_height/2]]
        ax.add_patch(Polygon(tri, closed=True, color=neg_color, lw=0))
        # 文本
        center = p - abs(v)/2
        ax.text(center, 0, f"{v:+.2f}", color="white",
                ha="center", va="center", fontsize=8, rotation=90)
        p -= abs(v)                               # 更新指针

    # ---- 4B. 正侧：从 0 往右累积 ----
    p = 0.0
    for v in sorted(pos_vals, key=abs, reverse=True):  # 从大 |v| → 小 |v|
        h = head_len(v)
        rect_w = v - h
        ax.add_patch(Rectangle((p, -bar_height/2),
                               rect_w, bar_height,
                               color=pos_color, lw=0))
        tri = [[p + rect_w,  bar_height/2],
               [p + rect_w + h, 0],
               [p + rect_w, -bar_height/2]]
        ax.add_patch(Polygon(tri, closed=True, color=pos_color, lw=0))
        center = p + v/2
        ax.text(center, 0, f"{v:+.2f}", color="white",
                ha="center", va="center", fontsize=8, rotation=90)
        p += v                                    # 更新指针

    # ========== 5. baseline & f(x) ==========
    ax.axvline(0, ls="--", lw=1.4, color="gray", zorder=0)
    fx = base + vals.sum()
    arrow_y = bar_height * 1.25
    ax.annotate("",
        xy=(fx, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="black"))
    ax.text(0,  arrow_y + 0.12, f"base={base:.2f}",
            ha="left", va="bottom", fontsize=10, color="gray")
    ax.text(fx, arrow_y + 0.12, f"f(x)={fx:.2f}",
            ha="right", va="bottom", fontsize=10)

    # ========== 6. 轴 & 保存 ==========
    pad = 0.05 * x_span
    ax.set_xlim(total_neg - pad, total_pos + pad)
    ax.set_ylim(-1.2, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("SHAP contribution (positive → right, negative → left)",
                  fontsize=11)
    # ax.set_title(f"Local SHAP Force Plot (sample {sample_index})",
    #              fontsize=14, color="#1f77b4", pad=6)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.grid(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Local SHAP force‑plot saved → {output_path}")


def plot_local_shap_lines(shap_data, sample_indices, output_path, top_n_features=8, link="identity", lineID=0, outputID=0):
    """
    绘制局部 SHAP 解释图（Decision Plot 风格），展示多个样本（sample_indices）的 SHAP 解释，
    在同一张图中绘制多条样本的决策曲线，便于比较不同样本的贡献路径，同时增加“Other”贡献部分，
    显示为类似“85 other features”这种风格，并强制将“Other”固定放在最后（即最终显示在图的底部）。
    """
    shap_values = shap_data["shap_values"]
    X_full = shap_data["X_full"]
    feature_names = shap_data["x_col_names"]
    # 对特征名进行简化
    feature_names = [short_label(f) for f in feature_names]
    base_values = shap_data.get("base_values", None)

    if isinstance(shap_values, list):
        shap_value = shap_values[outputID]
    else:
        shap_value = shap_values

    # 收集指定样本的 SHAP 值和数据
    selected_shap = []
    selected_data = []
    for idx in sample_indices:
        s_shap = np.array(shap_value[idx], dtype=float)
        s_shap = np.nan_to_num(s_shap, nan=0.0)
        selected_shap.append(s_shap)
        s_data = X_full[idx]
        if isinstance(s_data, np.ndarray) and np.issubdtype(s_data.dtype, np.number):
            s_data = np.nan_to_num(s_data, nan=0.0)
        selected_data.append(s_data)
    selected_shap = np.array(selected_shap)  # shape: (n_samples, n_features)
    selected_data = np.array(selected_data)

    # 针对第一个样本，根据绝对值大小选出 top_n_features 特征（不包括 Other）
    abs_shap = np.abs(selected_shap[0])
    sorted_idx = np.argsort(abs_shap)[::-1]    # 降序索引：最重要的在前
    top_idx = sorted_idx[:top_n_features]
    # top_idx_sorted_desc 为降序顺序（第0为最重要）
    top_idx_sorted_desc = sorted(top_idx, key=lambda i: abs_shap[i], reverse=True)
    # 为了最终使得决策图显示时“Other”位于底部，我们将 top 特征顺序反转，
    # 这样输入矩阵中，较不重要的 top 特征在前，最重要的 top 特征在最后。
    top_idx_sorted = list(reversed(top_idx_sorted_desc))

    # 取出各样本在 top 特征上的贡献和原始数据（输入顺序为：从低到高 importance）
    shap_values_top = selected_shap[:, top_idx_sorted]
    data_top = selected_data[:, top_idx_sorted]
    feature_names_top = [feature_names[i] for i in top_idx_sorted]

    # 计算每个样本的 "Other" 贡献：所有特征贡献之和减去 top 特征贡献之和
    others = np.array([np.sum(s) - np.sum(s[top_idx_sorted]) for s in selected_shap]).reshape(-1, 1)
    # 将 "Other" 列**预先放到最前面**（即输入矩阵的第0列），这样在决策图内部反转后，它就固定显示在最底部
    shap_values_top = np.hstack([others, shap_values_top])
    data_other = np.zeros((selected_data.shape[0], 1))
    data_top = np.hstack([data_other, data_top])
    # 计算未展示特征数量，并构造标签，例如 "85 other features"
    others_count = len(feature_names) - top_n_features
    feature_names_top = [f"{others_count} other features"] + feature_names_top

    if base_values is None:
        base_value = 0.0
    else:
        base_value = base_values[0] if isinstance(base_values, list) else base_values

    plt.figure(figsize=(12, 6))
    # 这里不传入 feature_order 参数（或传 None），让 decision_plot 保持原顺序
    shap.decision_plot(
        base_value=base_value,
        shap_values=shap_values_top,
        features=data_top,
        feature_names=feature_names_top,
        link=link,
        show=False,
        feature_order=None,
    )
    ax = plt.gca()
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 对第一条线进行加粗处理
    lines = ax.get_lines()
    lineID += 9
    if lines and lineID < len(lines):
        lines[lineID].set_color('black')
        lines[lineID].set_linewidth(3.0)
        lines[lineID].set_linestyle('dashdot')

    # 隐藏图中的文本标签
    for txt in ax.texts:
        txt.set_visible(False)

    # plt.title("Local SHAP (Decision Plot) for samples ", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=700)
    plt.close()
    print(f"[INFO] Local SHAP lines plot saved => {output_path}")
# ---------------------- SHAP Heatmap 函数 ----------------------
# ---------------------- SHAP Heatmap 函数 ----------------------
def plot_shap_heatmap_local(shap_data, output_path, sample_count=100, max_display=12, figsize=(14,8), outputID=0):
    """
    绘制 SHAP 热力图：
      - X 轴为样本，Y 轴为特征，颜色编码为 SHAP 值
      - 默认取前 sample_count 个样本（默认 100）
      - max_display 控制显示的特征数量
      - 针对多输出情况，通过 outputID 选择对应的输出
    输出为 JPG 格式，dpi=700。
    """
    # 如果 shap_values 是列表，则选择对应的输出，否则直接使用
    if isinstance(shap_data["shap_values"], list):
        heatmap_values = shap_data["shap_values"][outputID]
    else:
        heatmap_values = shap_data["shap_values"]

    # 取前 sample_count 个样本
    heatmap_values = heatmap_values[:sample_count]

    # 对特征名称进行简化
    simplified_feature_names = [short_label(f) for f in shap_data["x_col_names"]]

    # 将 numpy 数组转换为 Explanation 对象
    expl = shap.Explanation(values=heatmap_values,
                            feature_names=simplified_feature_names,
                            data=None)

    plt.figure(figsize=figsize, dpi=700)
    shap.plots.heatmap(expl, max_display=max_display, show=False, plot_width=figsize[0])
    plt.savefig(output_path, dpi=700, format='jpg')
    plt.close()
    print(f"[INFO] SHAP heatmap saved => {output_path}")

# --------------------------------------------------------------
def plot_cv_metrics(cv_metrics: dict,
                    save_name: str = "combined_cv_metrics.jpg",
                    show_label: bool = True):
    """
    4‑panel figure: 3 horizontal bar charts + 1 radar chart.

    show_label → 是否在子图左上角显示  a./b./c./d.
    """

    ensure_dir_for_file(save_name)

    # ------------ 数据 ------------
    model_names = list(cv_metrics.keys())
    mse_vals = [cv_metrics[m]["MSE"] for m in model_names]
    mae_vals = [cv_metrics[m]["MAE"] for m in model_names]
    r2_vals  = [cv_metrics[m]["R2"]  for m in model_names]

    # (best, worst, ordinary) colours
    colors_mse = ("#2ca02c", "#d62728", "#1f77b4")
    colors_mae = ("#17becf", "#e377c2", "#bcbd22")
    colors_r2  = ("#ff7f0e", "#9467bd", "#8c564b")

    # ---------- 归一化供雷达 ----------
    metrics = ["MSE", "MAE", "R2"]
    norm_data = {m: [] for m in model_names}
    for metric in metrics:
        col = [cv_metrics[m][metric] for m in model_names]
        mn, mx = min(col), max(col)
        span = mx - mn if mx != mn else 1
        for i, model in enumerate(model_names):
            norm_data[model].append((col[i] - mn) / span)

    # ---------- Figure & Grid ----------
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.08, right=0.96,
                            top=0.93, bottom=0.07,
                            wspace=0.25, hspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = cast(PolarAxes, fig.add_subplot(gs[1, 1], polar=True))

    subplot_font = {"size": 14}

    # ---------- 条形图函数 ----------
    def hbar_with_mean(ax, names, vals, label_char,
                       bigger_is_better, color_triplet):

        vals = np.asarray(vals)
        best = vals.argmax() if bigger_is_better else vals.argmin()
        worst = vals.argmin() if bigger_is_better else vals.argmax()

        bar_colors = [color_triplet[0] if i == best else
                      color_triplet[1] if i == worst else
                      color_triplet[2] for i in range(len(vals))]

        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals[y], color=np.array(bar_colors)[y],
                height=0.4, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(np.array(names)[y], fontsize=12)
        ax.invert_yaxis()

        if show_label:
            ax.text(-0.08, 1.05, f"{label_char}.",
                    transform=ax.transAxes,
                    ha="left", va="top", fontdict=subplot_font)

        # 注释
        right_lim = 1.5 * float(np.max(vals))           # ← 改：固定 0 → 1.5·max
        shift = 0.03 * right_lim
        for xv, yv in zip(vals, y[::-1]):
            ax.text(xv + shift, yv, f"{xv:.2f}",
                    va="center", ha="left", fontsize=10)

        # 平均线与阴影
        m = vals.mean()
        ax.axvline(m, color="gray", ls="--", lw=2)
        ax.axvspan(0, m, color="gray", alpha=0.2)

        ax.set_xlim(0, right_lim)              # ← 统一 0‑基坐标
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        legend_handles = [
            Patch(facecolor=color_triplet[0], label="Best"),
            Patch(facecolor=color_triplet[1], label="Worst"),
            Patch(facecolor=color_triplet[2], label="Ordinary"),
            Patch(facecolor="gray", alpha=0.2, label="Under Mean")
        ]
        ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    # ---------- 绘制三张条形 ----------
    hbar_with_mean(ax_a, model_names, mse_vals, "a", False, colors_mse)
    hbar_with_mean(ax_b, model_names, mae_vals, "b", False, colors_mae)
    hbar_with_mean(ax_c, model_names, r2_vals,  "c", True,  colors_r2)

    # ---------- 雷达 ----------
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    radar_colors = ["#E64B35", "#4DBBD5", "#00A087",
                    "#3C5488", "#F39B7F", "#8491B4"]

    for idx, model in enumerate(model_names):
        vals = np.asarray(norm_data[model] + norm_data[model][:1], dtype=float)
        color = radar_colors[idx % len(radar_colors)]
        ax_d.plot(angles, vals, lw=2, color=color, label=model)
        ax_d.fill(angles, vals, color=color, alpha=0.25)

    ax_d.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=12)
    if show_label:
        ax_d.text(-0.12, 1.1, "d.", transform=ax_d.transAxes,
                  ha="left", va="top", fontdict=subplot_font)

    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                ncol=len(model_names), frameon=False,
                prop={"size": 11})

    # ---------- 保存 ----------
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_cv_metrics_combined] → {save_name}")

# =============================================================
#  K‑Fold 交叉验证箱‑须图
# =============================================================
def plot_cv_boxplot(
    cv_metrics: dict,
    metric: str = "MSE",
    save_name: str = "cv_boxplot_MSE.jpg",
    show_overfit: bool = True
):
    """
    Enhanced box‑plot for 5‑fold CV.

    • metric  : "MSE" | "MAE" | "R2"
    • 若 show_overfit=True，则在右侧 twin‑x 轴同步画 over‑fit 指标
      ─ "MSE_ratio"  = Val_MSE / Train_MSE
      ─ "R2_diff"    = Train_R2 − Val_R2
    """

    ensure_dir_for_file(save_name)
    model_names, data_train, data_val, ovf_vals = [], [], [], []

    # -------- 1. 收集数据 --------
    for m, rec in cv_metrics.items():
        folds = rec.get("folds", {})
        tr_key, va_key = f"{metric}_train", f"{metric}_val"

        if not (tr_key in folds and va_key in folds):
            continue
        tr, va = folds[tr_key], folds[va_key]
        if len(tr) != len(va):
            continue

        model_names.append(m)
        data_train.append(tr)
        data_val.append(va)

        if show_overfit:
            if metric == "MSE":
                ovf = folds.get("MSE_ratio", [])
            elif metric == "R2":
                ovf = folds.get("R2_diff", [])
            else:
                ovf = []
            ovf_vals.append(ovf)

    if not model_names:
        print("[plot_cv_boxplot] – no valid data.")
        return

    # -------- 2. 参数 --------
    n_models   = len(model_names)
    box_width  = 0.25
    group_gap  = 1.0
    color_train, color_val = "#0072B2", "#D55E00"

    # -------- 3. 画主轴 --------
    fig, ax = plt.subplots(figsize=(1.5 * n_models, 6))

    positions_tr = [i * group_gap - box_width / 2 for i in range(n_models)]
    positions_va = [i * group_gap + box_width / 2 for i in range(n_models)]

    bp_tr = ax.boxplot(
        data_train, positions=positions_tr, widths=box_width,
        patch_artist=True, showfliers=False
    )
    bp_va = ax.boxplot(
        data_val,   positions=positions_va, widths=box_width,
        patch_artist=True, showfliers=False
    )

    # 着色
    for p in bp_tr["boxes"]:
        p.set_facecolor(color_train)
        p.set_alpha(0.55)
    for p in bp_va["boxes"]:
        p.set_facecolor(color_val)
        p.set_alpha(0.55)

    plt.setp(bp_tr["medians"], color="black", linewidth=2)
    plt.setp(bp_va["medians"], color="black", linewidth=2)

    # 均值点
    for pos, vals in zip(positions_tr, data_train):
        ax.scatter(pos, np.mean(vals), marker="o", color=color_train, s=65, zorder=3)
    for pos, vals in zip(positions_va, data_val):
        ax.scatter(pos, np.mean(vals), marker="o", color=color_val,   s=65, zorder=3)

    # 散点 jitter
    rng = np.random.default_rng(0)
    for pos_c, dlist, col in [(positions_tr, data_train, color_train),
                              (positions_va, data_val,   color_val)]:
        for p, series in zip(pos_c, dlist):
            jitter = (rng.random(len(series)) - 0.5) * box_width * 0.5
            ax.scatter(p + jitter, series, color=col, alpha=0.3, s=28, zorder=2)

    # x‑tick
    ax.set_xticks([i * group_gap for i in range(n_models)])
    ax.set_xticklabels(model_names, rotation=30, ha="right",
                       fontsize=11)

    # -------- 4. y‑axis（左） --------
    ax.set_ylabel(metric, fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    y_vals_flat = [*sum(data_train, []), *sum(data_val, [])]  # 打平成单列表
    y_min_main = min(y_vals_flat)
    y_max_main = max(y_vals_flat)
    span_main = y_max_main - y_min_main
    ax.set_ylim(y_min_main - 0.10 * span_main,
                y_max_main + 0.10 * span_main)  # ← 上/下各放 10 %

    # Legend
    ax.scatter([], [], color=color_train, label="Train",      s=65)
    ax.scatter([], [], color=color_val,   label="Validation", s=65)
    ax.legend(frameon=False, loc="upper left")

    # -------- 5. 右侧过拟合指标 --------
    if show_overfit and metric in ("MSE", "R2") and ovf_vals:
        ax2 = ax.twinx()
        mean_ovf = [np.mean(v) if v else np.nan for v in ovf_vals]

        ax2.plot([i * group_gap for i in range(n_models)],
                 mean_ovf, marker="^", markersize=8,
                 linewidth=2, color="purple", label="Over-fit")

        # y‑label
        label_text = "MSE ratio (Val/Train)" if metric == "MSE" \
                     else "R$^2$ diff (Train - Val)"
        ax2.set_ylabel(label_text)

        # ➜ 同样只保留 1 位小数
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        y_min_ovf, y_max_ovf = min(mean_ovf), max(mean_ovf)
        span_ovf = y_max_ovf - y_min_ovf
        ax2.set_ylim(y_min_ovf - 0.10 * span_ovf,
                     y_max_ovf + 0.10 * span_ovf)

        ax2.grid(False)
        ax2.legend(frameon=False, loc="upper right")

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_cv_boxplot] → {save_name}")

def plot_overfitting_horizontal(overfit_data,
                                save_name="overfitting_horizontal.jpg"):
    """
    画两张横向 **棒棒糖图**（lollipop）比较过拟合指标：
      • MSE_ratio（Val / Train）—— 越低越好
      • R2_diff  （Train - Val） —— 越低越好
    颜色 / 阴影区和原柱形版本相同，只是柱改为线+圆点。
    """

    # Nature Reviews 常用主色（色弱友好）
    NATURE_RED   = "#D55E00"
    NATURE_BLUE  = "#0072B2"
    NATURE_GREEN = "#009E73"
    GRAY         = "#BEBEBE"
    LIGHT_RED    = "#E69F9F"  # 更深一档，但仍色弱友好

    ensure_dir_for_file(save_name)
    model_names = list(overfit_data.keys())
    msr_vals = [overfit_data[m]["MSE_ratio"] for m in model_names]
    r2d_vals = [overfit_data[m]["R2_diff"] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --------------------------------------------------------------
    def lollipop(ax, names, vals, metric_label,
                 threshold_h, threshold_l):
        vals = np.asarray(vals)
        best, worst = vals.argmin(), vals.argmax()

        # 每个模型的配色
        colors = [NATURE_RED if i == best else
                  NATURE_BLUE if i == worst else
                  NATURE_GREEN for i in range(len(vals))]

        y_pos = np.arange(len(vals))[::-1]  # 反序让最好排最上

        # ------- 线段 + 圆点 -------
        for x, y, c in zip(vals[y_pos], y_pos, np.array(colors)[y_pos]):
            ax.hlines(y, 0, x, color=c, lw=3, alpha=0.9, zorder=1)  # ← 颜色与圆点一致
            ax.scatter(x, y, s=180, color=c, edgecolors="k", zorder=2)

        # 轴与标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(np.array(names)[y_pos])
        ax.invert_yaxis()
        ax.set_xlabel(metric_label, fontsize=12)

        # -------- 数值注释 --------
        # 改动：横坐标上限至少为 1.25 × threshold_h
        value_lim = max(1.25 * threshold_h, 1.8 * float(np.max(vals)))
        shift = 0.03 * value_lim
        for xv, yv in zip(vals, y_pos[::-1]):
            ax.text(xv + shift, yv, f"{xv:.2f}",
                    va="center", ha="left",
                    fontsize=12)

        # -------- 阈值阴影区 --------
        zones: list[Artist]
        if threshold_l == 0:
            ax.axvspan(0, threshold_h, color=GRAY, alpha=0.2, zorder=0)
            zones = [Patch(facecolor=GRAY, alpha=0.2, label="Acceptable")]
        else:
            ax.axvspan(0, threshold_l, color=GRAY, alpha=0.2, zorder=0)
            ax.axvspan(threshold_l, threshold_h,
                       color=LIGHT_RED, alpha=0.3, zorder=0)
            ax.axvline(threshold_l, color=GRAY, ls="--", lw=1.8)
            ax.axvline(threshold_h, color=GRAY, ls="--", lw=1.8)
            zones = [
                Patch(facecolor=GRAY, alpha=0.2, label="Acceptable"),
                Patch(facecolor=LIGHT_RED, alpha=0.3, label="Overfitting Risk")
            ]

        # -------- 坐标范围与刻度 --------
        ax.set_xlim(0, value_lim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        # -------- Legend --------
        zones.extend([
            Line2D([], [], marker='o', color='w',
                   markerfacecolor=NATURE_RED, markersize=10, label="Best"),
            Line2D([], [], marker='o', color='w',
                   markerfacecolor=NATURE_BLUE, markersize=10, label="Worst"),
            Line2D([], [], marker='o', color='w',
                   markerfacecolor=NATURE_GREEN, markersize=10, label="Ordinary")
        ])
        ax.legend(handles=zones, loc="lower right", fontsize=9)

    # 左：MSE Ratio
    lollipop(axes[0], model_names, msr_vals,
             "MSE Ratio (Val / Train)",
             threshold_h=10, threshold_l=5)

    # 右：R² diff
    lollipop(axes[1], model_names, r2d_vals,
             "R$^2$ difference (Train - Val)",
             threshold_h=0.20, threshold_l=0.15)

    plt.tight_layout()
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_overfitting_horizontal] → {save_name}")


class OnlyPositiveNoZeroLocator(ticker.Locator):
    """
    只在 [0, vmax] 范围生成 nbins 个刻度
    若 vmax <= 0 => 不生成任何刻度
    跳过 0 (不放在刻度中)
    """
    def __init__(self, nbins=6):
        self.nbins = nbins
    def __call__(self):
        assert self.axis is not None
        vmin, vmax = self.axis.get_data_interval()
        lower = max(0, min(vmin, vmax))
        upper = max(vmax, lower)
        if upper <= 0:
            return []
        ticks = np.linspace(lower, upper, self.nbins)
        # 过滤掉 0
        filtered = [t for t in ticks if t>1e-9]
        return filtered
    def tick_values(self, vmin, vmax):
        return self.__call__()

class OnlyPositiveIntegerLocator(ticker.Locator):
    """
    只在 [1, floor(vmax)] 范围生成 nbins 个整数刻度, 跳过0/负值
    """
    def __init__(self, nbins=4):
        self.nbins = nbins
    def __call__(self):
        assert self.axis is not None
        vmin, vmax = self.axis.get_data_interval()
        lower = max(5, int(np.ceil(vmin)))
        upper = int(np.floor(vmax))
        if upper<1:
            return []
        ticks = np.linspace(lower, upper, self.nbins)
        ticks = [int(round(t)) for t in ticks]
        ticks = sorted(set(ticks))
        return ticks
    def tick_values(self, vmin, vmax):
        return self.__call__()

class NoSciNoOffsetFormatter(ticker.ScalarFormatter):
    """
    禁用科学计数法 & offset, 强制 '%.2f' 保留两位小数
    """
    def __init__(self, decimals=2, useMathText=False):
        super().__init__(useMathText=useMathText)
        self.decimals = decimals
        self.set_scientific(False)
        self.set_useOffset(False)
    def _set_format(self):
        self.format = f'%.{self.decimals}f'

class TwoSigFigSciFormatter(ticker.ScalarFormatter):
    """两位有效数字 + MathText 科学计数法 (×10^n)"""
    def __init__(self, **kwargs):
        super().__init__(useMathText=True, **kwargs)
        # 始终使用科学记数，并把因子移到轴左上角
        self.set_scientific(True)
        self.set_powerlimits((0, 0))   # 任意范围都启用 offset

    def _set_format(self):
        # Matplotlib 会在 set_locs() 里调用，无参数
        self.format = "%1.2g"          # 2 位有效数字

def plot_multi_model_residual_distribution_single_dim(
    residuals_dict,
    out_label="Output",
    bins=6,
    filename="multi_model_residual_dual_axis.jpg",
    rug_negative_space=0.15,
    show_zero_line_arrow=True
):
    """
    笨方法:
      - 左轴: 只显示正整数(≥1), nbins=5, 跳过0/负值
      - 右轴: 只显示正浮点(≥>0), 保留2位小数, 无科学计数法/offset
      - y<0 用于 rug, 不放负值刻度
      - 其余逻辑(柱状图,KDE,rug,legend等)相同
    """

    if not residuals_dict:
        print("[plot_multi_model_residual_distribution_single_dim] => empty dict, skip.")
        return
    residuals_dict = {m: np.asarray(residuals_dict[m]) for m in residuals_dict}

    # 收集数据
    all_data_list = []
    for m in residuals_dict:
        arr = np.asarray(residuals_dict[m])
        if arr.size==0:
            warnings.warn(f"Model {m} has empty residual array.")
        else:
            all_data_list.append(arr)
    if not all_data_list:
        print("[plot_multi_model_residual_distribution_single_dim] => no valid data, skip.")
        return

    all_data = np.concatenate(all_data_list)
    data_min, data_max = np.min(all_data), np.max(all_data)
    if data_min==data_max:
        data_min -= 1e-6
        data_max += 1e-6

    # 对称区间 => [-R, R]
    R = max(abs(data_min), abs(data_max))
    range_left, range_right = -R, R

    bin_width = (range_right - range_left)/bins
    base_edges = np.linspace(range_left, range_right, bins+1)
    edges = base_edges + bin_width/2

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.rcParams["font.size"] = 16 ##问题
    fig, ax = plt.subplots(figsize=(5,4)) #修改画幅
    ax2 = ax.twinx()

    model_names = list(residuals_dict.keys())
    n_models = len(model_names)
    color_palette = ["#24345C", "#279DE1", "#329845", "#8A233F", "#912C2C"]
    if n_models > len(color_palette):
        color_palette = color_palette * (n_models // len(color_palette) + 1)

    # ========== 左轴: Histogram(Count) ==========
    hist_counts_dict = {}
    max_count = 0
    for m in model_names:
        arr = np.asarray(residuals_dict[m])
        counts, _ = np.histogram(arr, bins=edges)
        hist_counts_dict[m] = counts
        cmax = int(np.max(counts))
        if cmax>max_count:
            max_count=cmax

    group_width = bin_width*0.9
    bar_width = group_width/n_models
    gap = bin_width-group_width

    for b_idx in range(len(edges)-1):
        x_left_bin = edges[b_idx]
        for i,m in enumerate(model_names):
            c = hist_counts_dict[m][b_idx]
            color_ = color_palette[i]
            rect_left = x_left_bin + 0.5*gap + i*bar_width
            ax.bar(
                rect_left, c,
                width=bar_width, bottom=0,
                color=color_, alpha=0.6,
                align="edge", edgecolor='none'
            )

    ax.set_xlim(range_left, range_right)
    ax.set_ylim(-rug_negative_space*max_count, max_count*1.1)
    ax.set_xlabel("Residual", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)

    # # 左轴 => 只显示正整数 => OnlyPositiveIntegerLocator
    # ax.yaxis.set_major_locator(OnlyPositiveIntegerLocator(nbins=5))
    # # 不显示 offset
    # ax.yaxis.get_offset_text().set_visible(False)
    # ① 只放正刻度（最多 5 个），略去 0
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

    # ② 使用两位有效数字 + 科学计数法
    ax.yaxis.set_major_formatter(TwoSigFigSciFormatter())

    # ③ 可选：给 offset 文本调个字号，与整体一致
    ax.yaxis.get_offset_text().set_fontsize(14)

    # ========== 右轴: KDE + rug ==========
    max_density = 0
    for i,m in enumerate(model_names):
        arr = np.asarray(residuals_dict[m])
        color_ = color_palette[i]
        # KDE
        kde_obj = sns.kdeplot(
            arr, ax=ax2, color=color_,
            fill=False, alpha=0.9,
            linewidth=2, bw_adjust=0.8
        )
        if kde_obj.lines:
            ydata = np.asarray(kde_obj.lines[-1].get_ydata())
            if ydata.size > 0:
                cur_max = float(np.max(ydata))
                if cur_max>max_density:
                    max_density=cur_max

        # rug
        sns.rugplot(
            arr, ax=ax2,
            height=0.1, color=color_,
            alpha=0.4, lw=1, clip_on=False
        )

    ax2.set_ylim(-rug_negative_space*max_density, max_density*1.2)
    ax2.set_ylabel("Density", fontsize=12)

    # 在右轴 y=0 画线
    ax2.axhline(0, color='k', linewidth=1.5, zorder=2)
    ax2.spines["bottom"].set_visible(False)
    ax2.set_axisbelow(False)

    # 右轴 => 只显示正浮点 => 5个刻度
    ax2.yaxis.set_major_locator(OnlyPositiveNoZeroLocator(nbins=6))
    # 两位小数 => 自定义NoSciNoOffsetFormatter
    no_sci_fmt = NoSciNoOffsetFormatter(decimals=2, useMathText=False)
    ax2.yaxis.set_major_formatter(no_sci_fmt)

    # x=0 竖线
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    if show_zero_line_arrow and (range_left<0<range_right):
        arrow_y = max_density*0.6
        ax2.annotate(
            "Zero line",
            xy=(0, arrow_y),
            xytext=(0, max_density*1.05),
            ha="center",
            arrowprops=dict(arrowstyle="->", color='k')
        )

    # 图例(直方图风格)
    legend_patches = []
    for i,m in enumerate(model_names):
        patch = mpatches.Patch(
            facecolor=color_palette[i],
            edgecolor='none',
            alpha=0.6,
            label=m
        )
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10)

    # ax.set_title(out_label, fontsize=13)

    # 美化边框
    for spine in ax.spines.values():
        spine.set_visible(True)
    for spine in ax2.spines.values():
        spine.set_visible(True)

    ax.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_multi_model_residual_distribution_single_dim] => {filename}")


def plot_optuna_tuning_curve(trials_df, out_path):
    """
    使用 matplotlib 绘制优化历史曲线，
    图形为黑白风格，仅调整字体大小和加粗。
    要求 trials_df 至少包含 "value" 列，
    若存在 "number" 列，则用作迭代编号；否则使用索引。
    """

    # 提取 x 轴数据（迭代编号）和 y 轴数据（目标值）
    if "number" in trials_df.columns:
        x = trials_df["number"]
    else:
        x = trials_df.index
    if "value" in trials_df.columns:
        y = trials_df["value"]
    else:
        raise ValueError("trials_df 必须包含 'value' 列")

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linewidth=3, markersize=8, color="#4DBBD5")
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Objective Value", fontsize=14)
    # plt.title("Optimization History", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=700)
    plt.close()
    print(f"[INFO] Custom styled Optuna Optimization History saved => {out_path}")


def plot_optuna_summary_curve(trials_dict, out_path):
    """
    绘制汇总图：对传入 trials_dict 中的各模型数据，
    绘制调参历史曲线，并用五角星标出各模型的最佳结果（目标值最小）。

    参数：
      trials_dict: dict，键为模型名，值为对应的 trials DataFrame
      out_path: 保存生成图形的文件路径（如 "evaluation/figures/数据名/optuna/summary.jpg"）
    """
    colors = [
        "#1f77b4",  # 蓝
        "#2ca02c",  # 绿
        "#d62728",  # 红
        "#9467bd",  # 紫
        "#8c564b",  # 棕
        "#17becf"   # 青色
    ]
    plt.figure(figsize=(8, 6))

    for i, (mtype, trials_df) in enumerate(trials_dict.items()):
        # 根据是否存在 "number" 列选择迭代编号，否则使用索引
        if "number" in trials_df.columns:
            x = trials_df["number"]
        else:
            x = trials_df.index
        if "value" not in trials_df.columns:
            raise ValueError("trials_df 必须包含 'value' 列")
        y = trials_df["value"]
        color = colors[i % len(colors)]

        # 绘制调参历史曲线（加上透明度 alpha=0.7）
        plt.plot(x, y, marker='o', linewidth=3, markersize=5,
                 color=color, alpha=0.15, label=f"{mtype} History")

        # 找到目标最优（最小）点，并用不透明的五角星标记
        best_idx = y.idxmin()
        if "number" in trials_df.columns:
            best_x = trials_df.loc[best_idx, "number"]
        else:
            best_x = best_idx
        best_y = float(np.min(y))
        plt.plot(best_x, best_y, marker='*', markersize=14,
                 color=color, linestyle='None', label=f"{mtype} Best")

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Objective Value", fontsize=14)
    # plt.title("Optimization Summary for All Models", fontsize=18)

    # 将图例放置在主图上方外部，按一行排列所有图例项
    plt.legend(bbox_to_anchor=(0.5, 1.12), loc='upper center',
               ncol=len(trials_dict), fontsize=8, frameon=False)

    # 调整布局，留出图例所需空间
    plt.tight_layout(rect=(0, 0, 1, 0.13))
    plt.savefig(out_path, dpi=700)
    plt.close()
    print(f"[INFO] Custom styled Optuna Summary Curve saved => {out_path}")


def plot_optuna_slice(trials_df, params, out_path):
    """
    使用 matplotlib 和 seaborn 绘制参数切片图，
    整体风格为黑白风格，仅调整字体大小和加粗。
    trials_df 中必须包含 "value" 列以及 params 列表中指定的参数。
    """
    import seaborn as sns

    n_params = len(params)
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 4 * n_params), sharey=True)
    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        if param not in trials_df.columns:
            ax.text(0.5, 0.5, f"Column '{param}' not found", ha='center', va='center', fontsize=14)
            continue
        x = trials_df[param]
        y = trials_df["value"]
        # 使用 seaborn 绘制散点图（默认黑色）
        sns.scatterplot(x=x, y=y, ax=ax, color="black", s=50, edgecolor="black", alpha=0.7)
        # 将横坐标标签改为实际读取的参数名称
        ax.set_xlabel(param, fontsize=14, color="black")
        ax.set_ylabel("Objective Value", fontsize=14, color="black")
        ax.tick_params(labelsize=12, colors="black")
    # fig.suptitle("Parameter Slice Plot", fontsize=18, color="black")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=700)
    plt.close()
    print(f"[INFO] Custom styled Optuna Slice Plot saved => {out_path}")


def plot_optuna_param_importances(trials_df, out_path):
    """
    使用 matplotlib 绘制参数重要性水平条形图，
    参数重要性通过各参数列与 "value" 的相关系数（取绝对值）计算；
    图形整体采用蓝色配色（条形全为蓝色），字体加大加粗，其余部分不修改。
    注意：这里假设传入的 trials_df 已经重命名过，即参数列已去掉 "params_" 前缀，
    因此不再依赖于 startswith("params_") 过滤，而是排除一些内置字段。
    """

    # 定义需要排除的内置字段
    exclude = {'number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state'}

    # 选择参数列：所有不在 exclude 集合中的列
    param_cols = [col for col in trials_df.columns if col not in exclude]
    if not param_cols:
        print("[WARN] No parameter columns found!")
        return

    objective = trials_df["value"].to_numpy(dtype=float)
    importances = {}
    for col in param_cols:
        data = trials_df[col]
        try:
            # 先尝试将数据转换为 float 类型
            data_numeric = data.to_numpy(dtype=float)
        except Exception:
            data_numeric = None

        corr = 0.0
        try:
            if data_numeric is not None:
                valid_mask = ~np.isnan(data_numeric) & ~np.isnan(objective)
                if valid_mask.sum() < 2:
                    corr = 0.0
                else:
                    corr = np.corrcoef(data_numeric[valid_mask], objective[valid_mask])[0, 1]
            else:
                raise ValueError
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            try:
                data_cat = data.astype("category").cat.codes.to_numpy(dtype=float)
                valid_mask = ~np.isnan(data_cat) & ~np.isnan(objective)
                if valid_mask.sum() < 2:
                    corr = 0.0
                else:
                    corr = np.corrcoef(data_cat[valid_mask], objective[valid_mask])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0
        importances[col] = abs(corr)

    # 排序：降序排列
    param_names = list(importances.keys())
    imp_values = [importances[name] for name in param_names]
    sorted_idx = np.argsort(imp_values)[::-1]
    param_names_sorted = [param_names[i] for i in sorted_idx]
    imp_values_sorted = [imp_values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(param_names_sorted))
    # 固定所有条形使用蓝色
    ax.barh(y_pos, imp_values_sorted, color="blue", edgecolor="black", height=0.6)
    # 显示时直接使用原始列名称（已去掉前缀）
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names_sorted, fontsize=12, color="black")
    ax.set_xlabel("Importance Score", fontsize=14, color="black")
    # ax.set_title("Parameter Importances", fontsize=18, color="black")
    ax.invert_yaxis()  # 最大重要性在上
    plt.tight_layout()
    plt.savefig(out_path, dpi=700)
    plt.close()
    print(f"[INFO] Custom styled Optuna Parameter Importances Plot saved => {out_path}")

# inference
def _upsample_grid(grid_x, grid_y, Z, smooth=4, order=3):
    """
    将规则网格 (grid_x, grid_y, Z) 用 scipy.ndimage.zoom 做双三次插值，
    让图面更平滑。smooth=4 表示行、列各细分 4 倍。
    """
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)
    Z = np.asarray(Z)
    if smooth <= 1:
        return grid_x, grid_y, Z                # 不插值

    # 原网格维度
    H, W = Z.shape
    zoom_factor = (smooth, smooth)              # (y, x) 方向

    # Z 插值
    Z_fine = zoom(Z, zoom_factor, order=order)

    # 对应生成新的等间距网格坐标
    x_min, x_max = float(np.min(grid_x)), float(np.max(grid_x))
    y_min, y_max = float(np.min(grid_y)), float(np.max(grid_y))
    x_vals = np.linspace(x_min, x_max, W * smooth)
    y_vals = np.linspace(y_min, y_max, H * smooth)
    grid_x_fine, grid_y_fine = np.meshgrid(x_vals, y_vals)

    return grid_x_fine, grid_y_fine, Z_fine


# ===============================================================
# 1) 2-D Heatmap  (平滑版)
# ===============================================================
def plot_2d_heatmap_from_npy(grid_x, grid_y, heatmap_pred,
                             out_dir,
                             x_label="X-axis",
                             y_label="Y-axis",
                             y_col_names=None,
                             stats_dict=None,
                             colorbar_extend_ratio=0.25,
                             smooth=4):          # ← 新增参数，默认 ×4
    """
    smooth:   >=2 时会用双三次插值把网格细分，视觉更顺滑；
              设 1 则保持原始分辨率。
    """
    os.makedirs(out_dir, exist_ok=True)
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)
    heatmap_pred = np.asarray(heatmap_pred)
    _, _, out_dim = heatmap_pred.shape

    for odx in range(out_dim):
        # ---------- ① 插值 ----------
        gx_f, gy_f, z_f = _upsample_grid(grid_x,
                                         grid_y,
                                         heatmap_pred[:, :, odx],
                                         smooth=smooth,
                                         order=3)   # 双三次
        gx_f = np.asarray(gx_f)
        gy_f = np.asarray(gy_f)
        z_f = np.asarray(z_f)

        auto_min, auto_max = float(np.min(z_f)), float(np.max(z_f))
        if stats_dict and y_col_names and odx < len(y_col_names) \
           and y_col_names[odx] in stats_dict:
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
            vmin_ = max(0, real_min * (1 - colorbar_extend_ratio))
            vmax_ = real_max * (1 + colorbar_extend_ratio)
        else:
            vmin_, vmax_ = auto_min, auto_max

        norm_ = mcolors.Normalize(vmin=vmin_, vmax=vmax_)

        plt.figure(figsize=(6, 5))
        # shading='gouraud' + finer grid → 颜色平滑过渡
        cm_ = plt.pcolormesh(gx_f, gy_f, z_f,
                             shading="gouraud",
                             cmap="GnBu",
                             norm=norm_)
        cb_ = plt.colorbar(cm_)
        cb_.set_label(y_col_names[odx] if y_col_names and odx < len(y_col_names)
                      else f"Output_{odx}",
                      fontsize=12)
        cb_.locator = MaxNLocator(nbins=5, integer=True)
        cb_.update_ticks()

        ax = plt.gca()
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

        out_jpg = os.path.join(out_dir, f"heatmap_output_{odx + 1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches="tight")
        plt.close()
        print(f"[INFO] 2D Heatmap saved → {out_jpg}")


# ===============================================================
# 2) 3-D Surface  (平滑版)
# ===============================================================
def plot_3d_surface_from_heatmap(grid_x, grid_y, heatmap_pred,
                                 out_dir,
                                 x_label="X-axis",
                                 y_label="Y-axis",
                                 y_col_names=None,
                                 stats_dict=None,
                                 colorbar_extend_ratio=0.25,
                                 cmap_name="GnBu",
                                 smooth=4):          # ← 新增
    os.makedirs(out_dir, exist_ok=True)
    heatmap_pred = np.asarray(heatmap_pred)
    _, _, out_dim = heatmap_pred.shape

    for odx in range(out_dim):
        # ---------- ① 插值 ----------
        gx_f, gy_f, Z_f = _upsample_grid(grid_x,
                                         grid_y,
                                         heatmap_pred[:, :, odx],
                                         smooth=smooth,
                                         order=3)
        gx_f = np.asarray(gx_f)
        gy_f = np.asarray(gy_f)
        Z_f = np.asarray(Z_f)

        auto_min, auto_max = float(np.min(Z_f)), float(np.max(Z_f))
        if stats_dict and y_col_names and odx < len(y_col_names) \
           and y_col_names[odx] in stats_dict:
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
            vmin_ = max(0, real_min * (1 - colorbar_extend_ratio))
            vmax_ = real_max * (1 + colorbar_extend_ratio)
        else:
            vmin_, vmax_ = auto_min, auto_max

        norm_ = mcolors.Normalize(vmin=vmin_, vmax=vmax_)
        cmap_ = plt.get_cmap(cmap_name)
        colors_rgba = np.asarray(cmap_(norm_(np.ravel(Z_f)))).reshape(
            (Z_f.shape[0], Z_f.shape[1], 4)
        )

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(gx_f, gy_f, Z_f,
                        facecolors=colors_rgba,
                        rstride=1, cstride=1,
                        linewidth=0,
                        antialiased=True,   # ← 抗锯齿
                        shade=False)

        sm = cm.ScalarMappable(norm=norm_, cmap=cmap_)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, aspect=15)
        cb.set_label(y_col_names[odx] if y_col_names and odx < len(y_col_names)
                     else f"Output_{odx}",
                     fontsize=12)
        cb.locator = MaxNLocator(nbins=5, integer=True)
        cb.update_ticks()

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_zlabel("Value",  fontsize=12)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.grid(False)

        out_jpg = os.path.join(out_dir, f"heatmap_3d_surface_output_{odx + 1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches="tight")
        plt.close()
        print(f"[INFO] 3D Surface saved → {out_jpg}")


# ────────────────────────── 辅助函数 ────────────────────────── #
def _prep_labels(labels):            # 标签裁剪
    return [short_label(l) for l in labels]

def _draw_grid(ax, n_rows, n_cols, cell):
    for rr in range(n_rows + 1):
        ax.axhline(rr * cell, color='black', linewidth=1)
    for cc in range(n_cols + 1):
        ax.axvline(cc * cell, color='black', linewidth=1)

def _set_axes(ax, n_rows, n_cols, cell, row_lbls, col_lbls,
              row_axis_name, col_axis_name):
    ax.set_xlim(0, n_cols * cell)
    ax.set_ylim(0, n_rows * cell)
    ax.invert_yaxis()

    ax.set_xticks([(j + 0.5) * cell for j in range(n_cols)])
    ax.set_yticks([(i + 0.5) * cell for i in range(n_rows)])
    ax.set_xticklabels(col_lbls, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(row_lbls, fontsize=9)
    ax.set_xlabel(col_axis_name, fontsize=14)
    ax.set_ylabel(row_axis_name, fontsize=14)

def _save(fig, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=700, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Confusion saved ⇒ {path}")

# ───────────── 绘制顶端水平 ColorBar ───────────── #
def _draw_top_colorbars(fig, cmaps, norms, dim_used, y_col_names,
                        left0=0.03, width=0.21, height=0.02, bottom=0.93):
    """
    在 fig 顶端画一排水平色条。

    - 当 dim_used == 1 时：色条水平居中，宽度 = width
    - 当 dim_used >  1 时：沿用固定间距 (left0 + k*width)
    """
    for k in range(dim_used):
        sm = cm.ScalarMappable(norm=norms[k], cmap=cmaps[k])
        sm.set_array([])

        if dim_used == 1:
            # 让唯一的色条居中
            left = 0.5 - width / 2          # canvas x 轴 0‒1 范围
        else:
            left = left0 + k * width        # 原排布

        cax = fig.add_axes((left, bottom, width, height))
        cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')

        label = (y_col_names[k] if (y_col_names and k < len(y_col_names))
                 else f"Out {k}")
        cb.set_label(label, fontsize=12, labelpad=2)

        cb.set_ticks([])                    # 只留文字
        cb.ax.xaxis.set_label_position('bottom')
        cb.ax.xaxis.set_ticks_position('top')

# ──────────────────────────────────────────────── #

def plot_confusion_from_npy(confusion_pred,
                            row_labels, col_labels,
                            out_dir,
                            y_col_names=None,
                            stats_dict=None,
                            cell_scale=1/5,
                            row_axis_name="Row Axis",
                            col_axis_name="Col Axis"):
    """
    支持 out_dim == 1 (整块) 及 2–4 维 (四三角) 的混淆矩阵可视化。
    """
    confusion_pred = np.asarray(confusion_pred)
    n_rows, n_cols, out_dim = confusion_pred.shape
    # 若行标签数量 > 列标签数量，则把行/列对调，
    # 让“元素更多”的那一边作为横轴。
    if len(row_labels) > len(col_labels):
        confusion_pred = confusion_pred.transpose(1, 0, 2)  # (rows, cols, out) >> (cols, rows, out)
        row_labels, col_labels = col_labels, row_labels
        n_rows, n_cols = n_cols, n_rows
        # 如需连同轴标题一起交换，取消下一行注释
        row_axis_name, col_axis_name = col_axis_name, row_axis_name

    row_labels = _prep_labels(row_labels)
    col_labels = _prep_labels(col_labels)
    if y_col_names:
        y_col_names = _prep_labels(y_col_names)

    # ───────────── 单输出：整块填色 ───────────── #
    if out_dim == 1:
        vmin, vmax = float(np.min(confusion_pred)), float(np.max(confusion_pred))
        cmap, norm = plt.get_cmap("Purples"), mcolors.Normalize(vmin, vmax)

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_aspect("equal", "box")
        _draw_grid(ax, n_rows, n_cols, cell_scale)

        for i in range(n_rows):
            for j in range(n_cols):
                val = confusion_pred[i, j, 0]
                ax.add_patch(Rectangle((j * cell_scale, i * cell_scale),
                                       cell_scale, cell_scale,
                                       facecolor=cmap(norm(val)),
                                       edgecolor="black"))

        _set_axes(ax, n_rows, n_cols, cell_scale, row_labels, col_labels,
                  row_axis_name, col_axis_name)

        # ---- 顶端水平色条 ----
        _draw_top_colorbars(fig,
                            cmaps=[cmap],                    # 只有一种 colormap
                            norms=[norm],
                            width=0.63,
                            height=0.03,
                            dim_used=1,
                            y_col_names=y_col_names)

        _save(fig, out_dir, "confusion_matrix_1d.jpg")
        return

    # ───────────── 多输出：四三角 ───────────── #
    dim_used = min(4, out_dim)
    cmaps = [plt.get_cmap(c) for c in ["Purples", "Blues", "Greens", "Oranges"]]

    # 归一化到 [0,1]（支持用 stats_dict 限制范围）
    norms = []
    for k in range(dim_used):
        vals = confusion_pred[:, :, k]
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if (stats_dict and y_col_names and k < len(y_col_names)
                and y_col_names[k] in stats_dict):
            vmin = stats_dict[y_col_names[k]]["min"]
            vmax = stats_dict[y_col_names[k]]["max"]
        confusion_pred[:, :, k] = normalize_data(vals, vmin, vmax)
        norms.append(mcolors.Normalize(0, 1))

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_aspect("equal", "box")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.1)
    _draw_grid(ax, n_rows, n_cols, cell_scale)

    # 三角绘制模板
    tri_idx = [
        [(0, 1), (0.5, 0.5), (1, 1)],      # 左上
        [(1, 1), (0.5, 0.5), (1, 0)],      # 右上
        [(1, 0), (0.5, 0.5), (0, 0)],      # 右下
        [(0, 0), (0.5, 0.5), (0, 1)],      # 左下
    ]

    for i in range(n_rows):
        for j in range(n_cols):
            cx, cy = j * cell_scale, i * cell_scale
            for k in range(dim_used):
                poly = [(cx + dx*cell_scale, cy + dy*cell_scale)
                        for dx, dy in tri_idx[k]]
                ax.add_patch(
                    Polygon(poly,
                            facecolor=cmaps[k](norms[k](confusion_pred[i, j, k])),
                            alpha=0.9))

    _set_axes(ax, n_rows, n_cols, cell_scale, row_labels, col_labels,
              row_axis_name, col_axis_name)

    # 顶部水平 colorbar
    _draw_top_colorbars(fig, cmaps, norms, dim_used, y_col_names)

    _save(fig, out_dir, "confusion_matrix_mimo.jpg")


def plot_3d_bars_from_confusion(confusion_pred,
                                row_labels, col_labels,
                                out_dir,
                                y_col_names=None,
                                stats_dict=None,
                                colorbar_extend_ratio=0.02,
                                cmap_name="GnBu"):
    """
    绘制三维柱状图(Bar3D)的 “confusion-like” 图。
    - 若 stats_dict 存在且含有对应维度的统计范围，则用其 min/max；
      否则用该维度数据的最小值、最大值。
    - 将 x/y 刻度对准柱体中心，并使得刻度标签居中。
    - 每个维度单独输出一个 3D 柱状图。
    """

    os.makedirs(out_dir, exist_ok=True)
    confusion_pred = np.asarray(confusion_pred)
    n_rows, n_cols, out_dim = confusion_pred.shape

    # 标签处理
    row_labels = [short_label(lbl) for lbl in row_labels]
    col_labels = [short_label(lbl) for lbl in col_labels]
    if y_col_names:
        y_col_names = [short_label(name) for name in y_col_names]

    # 我们在此不限制维度个数，每个维度绘制一个图
    for odx in range(out_dim):
        all_vals_dim = confusion_pred[:, :, odx]
        auto_min, auto_max = float(np.min(all_vals_dim)), float(np.max(all_vals_dim))

        if (stats_dict is not None) and (y_col_names is not None) \
           and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
        else:
            real_min = auto_min
            real_max = auto_max

        # 归一化该维度数据到 [0,1]
        Z = normalize_data(all_vals_dim, real_min, real_max)

        norm_ = mcolors.Normalize(vmin=0, vmax=1)
        cmap_ = plt.get_cmap(cmap_name)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        dx = dy = 0.5

        x_vals, y_vals, z_vals = [], [], []
        dz_vals, facecolors = [], []

        for i in range(n_rows):
            for j in range(n_cols):
                val_ = Z[i, j]
                x_vals.append(j)
                y_vals.append(i)
                z_vals.append(0)
                dz_vals.append(val_)
                # 根据归一化后的值生成颜色
                facecolors.append(cmap_(norm_(val_)))

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        z_vals = np.array(z_vals)
        dz_vals = np.array(dz_vals)

        ax.bar3d(
            x_vals, y_vals, z_vals,
            dx, dy, dz_vals,
            color=facecolors, alpha=0.75, shade=True
        )

        ax.grid(False)

        # 让刻度居中对齐柱体
        ax.set_xticks(np.arange(n_cols) + dx / 2)
        ax.set_yticks(np.arange(n_rows) + dy / 2)

        # X 轴标签：旋转 45 度并右对齐
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
        # Y 轴标签：根据需求选择合适的旋转
        ax.set_yticklabels(row_labels, rotation=-15, ha='left', va='center', fontsize=10)

        # 仅保留 Z 轴名称
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("Value", fontsize=12)

        # 颜色条
        sm = cm.ScalarMappable(norm=norm_, cmap=cmap_)
        sm.set_array([])  # 不对应具体数组，只作颜色映射
        cb_ = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, aspect=15)

        # 标题
        if y_col_names and odx < len(y_col_names):
            var_name = y_col_names[odx]
            # ax.set_title(f"3D Bars Confusion - {var_name}", fontsize=14)
            cb_.set_label(var_name, fontsize=12)
        else:
            var_name = f"Output_{odx}"
            # ax.set_title(f"3D Bars Confusion - out {odx}", fontsize=14)
            cb_.set_label(var_name, fontsize=12)

        out_jpg = os.path.join(out_dir, f"3d_bars_confusion_output_{odx+1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 3D Bars Confusion saved => {out_jpg}")


def plot_3d_surface_from_3d_heatmap(
        grid_x, grid_y, grid_z, heatmap_pred,
        out_dir,
        axes_labels=("X", "Y", "Z"),
        y_col_names=None,
        out_idx=0,
        cmap_name="GnBu",
        alpha_mode="value",        # "value" or "inverse"
        alpha_gamma=1.8            # ★ 新增：α 线性偏置 (γ =1 → 纯线性)
):
    """
    透明 3-D slice 曲面（每个 Z-slice 画一片）

    Parameters
    ----------
    alpha_mode  : "value"  -> 值越大 α 越高
                  "inverse"-> 值越大 α 越低
    alpha_gamma : 线性 γ 偏置；>1 更突出高值；<1 整体更实
    """
    os.makedirs(out_dir, exist_ok=True)
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)
    grid_z = np.asarray(grid_z)
    heatmap_pred = np.asarray(heatmap_pred)

    Zval = heatmap_pred[..., out_idx]           # (H,W,D)
    vmin_, vmax_ = float(np.min(Zval)), float(np.max(Zval))
    norm_  = mcolors.Normalize(vmin_, vmax_)
    cmap_  = plt.get_cmap(cmap_name)

    rgba = np.asarray(cmap_(norm_(Zval)))       # (H,W,D,4)
    alpha_base = norm_(Zval) ** alpha_gamma     # 0-1 after γ
    if alpha_mode == "value":                   # 大值 → 不透明
        rgba[..., -1] = alpha_base
    elif alpha_mode == "inverse":               # 大值 → 透明
        rgba[..., -1] = 1.0 - alpha_base
    else:
        raise ValueError("alpha_mode must be 'value' or 'inverse'")

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    nz = grid_z.shape[2]
    for k in range(nz):
        ax.plot_surface(grid_x[:, :, k], grid_y[:, :, k], grid_z[:, :, k],
                        facecolors=rgba[:, :, k, :],
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)

    # ——— color-bar ———
    sm = cm.ScalarMappable(norm=norm_, cmap=cmap_)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, aspect=15)
    cb.set_label(
        y_col_names[out_idx] if (y_col_names and out_idx < len(y_col_names))
        else f"Output_{out_idx}", fontsize=12)

    # ——— 轴设置 ———
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(False)

    out_jpg = os.path.join(out_dir, f"surface3d_output_{out_idx+1}.jpg")
    plt.savefig(out_jpg, dpi=700, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 3D Color Surface saved → {out_jpg}")
