#!/usr/bin/env python3
"""
Plot KDE distributions for all numeric-like columns in a CSV file.

Usage:
  python data/data_KDE.py
  python data/data_KDE.py --csv data/Ea_20260226.csv --out data/Ea_20260226_kde_panels.png
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None


NATURE_PALETTE = [
    "#0072B2",  # blue
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#CC79A7",  # magenta
    "#E69F00",  # orange
]

NULL_LIKE = {"", " ", "na", "nan", "none", "null", "__missing__"}


def load_csv_with_fallback(csv_path: Path) -> tuple[pd.DataFrame, str]:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "gb18030"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            return df, enc
        except Exception as e:  # pragma: no cover
            last_error = e
    raise RuntimeError(f"Failed to read CSV '{csv_path}' with known encodings: {last_error}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    out = out.loc[:, out.columns != ""]
    out = out.loc[:, ~out.columns.str.lower().str.startswith("unnamed:")]
    return out


def to_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.strip()
    s = s.mask(s.str.lower().isin(NULL_LIKE))
    return pd.to_numeric(s, errors="coerce")


def detect_numeric_like_columns(
    df: pd.DataFrame,
    min_numeric_ratio: float,
    min_non_na: int,
) -> tuple[list[tuple[str, pd.Series]], dict[str, str]]:
    numeric_cols: list[tuple[str, pd.Series]] = []
    skipped: dict[str, str] = {}

    n_rows = max(1, len(df))
    for col in df.columns:
        num = to_numeric_series(df[col])
        valid_count = int(num.notna().sum())
        valid_ratio = valid_count / n_rows
        uniq_count = int(num.nunique(dropna=True))

        if valid_count < min_non_na:
            skipped[col] = f"valid values={valid_count} < {min_non_na}"
            continue
        if min_numeric_ratio > 0 and valid_ratio < min_numeric_ratio:
            skipped[col] = f"numeric ratio={valid_ratio:.2f} < {min_numeric_ratio:.2f}"
            continue
        if uniq_count < 2:
            skipped[col] = "unique numeric values < 2"
            continue

        numeric_cols.append((col, num.dropna().astype(float)))

    return numeric_cols, skipped


def apply_nature_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1A1A1A",
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "TeX Gyre Heros",
                "Nimbus Sans",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_panel_title(name: str, width: int = 28) -> str:
    return textwrap.fill(str(name).strip(), width=width)


def plot_single_kde(ax: plt.Axes, values: np.ndarray, title: str, color: str) -> None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 2 or np.isclose(np.std(x), 0.0):
        bins = min(10, max(3, x.size))
        ax.hist(x, bins=bins, density=True, color=color, alpha=0.35, edgecolor="none")
    else:
        use_hist_fallback = True
        if gaussian_kde is not None and np.unique(x).size >= 3:
            xmin, xmax = float(np.min(x)), float(np.max(x))
            span = xmax - xmin
            pad = 0.05 * span if span > 0 else 0.5
            grid = np.linspace(xmin - pad, xmax + pad, 320)
            try:
                kde = gaussian_kde(x)
                y = kde(grid)
                ax.fill_between(grid, 0, y, color=color, alpha=0.25, linewidth=0)
                ax.plot(grid, y, color=color, linewidth=1.8)
                use_hist_fallback = False
            except Exception:
                use_hist_fallback = True

        if use_hist_fallback:
            bins = min(24, max(8, int(np.sqrt(x.size))))
            ax.hist(x, bins=bins, density=True, color=color, alpha=0.35, edgecolor="none")

    ax.set_title(format_panel_title(title), pad=6)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_kde_panel_figure(
    numeric_cols: list[tuple[str, pd.Series]],
    out_path: Path,
    dpi: int,
    ncols: int,
) -> None:
    if not numeric_cols:
        raise ValueError("No numeric-like columns detected for KDE plotting.")

    ncols = max(1, int(ncols))
    n_panels = len(numeric_cols)
    nrows = math.ceil(n_panels / ncols)

    fig_w = 3.4 * ncols
    fig_h = 2.9 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    flat_axes = axes.ravel()

    for i, (col_name, series) in enumerate(numeric_cols):
        color = NATURE_PALETTE[i % len(NATURE_PALETTE)]
        plot_single_kde(flat_axes[i], series.to_numpy(), col_name, color)

    for j in range(n_panels, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.06,
        top=0.97,
        wspace=0.30,
        hspace=0.58,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-panel KDE distributions from CSV columns.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/Ea_20260226.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/Ea_20260226_kde_panels.png"),
        help="Output figure path (PNG).",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI.")
    parser.add_argument("--ncols", type=int, default=3, help="Panels per row.")
    parser.add_argument(
        "--min-numeric-ratio",
        type=float,
        default=0.0,
        help="Minimum fraction of numeric-parsable entries for keeping a column.",
    )
    parser.add_argument(
        "--min-non-na",
        type=int,
        default=8,
        help="Minimum count of valid numeric values for keeping a column.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also save a PDF with the same basename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_nature_style()

    df_raw, used_encoding = load_csv_with_fallback(args.csv)
    df = clean_dataframe(df_raw)
    numeric_cols, skipped = detect_numeric_like_columns(
        df,
        min_numeric_ratio=float(args.min_numeric_ratio),
        min_non_na=int(args.min_non_na),
    )

    print(f"[INFO] CSV loaded: {args.csv} (encoding={used_encoding})")
    print(f"[INFO] Columns total={len(df.columns)}, numeric-like={len(numeric_cols)}, skipped={len(skipped)}")
    if skipped:
        for col, reason in skipped.items():
            print(f"[SKIP] {col}: {reason}")

    build_kde_panel_figure(
        numeric_cols=numeric_cols,
        out_path=args.out,
        dpi=int(args.dpi),
        ncols=int(args.ncols),
    )
    print(f"[INFO] KDE panel figure saved: {args.out}")

    if args.save_pdf:
        pdf_path = args.out.with_suffix(".pdf")
        build_kde_panel_figure(
            numeric_cols=numeric_cols,
            out_path=pdf_path,
            dpi=max(300, int(args.dpi)),
            ncols=int(args.ncols),
        )
        print(f"[INFO] KDE panel figure saved: {pdf_path}")


if __name__ == "__main__":
    main()
