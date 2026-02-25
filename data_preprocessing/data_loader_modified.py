"""
智能数据加载器
处理元素、化合物、材料类型等多种数据
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple, cast
from collections import Counter, defaultdict
import csv
import os

import numpy as np
import pandas as pd
import re


# -----------------------------
# 缺失值处理
# -----------------------------
_MISSING_STRINGS = {
    "", " ", "none", "nan", "na", "n/a",
    "NONE", "NaN", "NA", "N/A", "None"
}

_DEFAULT_NA_STRINGS = {
    "", " ", "none", "nan", "na", "n/a",
    "NONE", "NaN", "NA", "N/A", "None"
}


class MaterialFeaturizer(Protocol):
    @property
    def dim(self) -> int:
        ...

    @property
    def feature_labels(self) -> List[str]:
        ...

    def featurize(self, material: Optional[str]) -> np.ndarray:
        ...

    def print_stats(self) -> None:
        ...


def _normalize_missing(x):
    """Convert common 'null-like' tokens to np.nan; keep other values unchanged."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip()
        if s in _MISSING_STRINGS:
            return np.nan
        return s
    return x


def _null_token_vector(dim: int) -> np.ndarray:
    """Return a stable vector for explicit 'Null' tokens (not treated as missing)."""
    vec = np.zeros(dim, dtype=np.float32)
    if dim > 0:
        vec[-1] = -1.0
    return vec


def _resolve_y_cols(df: pd.DataFrame, y_cols: Optional[Sequence[str]]) -> List[str]:
    """Resolve target columns using the same fallback logic as the main loader."""
    if y_cols is None:
        preferred_y = [
            "CO selectivity (%)",
            "Methanol selectivity (%)",
            "STY_CH3OH (g/kg·h) (LN scale)",
            "CO2 conversion efficiency (%)",
        ]
        if all(c in df.columns for c in preferred_y):
            y_cols = preferred_y
        else:
            cols14 = list(df.columns[:14])
            y_cols = cols14[10:14]
    return [c.strip() for c in y_cols]


def _resolve_promoter_ratio_cols(
    element_cols: Sequence[str],
    promoter_ratio_cols: Optional[Sequence[str]],
    df: pd.DataFrame
) -> Dict[str, Optional[str]]:
    """Resolve ratio columns for each promoter column (by config or heuristic)."""
    ratio_map: Dict[str, Optional[str]] = {}
    if promoter_ratio_cols is None:
        # Heuristic: find a column containing both the promoter name and "ratio".
        cols_lower = {c.lower(): c for c in df.columns}
        for col in element_cols:
            col_l = str(col).lower()
            match = None
            for c in df.columns:
                c_l = str(c).lower()
                if "ratio" in c_l and col_l in c_l:
                    match = c
                    break
            ratio_map[col] = match
        return ratio_map

    if isinstance(promoter_ratio_cols, dict):
        for col in element_cols:
            ratio_map[col] = promoter_ratio_cols.get(col)
        return ratio_map

    if len(promoter_ratio_cols) != len(element_cols):
        raise ValueError("promoter_ratio_cols length must match element_cols length.")
    for col, rcol in zip(element_cols, promoter_ratio_cols):
        ratio_map[col] = rcol
    return ratio_map


def _is_explicit_null_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "null"


def _stable_seed_from_text(text: str) -> int:
    return int(sum(ord(ch) for ch in text) % 100000)


def _sample_numeric_kde(
    observed: np.ndarray,
    n_samples: int,
    seed: int
) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float32)
    observed = observed[np.isfinite(observed)]
    if n_samples <= 0:
        return np.zeros((0,), dtype=np.float32)
    if observed.size == 0:
        return np.zeros((n_samples,), dtype=np.float32)
    if observed.size < 2:
        fill_val = float(np.nanmedian(observed))
        if not np.isfinite(fill_val):
            fill_val = 0.0
        return np.full((n_samples,), fill_val, dtype=np.float32)

    try:
        from scipy.stats import gaussian_kde
    except Exception:
        fill_val = float(np.nanmedian(observed))
        if not np.isfinite(fill_val):
            fill_val = 0.0
        return np.full((n_samples,), fill_val, dtype=np.float32)

    try:
        kde = gaussian_kde(observed)
        rng = np.random.default_rng(seed)
        try:
            samples = kde.resample(n_samples, seed=rng).reshape(-1)
        except TypeError:
            state = np.random.get_state()
            np.random.seed(seed)
            try:
                samples = kde.resample(n_samples).reshape(-1)
            finally:
                np.random.set_state(state)
    except Exception:
        fill_val = float(np.nanmedian(observed))
        if not np.isfinite(fill_val):
            fill_val = 0.0
        samples = np.full((n_samples,), fill_val, dtype=np.float32)

    samples = np.clip(samples, 0.0, 1e9)
    return np.asarray(samples, dtype=np.float32)


def _apply_promoter_ratio_missing_rules(
    df: pd.DataFrame,
    element_cols: Sequence[str],
    ratio_col_map: Dict[str, Optional[str]],
    impute_seed: int = 42,
    prefer_kde: bool = True,
) -> pd.DataFrame:
    """
    Ratio rules:
    1) If promoter is explicit "Null" -> ratio must be 0.
    2) For non-null promoters with missing ratio -> impute (prefer KDE).
    """
    df = df.copy()
    for idx, pcol in enumerate(element_cols):
        rcol = ratio_col_map.get(pcol)
        if not rcol or pcol not in df.columns or rcol not in df.columns:
            continue

        promoter_series = df[pcol]
        ratio_series = pd.to_numeric(df[rcol], errors="coerce")
        null_mask = promoter_series.map(_is_explicit_null_token).fillna(False)
        ratio_series.loc[null_mask] = 0.0

        missing_non_null = ratio_series.isna() & (~null_mask)
        if missing_non_null.any():
            promoter_tokens = promoter_series.map(_clean_material_token)
            observed_global = ratio_series[(~ratio_series.isna()) & (~null_mask)].to_numpy(dtype=np.float32)

            # Fill per promoter token first, then fallback to global pool.
            for token in promoter_tokens[missing_non_null].unique():
                if not token:
                    continue
                grp_missing = missing_non_null & promoter_tokens.eq(token)
                n_grp = int(grp_missing.sum())
                if n_grp <= 0:
                    continue
                grp_obs = ratio_series[
                    (~ratio_series.isna()) & (~null_mask) & promoter_tokens.eq(token)
                ].to_numpy(dtype=np.float32)
                base = grp_obs if grp_obs.size >= 2 else observed_global

                if prefer_kde:
                    samples = _sample_numeric_kde(
                        base,
                        n_grp,
                        seed=int(impute_seed + idx * 1009 + _stable_seed_from_text(token))
                    )
                else:
                    fill_val = float(np.nanmean(base)) if base.size > 0 else 0.0
                    if not np.isfinite(fill_val):
                        fill_val = 0.0
                    samples = np.full((n_grp,), fill_val, dtype=np.float32)
                ratio_series.loc[grp_missing] = samples

            remaining = ratio_series.isna() & (~null_mask)
            if remaining.any():
                n_rem = int(remaining.sum())
                if prefer_kde:
                    samples = _sample_numeric_kde(
                        observed_global,
                        n_rem,
                        seed=int(impute_seed + idx * 2029 + 17)
                    )
                else:
                    fill_val = float(np.nanmean(observed_global)) if observed_global.size > 0 else 0.0
                    if not np.isfinite(fill_val):
                        fill_val = 0.0
                    samples = np.full((n_rem,), fill_val, dtype=np.float32)
                ratio_series.loc[remaining] = samples

        if ratio_series.isna().any():
            fallback = float(np.nanmedian(ratio_series.to_numpy(dtype=np.float32)))
            if not np.isfinite(fallback):
                fallback = 0.0
            ratio_series = ratio_series.fillna(fallback)

        df[rcol] = np.clip(ratio_series.to_numpy(dtype=np.float32), 0.0, 1e9)
    return df


def _clean_material_token(value: Any) -> str:
    """Normalize material token used by interaction features."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"none", "nan"}:
        return ""
    return text


def _build_promoter_pair_interaction_block(
    left_col: str,
    right_col: str,
    left_tokens: np.ndarray,
    right_tokens: np.ndarray,
    left_ratio: np.ndarray,
    right_ratio: np.ndarray,
    left_base_vec: np.ndarray,
    right_base_vec: np.ndarray,
    left_is_null: np.ndarray,
    right_is_null: np.ndarray,
    eps: float = 1e-8,
    add_pair_onehot: bool = True,
    pair_onehot_min_count: int = 2,
    pair_onehot_max_categories: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build cross features to bind promoter1/promoter2 identity and ratio.
    """
    if left_ratio.shape != right_ratio.shape:
        raise ValueError("Left/right ratio vectors must have the same shape.")
    if left_base_vec.shape != right_base_vec.shape:
        raise ValueError("Left/right base vectors must have the same shape.")

    N = left_ratio.shape[0]
    if N == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    eps = float(eps) if eps is not None else 1e-8
    if eps <= 0:
        eps = 1e-8

    ratio_sum = left_ratio + right_ratio
    ratio_diff = left_ratio - right_ratio
    ratio_abs_diff = np.abs(ratio_diff)
    ratio_prod = left_ratio * right_ratio
    ratio_share_left = left_ratio / (ratio_sum + eps)
    ratio_share_right = right_ratio / (ratio_sum + eps)
    ratio_min = np.minimum(left_ratio, right_ratio)
    ratio_max = np.maximum(left_ratio, right_ratio)
    ratio_balance = ratio_min / (ratio_max + eps)

    dot = np.sum(left_base_vec * right_base_vec, axis=1)
    norm_left = np.linalg.norm(left_base_vec, axis=1)
    norm_right = np.linalg.norm(right_base_vec, axis=1)
    cosine = dot / (norm_left * norm_right + eps)
    l2_dist = np.linalg.norm(left_base_vec - right_base_vec, axis=1)
    weighted_dot = dot * ratio_prod

    same_material = (
        (left_tokens == right_tokens) & (left_tokens != "") & (right_tokens != "")
    ).astype(np.float32)
    any_null = ((left_is_null > 0.5) | (right_is_null > 0.5)).astype(np.float32)
    both_null = ((left_is_null > 0.5) & (right_is_null > 0.5)).astype(np.float32)
    both_non_null = ((left_is_null < 0.5) & (right_is_null < 0.5)).astype(np.float32)

    pair_prefix = f"{left_col}__x__{right_col}"
    dense_block = np.column_stack(
        [
            ratio_sum,
            ratio_diff,
            ratio_abs_diff,
            ratio_prod,
            ratio_share_left,
            ratio_share_right,
            ratio_min,
            ratio_max,
            ratio_balance,
            dot,
            cosine,
            l2_dist,
            weighted_dot,
            same_material,
            any_null,
            both_null,
            both_non_null,
        ]
    ).astype(np.float32)
    dense_labels = [
        f"{pair_prefix}__ratio_sum",
        f"{pair_prefix}__ratio_diff",
        f"{pair_prefix}__ratio_abs_diff",
        f"{pair_prefix}__ratio_prod",
        f"{pair_prefix}__ratio_share_left",
        f"{pair_prefix}__ratio_share_right",
        f"{pair_prefix}__ratio_min",
        f"{pair_prefix}__ratio_max",
        f"{pair_prefix}__ratio_balance",
        f"{pair_prefix}__embed_dot",
        f"{pair_prefix}__embed_cosine",
        f"{pair_prefix}__embed_l2",
        f"{pair_prefix}__embed_weighted_dot",
        f"{pair_prefix}__same_material",
        f"{pair_prefix}__any_null",
        f"{pair_prefix}__both_null",
        f"{pair_prefix}__both_non_null",
    ]

    if not add_pair_onehot:
        return dense_block, dense_labels

    min_count = max(int(pair_onehot_min_count), 1)
    max_categories = int(pair_onehot_max_categories)
    left_safe = np.where(left_tokens == "", "__MISSING__", left_tokens)
    right_safe = np.where(right_tokens == "", "__MISSING__", right_tokens)
    pair_tokens = np.array(
        [f"{a}__PAIR__{b}" for a, b in zip(left_safe, right_safe)],
        dtype=object
    )
    pair_counts = Counter(cast(List[str], pair_tokens.tolist()))
    if not pair_counts:
        return dense_block, dense_labels

    kept = [tok for tok, cnt in pair_counts.items() if cnt >= min_count]
    kept.sort(key=lambda t: (-pair_counts[t], t))
    if max_categories > 0 and len(kept) > max_categories:
        kept = kept[:max_categories]
    if not kept:
        kept = [max(pair_counts.items(), key=lambda kv: kv[1])[0]]

    use_other = len(kept) < len(pair_counts)
    labels = list(kept)
    if use_other:
        labels.append("__OTHER__")

    idx_map = {tok: i for i, tok in enumerate(labels)}
    onehot = np.zeros((N, len(labels)), dtype=np.float32)
    other_idx = idx_map.get("__OTHER__")
    for i, tok in enumerate(cast(List[str], pair_tokens.tolist())):
        j = idx_map.get(tok)
        if j is None:
            if other_idx is None:
                continue
            j = other_idx
        onehot[i, j] = 1.0

    onehot_labels = [f"{pair_prefix}__pair__{tok}" for tok in labels]
    return np.concatenate([dense_block, onehot], axis=1), dense_labels + onehot_labels


def save_duplicate_input_conflict_report(
    csv_path: str,
    y_cols: Optional[Sequence[str]] = None,
    drop_metadata_cols: Sequence[str] = ("DOI", "Name", "Year"),
    output_dir: Optional[str] = None,
    output_prefix: str = "duplicate_input",
    preserve_null: bool = True,
) -> Tuple[str, str, int, int]:
    """
    Save a conflict report for rows with identical inputs but different outputs.

    Outputs:
      - {output_prefix}_conflicts.csv: row-level conflict list
      - {output_prefix}_aggregated.csv: per-group aggregation (mean/median/min/max)

    Returns: (conflict_path, aggregated_path, conflict_group_count, conflict_row_count)
    """
    df = _read_csv_with_missing(csv_path, preserve_null=preserve_null)
    df = _strip_colnames(df)
    df = _drop_unnamed_cols(df)

    # Normalize common missing tokens, but keep explicit "Null" as a real value.
    for c in df.columns:
        df[c] = df[c].map(_normalize_missing)

    y_cols = _resolve_y_cols(df, y_cols)
    drop_set = set(y_cols) | set(drop_metadata_cols)
    input_cols = [c for c in df.columns if c not in drop_set]

    if not input_cols or not y_cols:
        return ("", "", 0, 0)

    # Prepare inputs/outputs for grouping.
    input_df = df[input_cols].where(~df[input_cols].isna(), "")
    y_raw = df[y_cols].copy()
    y_norm = y_raw.apply(pd.to_numeric, errors="coerce")

    # Group by exact input tuple and check output conflicts.
    groups: Dict[Tuple[Any, ...], List[Tuple[int, Tuple[Any, ...], Tuple[Any, ...]]]] = defaultdict(list)
    row_index_offset = 2  # CSV row number: +1 for header, +1 for 1-based index
    for i, (inp_row, y_raw_row, y_norm_row) in enumerate(
        zip(input_df.itertuples(index=False),
            y_raw.itertuples(index=False),
            y_norm.itertuples(index=False)),
        start=row_index_offset
    ):
        inp = tuple(inp_row)
        y_norm_key = tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in y_norm_row)
        groups[inp].append((i, y_norm_key, tuple(y_raw_row)))

    conflict_groups = []
    for inp, items in groups.items():
        if len(items) <= 1:
            continue
        uniq = {it[1] for it in items}
        if len(uniq) > 1:
            conflict_groups.append((inp, items))

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    conflict_path = os.path.join(output_dir, f"{output_prefix}_conflicts.csv")
    aggregated_path = os.path.join(output_dir, f"{output_prefix}_aggregated.csv")

    # Write row-level conflicts.
    with open(conflict_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "row_index"] + input_cols + y_cols)
        for gid, (inp, items) in enumerate(conflict_groups, start=1):
            for row_idx, _out_norm, out_raw in items:
                w.writerow([gid, row_idx] + list(inp) + list(out_raw))

    # Write aggregated suggestions per group.
    with open(aggregated_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header_out = ["group_id", "group_size"] + input_cols
        for c in y_cols:
            header_out += [f"{c}__mean", f"{c}__median", f"{c}__min", f"{c}__max"]
        w.writerow(header_out)

        for gid, (inp, items) in enumerate(conflict_groups, start=1):
            out_lists: List[List[float]] = [[] for _ in y_cols]
            for _row_idx, _out_norm, out_raw in items:
                for k, v in enumerate(out_raw):
                    try:
                        x = float(v)
                        if not np.isnan(x):
                            out_lists[k].append(x)
                    except Exception:
                        pass
            row_out: List[Any] = [gid, len(items)] + list(inp)
            for vals in out_lists:
                if vals:
                    row_out += [
                        float(np.mean(vals)),
                        float(np.median(vals)),
                        float(np.min(vals)),
                        float(np.max(vals)),
                    ]
                else:
                    row_out += ["", "", "", ""]
            w.writerow(row_out)

    return (conflict_path, aggregated_path, len(conflict_groups),
            sum(len(v) for _, v in conflict_groups))


def _aggregate_duplicate_input_rows(
    df: pd.DataFrame,
    y_cols: Sequence[str],
    drop_metadata_cols: Sequence[str] = (),
    target_agg: str = "median",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Aggregate rows with identical inputs (excluding metadata columns).
    Targets are aggregated by mean/median to reduce label noise.
    """
    y_cols = [c for c in y_cols if c in df.columns]
    if not y_cols:
        return df, {
            "original_rows": int(df.shape[0]),
            "dedup_rows": int(df.shape[0]),
            "reduced_rows": 0,
            "duplicate_groups": 0,
            "conflict_groups": 0,
            "conflict_rows": 0,
        }

    agg_mode = str(target_agg).strip().lower()
    if agg_mode not in {"mean", "median"}:
        raise ValueError("duplicate target aggregation must be 'mean' or 'median'.")

    drop_set = set(y_cols) | set(drop_metadata_cols)
    input_cols = [c for c in df.columns if c not in drop_set]
    if not input_cols:
        return df, {
            "original_rows": int(df.shape[0]),
            "dedup_rows": int(df.shape[0]),
            "reduced_rows": 0,
            "duplicate_groups": 0,
            "conflict_groups": 0,
            "conflict_rows": 0,
        }

    rows_out: List[pd.Series] = []
    duplicate_groups = 0
    conflict_groups = 0
    conflict_rows = 0

    for _, grp in df.groupby(input_cols, dropna=False, sort=False):
        g = grp.copy()
        n = int(g.shape[0])
        if n > 1:
            duplicate_groups += 1

        y_num = g[y_cols].apply(pd.to_numeric, errors="coerce")
        y_keys = {
            tuple(None if np.isnan(v) else float(v) for v in row)
            for row in y_num.to_numpy(dtype=float)
        }
        if len(y_keys) > 1:
            conflict_groups += 1
            conflict_rows += n

        row = g.iloc[0].copy()
        for ycol in y_cols:
            vals = y_num[ycol].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row[ycol] = np.nan
            elif agg_mode == "median":
                row[ycol] = float(np.nanmedian(vals))
            else:
                row[ycol] = float(np.nanmean(vals))
        rows_out.append(row)

    dedup_df = pd.DataFrame(rows_out, columns=df.columns).reset_index(drop=True)
    stats = {
        "original_rows": int(df.shape[0]),
        "dedup_rows": int(dedup_df.shape[0]),
        "reduced_rows": int(df.shape[0] - dedup_df.shape[0]),
        "duplicate_groups": int(duplicate_groups),
        "conflict_groups": int(conflict_groups),
        "conflict_rows": int(conflict_rows),
    }
    return dedup_df, stats


def _read_csv_with_missing(csv_path: str, preserve_null: bool = True) -> pd.DataFrame:
    if preserve_null:
        na_values = sorted(_DEFAULT_NA_STRINGS)
        return pd.read_csv(csv_path, keep_default_na=False, na_values=na_values)
    return pd.read_csv(csv_path)


def _strip_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def _drop_unnamed_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in df.columns if isinstance(c, str) and c.strip().lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _impute_missing_values_kde(
    df: pd.DataFrame,
    element_cols: Sequence[str] = (),
    text_cols: Sequence[str] = (),
    skip_cols: Sequence[str] = (),
    type_col_substring: str = "Type",
    skip_name_substring: str = "ame",
    random_seed: int = 42,
    missing_text_token: str = "__MISSING__",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    KDE-based missing value imputation:
    - Numeric columns: fit gaussian_kde on observed values, then sample to fill NaNs.
    - Categorical columns: sample from empirical distribution.
    """
    df = df.copy()
    np.random.seed(random_seed)

    element_set = set(element_cols)
    text_set = set(text_cols)
    skip_set = set(skip_cols)

    cols_with_na = [c for c in df.columns if df[c].isna().any()]
    number_cols: List[str] = []
    type_cols: List[str] = []

    for col in cols_with_na:
        if col in skip_set:
            continue
        col_str = str(col)
        if skip_name_substring and skip_name_substring in col_str:
            continue
        if col in element_set or col in text_set:
            type_cols.append(col)
        elif type_col_substring and type_col_substring in col_str:
            type_cols.append(col)
        else:
            number_cols.append(col)

    try:
        from scipy.stats import gaussian_kde
    except Exception:
        gaussian_kde = None
        if number_cols:
            print("[WARN] scipy not available; falling back to median fill for numeric columns.")

    extra_type_cols: List[str] = []

    # Numeric columns imputation via KDE
    for col_name in number_cols:
        col_raw = df[col_name]
        col_num = pd.to_numeric(col_raw, errors="coerce")
        observed = np.asarray(col_num.dropna().to_numpy(dtype=np.float32))
        n_missing = int(col_num.isna().sum())

        # If non-numeric column was misclassified, fall back to categorical imputation.
        raw_non_null = np.asarray(col_raw.dropna().to_numpy())
        if observed.size == 0 and raw_non_null.size > 0:
            extra_type_cols.append(col_name)
            continue

        if n_missing == 0:
            df[col_name] = col_num
            continue

        if observed.size == 0:
            samples = np.zeros(n_missing, dtype=np.float32)
        elif gaussian_kde is None or observed.size < 2:
            fill_val = float(np.nanmedian(observed)) if observed.size > 0 else 0.0
            samples = np.full(n_missing, fill_val, dtype=np.float32)
        else:
            try:
                kde = gaussian_kde(observed)
                samples = kde.resample(n_missing).flatten()
            except Exception:
                fill_val = float(np.nanmedian(observed)) if observed.size > 0 else 0.0
                samples = np.full(n_missing, fill_val, dtype=np.float32)

        if "Calcination time" in str(col_name):
            samples = np.clip(np.round(samples), 0, 200)
        else:
            samples = np.clip(samples, 0, 1e9)

        col_num.loc[col_num.isna()] = samples
        df[col_name] = col_num

    # Categorical columns imputation via empirical distribution
    type_cols = type_cols + extra_type_cols
    for col_name in type_cols:
        col = df[col_name]
        n_missing = int(col.isna().sum())
        if n_missing == 0:
            continue

        non_null = col.dropna()
        if non_null.empty:
            samples = [missing_text_token] * n_missing
        else:
            vals, counts = np.unique(non_null, return_counts=True)
            probs = counts / counts.sum()
            samples = np.random.choice(vals, size=n_missing, p=probs)
        df.loc[col.isna(), col_name] = samples

    if verbose:
        print(f"[INFO] KDE impute numeric cols: {number_cols}")
        print(f"[INFO] Categorical impute cols: {type_cols}")

    return df, {"numeric_cols": number_cols, "type_cols": type_cols}


def _impute_missing_values_simple(
    df: pd.DataFrame,
    skip_cols: Sequence[str] = (),
    missing_text_token: str = "__MISSING__",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Simple missing value handling aligned with the original one-hot loader:
    - numeric columns: fill NaNs with mean
    - non-numeric columns: fill NaNs with mode (fallback to missing_text_token)
    """
    df = df.copy()
    skip_set = set(skip_cols)
    for col in df.columns:
        if col in skip_set:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            mean_val = series.mean(skipna=True)
            if pd.isna(mean_val):
                mean_val = 0.0
            df[col] = series.fillna(mean_val)
        else:
            if series.isna().any():
                mode_vals = series.mode(dropna=True)
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else missing_text_token
                df[col] = series.fillna(fill_val)
    if verbose:
        print("[INFO] Simple impute applied (mean/mode).")
    return df


# -----------------------------
# 智能材料特征器
# -----------------------------
class AdvancedMaterialFeaturizer:
    """
    智能材料特征器，处理多种类型：
    1. 单一元素 (Cu, Zn) -> Magpie特征
    2. 化合物/合金 (CuZn, Al2O3) -> 元素加权平均特征
    3. 材料类型 (Zeolite, MOF) -> 预定义特征
    4. 文本描述 -> NLP嵌入特征
    5. 无法识别 -> 特殊处理
    """
    def __init__(self):
        self._magpie: MagpieFeaturizer | None = None
        self._common_elements = None
        self.stats = {
            'single_elements': Counter(),
            'compounds': Counter(),
            'materials': Counter(),
            'text_descriptions': Counter(),
            'unknown': Counter(),
            'failed': Counter()
        }

    @property
    def magpie(self) -> "MagpieFeaturizer":
        if self._magpie is None:
            self._magpie = MagpieFeaturizer()
        return self._magpie

    @property
    def dim(self) -> int:
        return self.magpie.dim

    @property
    def feature_labels(self) -> List[str]:
        return self.magpie.feature_labels

    def _is_single_element(self, text: str) -> bool:
        """检查是否是单一元素符号"""
        if not text or not isinstance(text, str):
            return False

        # 检查常见元素模式
        element_pattern = r'^[A-Z][a-z]?$'
        if re.match(element_pattern, text):
            # 常见元素列表
            common_elements = {
                'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
            }
            if text in common_elements:
                return True

        return False

    def _parse_chemical_formula(self, text: str):
        """尝试解析化学式"""
        try:
            from pymatgen.core import Composition

            # 清理文本
            text = text.strip()
            if not text:
                return None

            # 尝试解析
            comp = Composition(text)

            # 获取成分字典，处理新老版本pymatgen差异
            try:
                # 尝试新版本的API
                composition_dict = comp.as_reduced_dict
                # 如果是可调用对象，调用它
                if callable(composition_dict):
                    composition_dict = composition_dict()
            except AttributeError:
                # 回退到旧版本API
                try:
                    composition_dict = comp.to_reduced_dict
                    if callable(composition_dict):
                        composition_dict = composition_dict()
                except AttributeError:
                    # 如果两种方法都失败，尝试直接获取
                    composition_dict = comp.get_reduced_dict() if hasattr(comp, 'get_reduced_dict') else {}

            elements = comp.elements

            return {
                'elements': elements,
                'composition': composition_dict,
                'is_single': len(elements) == 1
            }

        except Exception as e:
            return None

    def _get_material_features(self, material_type: str) -> np.ndarray:
        """获取材料类型的特征"""
        material_lower = material_type.lower()

        # 常见材料类型特征
        if 'zeolite' in material_lower:
            # 沸石特征：高表面积、酸性、微孔
            features = np.zeros(self.dim, dtype=np.float32)
            features[0] = 0.8   # 假设高表面积
            features[1] = 0.6   # 假设酸性
            features[2] = 0.7   # 假设微孔结构
            return features

        elif 'mof' in material_lower:
            # MOF特征：极高表面积、可调孔隙
            features = np.zeros(self.dim, dtype=np.float32)
            features[0] = 0.9   # 极高表面积
            features[3] = 0.8   # 可调孔隙
            return features

        elif 'perovskite' in material_lower:
            # 钙钛矿特征
            features = np.zeros(self.dim, dtype=np.float32)
            features[4] = 0.7   # 钙钛矿结构特征
            return features

        elif 'alumina' in material_lower or 'al2o3' in material_lower:
            # 氧化铝特征
            features = np.zeros(self.dim, dtype=np.float32)
            features[5] = 0.6   # 氧化铝特征
            return features

        elif 'silica' in material_lower or 'sio2' in material_lower:
            # 二氧化硅特征
            features = np.zeros(self.dim, dtype=np.float32)
            features[6] = 0.6   # 二氧化硅特征
            return features

        else:
            # 未知材料类型，返回平均特征
            return self._get_mean_element_features()

    def _get_mean_element_features(self) -> np.ndarray:
        """获取常见元素的平均特征"""
        if self._common_elements is None:
            # 常见催化元素
            common_catalyst_elements = [
                'Cu', 'Zn', 'Al', 'Zr', 'Ti', 'Ce', 'Mg', 'Ca',
                'Ni', 'Co', 'Fe', 'Mn', 'Cr', 'V', 'Mo', 'W',
                'Pd', 'Pt', 'Rh', 'Ru', 'Ir', 'Au', 'Ag',
                'Si', 'Sn', 'Pb', 'Bi', 'Sb', 'Te'
            ]

            features = []
            for el in common_catalyst_elements:
                try:
                    feat = self.magpie.featurize(el)
                    if not np.all(feat == 0):  # 跳过零向量
                        features.append(feat)
                except:
                    continue

            if features:
                self._common_elements = np.mean(features, axis=0)
            else:
                self._common_elements = np.zeros(self.dim, dtype=np.float32)

        return self._common_elements.copy()

    def _handle_compound(self, formula: str, composition_data: dict) -> np.ndarray:
        """处理化合物：计算组成元素的加权平均特征"""
        composition_dict = composition_data['composition']

        # 确保composition_dict是字典
        if not isinstance(composition_dict, dict):
            print(f"警告: composition_dict不是字典类型，而是{type(composition_dict)}")
            if hasattr(composition_dict, '__call__'):
                try:
                    composition_dict = composition_dict()
                except:
                    return self._get_mean_element_features()
            else:
                return self._get_mean_element_features()

        features = []
        weights = []

        for element_symbol, amount in composition_dict.items():
            try:
                # 获取元素特征
                elem_feat = self.magpie.featurize(element_symbol)
                if not np.all(elem_feat == 0):  # 跳过零向量
                    features.append(elem_feat)
                    weights.append(amount)
            except:
                continue

        if features:
            # 按原子数加权平均
            features_array = np.array(features)
            weights_array = np.array(weights)

            # 归一化权重
            if weights_array.sum() > 0:
                weights_array = weights_array / weights_array.sum()

            # 加权平均
            weighted_avg = np.average(features_array, axis=0, weights=weights_array)

            # 添加化合物标志（可选）
            if weighted_avg.shape[0] > 0:
                weighted_avg[-1] = 0.5  # 化合物标志

            return weighted_avg
        else:
            # 所有元素都无法识别，返回平均特征
            return self._get_mean_element_features()

    def _handle_text_description(self, text: str) -> np.ndarray:
        """处理文本描述：使用NLP嵌入"""
        # 不使用文本嵌入器：直接返回特殊标记向量
        features = np.ones(self.dim, dtype=np.float32) * 0.1  # 特殊值，区别于0
        if features.shape[0] > 0:
            features[-1] = 0.8  # 文本描述标志
        return features

    def featurize(self, material: Optional[str]) -> np.ndarray:
        """
        智能材料特征化

        Args:
            material: 材料描述，可以是元素、化合物、材料类型或文本

        Returns:
            Magpie维度的特征向量
        """
        # 处理缺失值
        if material is None or pd.isna(material):
            self.stats['unknown']['<MISSING>'] += 1
            return np.zeros(self.dim, dtype=np.float32)

        material_str = str(material).strip()

        if not material_str or material_str.lower() in {'none', 'nan', ''}:
            self.stats['unknown']['<EMPTY>'] += 1
            return np.zeros(self.dim, dtype=np.float32)

        if material_str.lower() == 'null':
            self.stats['unknown']['<NULL>'] += 1
            return _null_token_vector(self.dim)

        # 1. 检查是否是单一元素
        if self._is_single_element(material_str):
            try:
                feat = self.magpie.featurize(material_str)
                if not np.all(feat == 0):  # 成功识别
                    self.stats['single_elements'][material_str] += 1
                    return feat
            except:
                pass

        # 2. 尝试解析为化学式
        composition_data = self._parse_chemical_formula(material_str)
        if composition_data:
            if composition_data['is_single']:
                # 实际上是单一元素，但可能格式不同（如"Cu1"）
                element_symbol = str(composition_data['elements'][0])
                feat = self.magpie.featurize(element_symbol)
                if not np.all(feat == 0):
                    self.stats['single_elements'][element_symbol] += 1
                    return feat
            else:
                # 化合物
                self.stats['compounds'][material_str] += 1
                return self._handle_compound(material_str, composition_data)

        # 3. 检查是否是已知材料类型
        material_lower = material_str.lower()
        known_materials = {
            'zeolite', 'mof', 'perovskite', 'alumina', 'silica',
            'zsm-5', 'beta', 'zif-8', 'alfum', 'al_fum', 'al fumarate'
        }

        for known_mat in known_materials:
            if known_mat in material_lower:
                self.stats['materials'][material_str] += 1
                return self._get_material_features(material_str)

        # 4. 尝试提取可能的元素符号
        # 查找大写字母开头的单词作为可能的元素符号
        element_pattern = r'\b[A-Z][a-z]?\b'
        possible_elements = re.findall(element_pattern, material_str)

        if possible_elements:
            # 取第一个可能的元素
            first_element = possible_elements[0]
            try:
                feat = self.magpie.featurize(first_element)
                if not np.all(feat == 0):
                    self.stats['single_elements'][first_element] += 1
                    return feat
            except:
                pass

        # 5. 作为文本描述处理
        self.stats['text_descriptions'][material_str] += 1
        return self._handle_text_description(material_str)

    def print_stats(self):
        """打印特征化统计信息"""
        print("\n材料特征化统计:")
        print(f"  单一元素: {sum(self.stats['single_elements'].values())} 次")
        if self.stats['single_elements']:
            elements = list(self.stats['single_elements'].items())
            print(f"    示例: {elements[:10] if len(elements) > 10 else elements}")

        print(f"  化合物: {sum(self.stats['compounds'].values())} 次")
        if self.stats['compounds']:
            compounds = list(self.stats['compounds'].items())
            print(f"    示例: {compounds[:10] if len(compounds) > 10 else compounds}")

        print(f"  材料类型: {sum(self.stats['materials'].values())} 次")
        if self.stats['materials']:
            materials = list(self.stats['materials'].items())
            print(f"    示例: {materials[:10] if len(materials) > 10 else materials}")

        print(f"  文本描述: {sum(self.stats['text_descriptions'].values())} 次")
        if self.stats['text_descriptions']:
            texts = list(self.stats['text_descriptions'].items())
            print(f"    示例: {texts[:5] if len(texts) > 5 else texts}")

        print(f"  未知/缺失: {sum(self.stats['unknown'].values())} 次")


# -----------------------------
# 原始MagpieFeaturizer (保持兼容)
# -----------------------------
@dataclass
class MagpieFeaturizer:
    """原始Magpie元素特征器"""
    _featurizer: Any = None
    _Composition: Any = None
    _feature_labels: Optional[List[str]] = None

    def _ensure(self):
        if self._featurizer is not None:
            return

        try:
            from pymatgen.core import Composition
            from matminer.featurizers.composition import ElementProperty
        except Exception as e:
            raise ImportError(
                "Magpie featurization requires `pymatgen` and `matminer`.\n"
                "Install with:\n"
                "  pip install pymatgen matminer\n"
            ) from e

        self._Composition = Composition
        self._featurizer = ElementProperty.from_preset("magpie")
        self._feature_labels = list(self._featurizer.feature_labels())

    @property
    def dim(self) -> int:
        self._ensure()
        assert self._feature_labels is not None
        return len(self._feature_labels)

    @property
    def feature_labels(self) -> List[str]:
        self._ensure()
        assert self._feature_labels is not None
        return self._feature_labels

    def featurize(self, material: Optional[str]) -> np.ndarray:
        self._ensure()
        if material is None or (isinstance(material, float) and np.isnan(material)):
            return np.zeros(self.dim, dtype=np.float32)

        sym = str(material).strip()
        if sym == "" or sym.lower() in {"none", "nan"}:
            return np.zeros(self.dim, dtype=np.float32)
        if sym.lower() == "null":
            return _null_token_vector(self.dim)

        try:
            comp = self._Composition(f"{sym}1")
            vec = np.asarray(self._featurizer.featurize(comp), dtype=np.float32)
        except Exception:
            vec = np.zeros(self.dim, dtype=np.float32)
        return vec

    def print_stats(self) -> None:
        """兼容接口：Magpie 本身不统计，保持空实现。"""
        return None


# -----------------------------
# 简化的智能数据加载器
# -----------------------------
def load_smart_data_simple(
    csv_path: str,
    element_cols: tuple[str, ...] = ("Promoter 1", "Promoter 2"),
    text_cols: Tuple[str, ...] = ("Type of sysnthesis procedure",),
    y_cols: Optional[Sequence[str]] = None,
    promoter_ratio_cols: Optional[Sequence[str]] = None,
    promoter_onehot: bool = True,
    promoter_interaction_features: bool = True,
    promoter_pair_onehot: bool = True,
    promoter_pair_onehot_min_count: int = 2,
    promoter_pair_onehot_max_categories: int = 64,
    promoter_interaction_eps: float = 1e-8,
    log_transform_cols: Optional[Sequence[str]] = None,
    log_transform_eps: float = 1e-8,
    # 特征器配置
    element_embedding: str = "advanced",  # "basic" 或 "advanced"
    # 其他配置
    drop_metadata_cols: Tuple[str, ...] = ("DOI", "Name", "Year"),
    fill_numeric: str = "median",
    missing_text_token: str = "__MISSING__",
    impute_missing: bool = True,
    impute_method: str = "simple",
    impute_seed: int = 42,
    preserve_null: bool = True,
    impute_type_substring: str = "Type",
    impute_skip_substring: str = "ame",
    aggregate_duplicate_inputs: bool = False,
    duplicate_target_agg: str = "median",
    return_dataframe: bool = False,
):
    """
    简化的智能数据加载器：
    - Promoter 向量 + ratio 加权 + Null 指示
    - 可选 Promoter1/Promoter2 交互特征（ratio 与类别绑定）
    """
    df = _read_csv_with_missing(csv_path, preserve_null=preserve_null)
    df = _strip_colnames(df)
    df = _drop_unnamed_cols(df)

    # 归一化缺失值
    for c in df.columns:
        df[c] = df[c].map(_normalize_missing)

    # 先确定目标列，便于重复输入聚合时聚合Y
    y_cols = _resolve_y_cols(df, y_cols)

    # 先对重复输入做聚合，降低标签冲突噪声
    if aggregate_duplicate_inputs:
        df, dup_stats = _aggregate_duplicate_input_rows(
            df,
            y_cols=y_cols,
            drop_metadata_cols=drop_metadata_cols,
            target_agg=duplicate_target_agg,
        )
        print(
            "[INFO] Duplicate-input aggregation: "
            f"{dup_stats['original_rows']} -> {dup_stats['dedup_rows']} rows "
            f"(reduced={dup_stats['reduced_rows']}, "
            f"duplicate_groups={dup_stats['duplicate_groups']}, "
            f"conflict_groups={dup_stats['conflict_groups']}, "
            f"conflict_rows={dup_stats['conflict_rows']})"
        )

    element_cols = tuple(c.strip() for c in element_cols)
    text_cols = tuple(c.strip() for c in text_cols)
    ratio_col_map = _resolve_promoter_ratio_cols(element_cols, promoter_ratio_cols, df)
    ratio_cols = tuple(c for c in ratio_col_map.values() if c and c in df.columns)
    skip_cols = tuple(drop_metadata_cols) + ratio_cols

    # 通用缺失值填补（ratio 列单独按 promoter 规则处理）
    if impute_missing:
        if str(impute_method).lower() == "kde":
            df, _ = _impute_missing_values_kde(
                df,
                element_cols=element_cols,
                text_cols=text_cols,
                skip_cols=skip_cols,
                type_col_substring=impute_type_substring,
                skip_name_substring=impute_skip_substring,
                random_seed=impute_seed,
                missing_text_token=missing_text_token
            )
        else:
            df = _impute_missing_values_simple(
                df,
                skip_cols=skip_cols,
                missing_text_token=missing_text_token
            )

    df = _apply_promoter_ratio_missing_rules(
        df,
        element_cols=element_cols,
        ratio_col_map=ratio_col_map,
        impute_seed=impute_seed,
        prefer_kde=True
    )

    # 选择Y列
    y_col_names = list(y_cols)

    # 选择X列
    drop_set = set(y_cols) | set(drop_metadata_cols)
    x_cols = [c for c in df.columns if c not in drop_set]

    # 检查列是否存在
    for c in element_cols:
        if c not in df.columns:
            raise KeyError(f"Element column '{c}' not found in CSV.")
    for c in text_cols:
        if c not in df.columns:
            raise KeyError(f"Text column '{c}' not found in CSV.")
    for c in y_cols:
        if c not in df.columns:
            raise KeyError(f"Y column '{c}' not found in CSV.")

    X_df = df[x_cols].copy()
    Y_df = df[y_cols].copy()

    # 处理Y
    Y = Y_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    for i, cname in enumerate(y_cols):
        col = Y[:, i]
        if np.isnan(col).any():
            mean_val = np.nanmean(col)
            if np.isnan(mean_val):
                mean_val = 0.0
            col[np.isnan(col)] = mean_val
            Y[:, i] = col

        if "%" in cname:
            below0 = int((col < 0).sum())
            above100 = int((col > 100).sum())
            if below0 > 0 or above100 > 0:
                print(f"[WARN] Y-col '{cname}' has {below0} <0, {above100} >100 (after NaN fill)")

    special_cols = set(element_cols) | set(text_cols)
    numeric_cols = [c for c in X_df.columns if c not in special_cols]

    # 数值列处理
    for c in numeric_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    if not impute_missing:
        if fill_numeric not in {"median", "zero", "mean"}:
            raise ValueError("fill_numeric must be 'median', 'mean', or 'zero'")

        for c in numeric_cols:
            col = X_df[c]
            if col.isna().all():
                fill_val = 0.0
            else:
                if fill_numeric == "mean":
                    fill_val = float(col.mean())
                elif fill_numeric == "median":
                    fill_val = float(col.median())
                else:
                    fill_val = 0.0
            X_df[c] = col.fillna(fill_val)

    # 可选：对指定数值列做 ln 变换（避免 <=0 取对数的问题）
    if log_transform_cols:
        try:
            eps = float(log_transform_eps) if log_transform_eps is not None else 1e-8
        except (TypeError, ValueError):
            eps = 1e-8
            print(f"[WARN] log_transform_eps='{log_transform_eps}' invalid; fallback to {eps}.")
        for c in log_transform_cols:
            if c not in X_df.columns:
                continue
            col = pd.to_numeric(X_df[c], errors="coerce")
            if col.isna().all():
                continue
            if (col <= 0).any():
                print(f"[WARN] log_transform '{c}': non-positive values found, clamping to {eps}.")
            X_df[c] = np.log(np.clip(col.to_numpy(dtype=np.float32), eps, None))

    # 文本列处理
    for c in text_cols:
        X_df[c] = X_df[c].astype("object").where(~X_df[c].isna(), missing_text_token)

    # 元素列保持为object
    for c in element_cols:
        X_df[c] = X_df[c].astype("object")

    # 创建材料特征器
    element_featurizer: MaterialFeaturizer
    if element_embedding.lower() == "advanced":
        try:
            element_featurizer = AdvancedMaterialFeaturizer()
            print("\n使用 AdvancedMaterialFeaturizer (pymatgen + matminer)...")
        except Exception as e:
            print(f"[WARN] AdvancedMaterialFeaturizer failed: {e}; fallback to Simplified.")
            element_featurizer = SimplifiedMaterialFeaturizer()
    elif element_embedding.lower() in {"simplified", "simple"}:
        element_featurizer = SimplifiedMaterialFeaturizer()
        print("\n使用 SimplifiedMaterialFeaturizer...")
    else:
        element_featurizer = MagpieFeaturizer()

    N = len(X_df)

    # 构建X
    X_parts: List[np.ndarray] = []
    x_col_names: List[str] = []
    numeric_cols_idx: List[int] = []
    cur_idx = 0

    # 数值特征
    if numeric_cols:
        X_num = X_df[numeric_cols].to_numpy(dtype=np.float32)
        X_parts.append(X_num)
        x_col_names.extend(list(numeric_cols))
        numeric_cols_idx.extend(list(range(cur_idx, cur_idx + X_num.shape[1])))
        cur_idx += X_num.shape[1]

    observed_combos: Dict[str, List[str]] = {}
    observed_value_counts: Dict[str, Dict[str, int]] = {}
    observed_value_ratios: Dict[str, Dict[str, float]] = {}
    onehot_groups: List[List[int]] = []
    oh_index_map: List[int] = [-1] * cur_idx
    material_tokens_by_col: Dict[str, np.ndarray] = {}
    base_vecs_by_col: Dict[str, np.ndarray] = {}
    ratio_by_col: Dict[str, np.ndarray] = {}
    null_by_col: Dict[str, np.ndarray] = {}

    # 元素/材料特征
    for c in element_cols:
        materials = X_df[c].tolist()
        ratio_col = ratio_col_map.get(c)
        if ratio_col and ratio_col in X_df.columns:
            ratio_series = pd.to_numeric(X_df[ratio_col], errors="coerce").to_numpy(dtype=np.float32)
            ratio_series = np.where(np.isnan(ratio_series), 0.0, ratio_series)
        else:
            ratio_series = np.ones(len(X_df), dtype=np.float32)

        # 先统计类别，保证 one-hot 维度一致
        counts = Counter()
        for mat in materials:
            if pd.isna(mat):
                continue
            mat_str = str(mat).strip()
            if mat_str and mat_str.lower() not in {"none", "nan"}:
                counts[mat_str] += 1
        values = sorted(counts.keys())
        idx_map = {v: i for i, v in enumerate(values)}
        onehot_eye = np.eye(len(values), dtype=np.float32) if promoter_onehot else None

        # 特征化所有材料
        vecs_list = []
        base_vecs_list = []
        material_tokens = []
        ratio_effective = []
        null_flags = []
        ratio_sum = defaultdict(float)
        ratio_cnt = defaultdict(int)
        ratio_global_sum = 0.0
        ratio_global_cnt = 0
        for mat, ratio_val in zip(materials, ratio_series):
            vec = element_featurizer.featurize(mat)
            base_vecs_list.append(vec)
            mat_str = ""
            if mat is not None and not (isinstance(mat, float) and np.isnan(mat)):
                mat_str = str(mat).strip()
            mat_token = _clean_material_token(mat_str)
            is_null = 1.0 if (isinstance(mat, str) and mat.strip().lower() == "null") else 0.0
            r = float(ratio_val) if ratio_val is not None else 0.0
            if is_null:
                r = 0.0
            material_tokens.append(mat_token)
            ratio_effective.append(r)
            null_flags.append(is_null)
            # Feature = [base magpie | ratio-weighted magpie | is_null | optional one-hot]
            weighted = vec * r
            oh = np.array([], dtype=np.float32)
            if promoter_onehot and onehot_eye is not None:
                idx = idx_map.get(mat_str, None)
                if idx is not None:
                    oh = onehot_eye[idx]
                else:
                    oh = np.zeros(len(values), dtype=np.float32)
            full_vec = np.concatenate([vec, weighted, np.array([is_null], dtype=np.float32), oh])
            vecs_list.append(full_vec)

            if mat_str:
                ratio_sum[mat_str] += r
                ratio_cnt[mat_str] += 1
                ratio_global_sum += r
                ratio_global_cnt += 1

        vecs = np.vstack(vecs_list)
        material_tokens_by_col[c] = np.asarray(material_tokens, dtype=object)
        base_vecs_by_col[c] = np.vstack(base_vecs_list).astype(np.float32)
        ratio_by_col[c] = np.asarray(ratio_effective, dtype=np.float32)
        null_by_col[c] = np.asarray(null_flags, dtype=np.float32)
        start, end = cur_idx, cur_idx + vecs.shape[1]
        X_parts.append(vecs)

        # 列名
        base_dim = element_featurizer.dim
        base_labels = getattr(element_featurizer, "feature_labels", None)
        if not base_labels or len(base_labels) != base_dim:
            base_labels = [f"magpie_{i}" for i in range(base_dim)]
        weighted_labels = [f"{lbl}__ratio" for lbl in base_labels]
        feat_labels = list(base_labels) + weighted_labels + ["is_null"]
        if promoter_onehot:
            feat_labels += [f"onehot__{v}" for v in values]

        x_col_names.extend([f"{c}__{label}" for label in feat_labels])

        gid = len(onehot_groups)
        idxs = list(range(start, end))
        onehot_groups.append(idxs)
        oh_index_map.extend([gid] * (end - start))

        # 记录观察到的值
        observed_combos[c] = values
        observed_value_counts[c] = dict(counts)
        ratio_map = {k: (ratio_sum[k] / ratio_cnt[k]) for k in ratio_sum if ratio_cnt[k] > 0}
        ratio_map["__GLOBAL__"] = (ratio_global_sum / ratio_global_cnt) if ratio_global_cnt > 0 else 0.0
        observed_value_ratios[c] = ratio_map
        cur_idx = end

    # Promoter 间交互特征（绑定 promoter1/promoter2 的类别与 ratio）
    if promoter_interaction_features and len(element_cols) >= 2:
        try:
            inter_eps = float(promoter_interaction_eps)
        except (TypeError, ValueError):
            inter_eps = 1e-8
            print(f"[WARN] promoter_interaction_eps='{promoter_interaction_eps}' invalid; fallback to {inter_eps}.")
        if inter_eps <= 0:
            inter_eps = 1e-8

        for i in range(len(element_cols)):
            for j in range(i + 1, len(element_cols)):
                c_left = element_cols[i]
                c_right = element_cols[j]
                if (
                    c_left not in material_tokens_by_col
                    or c_right not in material_tokens_by_col
                    or c_left not in base_vecs_by_col
                    or c_right not in base_vecs_by_col
                ):
                    continue

                inter_vecs, inter_labels = _build_promoter_pair_interaction_block(
                    left_col=c_left,
                    right_col=c_right,
                    left_tokens=material_tokens_by_col[c_left],
                    right_tokens=material_tokens_by_col[c_right],
                    left_ratio=ratio_by_col[c_left],
                    right_ratio=ratio_by_col[c_right],
                    left_base_vec=base_vecs_by_col[c_left],
                    right_base_vec=base_vecs_by_col[c_right],
                    left_is_null=null_by_col[c_left],
                    right_is_null=null_by_col[c_right],
                    eps=inter_eps,
                    add_pair_onehot=promoter_pair_onehot,
                    pair_onehot_min_count=promoter_pair_onehot_min_count,
                    pair_onehot_max_categories=promoter_pair_onehot_max_categories,
                )
                if inter_vecs.size == 0:
                    continue
                start, end = cur_idx, cur_idx + inter_vecs.shape[1]
                X_parts.append(inter_vecs)
                x_col_names.extend(inter_labels)
                numeric_cols_idx.extend(list(range(start, end)))
                oh_index_map.extend([-1] * (end - start))
                cur_idx = end

    # 打印智能特征器统计
    if hasattr(element_featurizer, "print_stats"):
        element_featurizer.print_stats()

    # 文本特征（改为 one-hot）
    for c in text_cols:
        texts = [str(t) if not pd.isna(t) else missing_text_token for t in X_df[c].tolist()]
        counts = Counter(t.strip() for t in texts if t.strip() != "")
        values = sorted(counts.keys())
        if not values:
            values = [missing_text_token]
            counts = Counter({missing_text_token: len(texts)})

        idx_map = {v: i for i, v in enumerate(values)}
        vecs = np.zeros((len(texts), len(values)), dtype=np.float32)
        for i, t in enumerate(texts):
            key = t.strip()
            if key == "":
                key = missing_text_token
            j = idx_map.get(key, None)
            if j is not None:
                vecs[i, j] = 1.0

        start, end = cur_idx, cur_idx + vecs.shape[1]
        X_parts.append(vecs)
        x_col_names.extend([f"{c}__{v}" for v in values])

        gid = len(onehot_groups)
        idxs = list(range(start, end))
        onehot_groups.append(idxs)
        oh_index_map.extend([gid] * (end - start))

        observed_combos[c] = values
        observed_value_counts[c] = dict(counts)
        cur_idx = end

    # 合并所有特征
    if len(X_parts) > 1:
        X = np.concatenate(X_parts, axis=1)
    else:
        X = X_parts[0]

    assert X.shape[0] == N
    assert len(x_col_names) == X.shape[1]
    assert len(oh_index_map) == X.shape[1]

    if return_dataframe:
        return (X, Y, numeric_cols_idx, x_col_names, y_col_names,
                observed_combos, observed_value_counts, observed_value_ratios,
                onehot_groups, oh_index_map, df)

    return (X, Y, numeric_cols_idx, x_col_names, y_col_names,
            observed_combos, observed_value_counts, observed_value_ratios,
            onehot_groups, oh_index_map)


def load_raw_data_for_correlation(
    csv_path: str,
    input_len: Optional[int] = None,
    output_len: Optional[int] = None,
    drop_nan: bool = True,
    fill_same_as_train: bool = True,
    element_cols: tuple[str, ...] = ("Promoter 1", "Promoter 2"),
    promoter_ratio_cols: Optional[Sequence[str]] = None,
    text_cols: Tuple[str, ...] = ("Type of sysnthesis procedure",),
    drop_metadata_cols: Tuple[str, ...] = ("DOI", "Name", "Year"),
    impute_seed: int = 42,
    impute_type_substring: str = "Type",
    impute_skip_substring: str = "ame",
    missing_text_token: str = "__MISSING__",
    impute_method: str = "simple",
    aggregate_duplicate_inputs: bool = False,
    duplicate_target_agg: str = "median",
    preserve_null: bool = True,
) -> pd.DataFrame:
    """
    返回用于相关性分析的原始数据子集（可选 KDE 缺失值填补）。
    """
    df = _read_csv_with_missing(csv_path, preserve_null=preserve_null)
    df = _strip_colnames(df)
    df = _drop_unnamed_cols(df)

    # 归一化缺失值
    for c in df.columns:
        df[c] = df[c].map(_normalize_missing)

    y_cols = _resolve_y_cols(df, None)
    if aggregate_duplicate_inputs:
        df, _ = _aggregate_duplicate_input_rows(
            df,
            y_cols=y_cols,
            drop_metadata_cols=drop_metadata_cols,
            target_agg=duplicate_target_agg,
        )

    element_cols = tuple(c.strip() for c in element_cols)
    text_cols = tuple(c.strip() for c in text_cols)
    ratio_col_map = _resolve_promoter_ratio_cols(element_cols, promoter_ratio_cols, df)
    ratio_cols = tuple(c for c in ratio_col_map.values() if c and c in df.columns)
    skip_cols = tuple(drop_metadata_cols) + ratio_cols

    if fill_same_as_train:
        if str(impute_method).lower() == "kde":
            df, _ = _impute_missing_values_kde(
                df,
                element_cols=element_cols,
                text_cols=text_cols,
                skip_cols=skip_cols,
                type_col_substring=impute_type_substring,
                skip_name_substring=impute_skip_substring,
                random_seed=impute_seed,
                missing_text_token=missing_text_token
            )
        else:
            df = _impute_missing_values_simple(
                df,
                skip_cols=skip_cols,
                missing_text_token=missing_text_token
            )

    df = _apply_promoter_ratio_missing_rules(
        df,
        element_cols=element_cols,
        ratio_col_map=ratio_col_map,
        impute_seed=impute_seed,
        prefer_kde=True
    )

    if input_len is not None and output_len is not None:
        check_len = input_len + output_len
        df = df.iloc[:, :check_len].copy()

    if drop_nan:
        df.dropna(subset=df.columns, how="any", inplace=True)
    return df


def extract_data_statistics(
    X: np.ndarray,
    x_col_names: Sequence[str],
    numeric_cols_idx: Sequence[int],
    Y: Optional[np.ndarray] = None,
    y_col_names: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """
    提取并返回一个 dict:
      {
        "continuous_cols": { <colname>: { "min":..., "max":..., "mean":... }, ... },
        "onehot_groups": []
      }
    """
    stats = {
        "continuous_cols": {},
        "onehot_groups": []
    }

    for idx in numeric_cols_idx:
        cname = x_col_names[idx]
        col_data = X[:, idx]
        stats["continuous_cols"][cname] = {
            "min": float(np.nanmin(col_data)),
            "max": float(np.nanmax(col_data)),
            "mean": float(np.nanmean(col_data))
        }

    if (Y is not None) and (y_col_names is not None):
        for i, cname in enumerate(y_col_names):
            col_data = Y[:, i]
            stats["continuous_cols"][cname] = {
                "min": float(np.nanmin(col_data)),
                "max": float(np.nanmax(col_data)),
                "mean": float(np.nanmean(col_data))
            }

    return stats


def build_group_value_vectors(
    observed_values: Dict[str, List[str]],
    element_cols: Sequence[str],
    text_cols: Sequence[str],
    element_embedding: str = "advanced",
    observed_value_counts: Optional[Dict[str, Dict[str, int]]] = None,
    observed_value_ratios: Optional[Dict[str, Dict[str, float]]] = None,
    promoter_onehot: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    为每个元素/文本列构建 observed 值的向量表示，供 inference 复用。
    """
    group_vectors: Dict[str, Dict[str, Any]] = {}

    element_featurizer: MaterialFeaturizer | None = None
    if element_cols:
        if element_embedding.lower() == "advanced":
            try:
                element_featurizer = AdvancedMaterialFeaturizer()
            except Exception as e:
                print(f"[WARN] AdvancedMaterialFeaturizer failed: {e}; fallback to Simplified.")
                element_featurizer = SimplifiedMaterialFeaturizer()
        elif element_embedding.lower() in {"simplified", "simple"}:
            element_featurizer = SimplifiedMaterialFeaturizer()
        else:
            element_featurizer = MagpieFeaturizer()

    if element_cols:
        assert element_featurizer is not None
        element_featurizer = cast(MaterialFeaturizer, element_featurizer)

    for col in element_cols:
        values = observed_values.get(col, [])
        if not values:
            continue
        ratio_map = (observed_value_ratios or {}).get(col, {})
        ratio_default = float(ratio_map.get("__GLOBAL__", 0.0))
        onehot_eye = np.eye(len(values), dtype=np.float32) if promoter_onehot else None
        vecs_list = []
        for i, v in enumerate(values):
            vec = element_featurizer.featurize(v)
            is_null = 1.0 if str(v).strip().lower() == "null" else 0.0
            ratio = float(ratio_map.get(v, ratio_default))
            if is_null:
                ratio = 0.0
            weighted = vec * ratio
            oh = np.array([], dtype=np.float32)
            if promoter_onehot and onehot_eye is not None:
                oh = onehot_eye[i]
            full_vec = np.concatenate([vec, weighted, np.array([is_null], dtype=np.float32), oh])
            vecs_list.append(full_vec)
        weights = None
        if observed_value_counts and col in observed_value_counts:
            counts = observed_value_counts[col]
            weights = np.array([counts.get(v, 0) for v in values], dtype=float)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = None
        group_vectors[col] = {
            "values": list(values),
            "vectors": np.vstack(vecs_list),
            "weights": weights.tolist() if weights is not None else None
        }

    for col in text_cols:
        values = observed_values.get(col, [])
        if not values:
            continue
        # Text features are one-hot in training; keep the same here.
        vecs = np.eye(len(values), dtype=np.float32)
        weights = None
        if observed_value_counts and col in observed_value_counts:
            counts = observed_value_counts[col]
            weights = np.array([counts.get(v, 0) for v in values], dtype=float)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = None
        group_vectors[col] = {
            "values": list(values),
            "vectors": vecs,
            "weights": weights.tolist() if weights is not None else None
        }

    return group_vectors


class SimplifiedMaterialFeaturizer:
    """
    简化版材料特征器，避免复杂的pymatgen API问题
    主要使用启发式规则处理
    """
    def __init__(self):
        self.magpie = MagpieFeaturizer()
        self.stats = {
            'single_elements': Counter(),
            'compounds': Counter(),
            'materials': Counter(),
            'text_descriptions': Counter(),
            'unknown': Counter()
        }

    @property
    def dim(self) -> int:
        return self.magpie.dim

    @property
    def feature_labels(self) -> List[str]:
        return self.magpie.feature_labels

    def featurize(self, material: Optional[str]) -> np.ndarray:
        """
        简化版特征化：主要基于启发式规则
        """
        # 处理缺失值
        if material is None or pd.isna(material):
            self.stats['unknown']['<MISSING>'] += 1
            return np.zeros(self.dim, dtype=np.float32)

        material_str = str(material).strip()

        if not material_str or material_str.lower() in {'none', 'nan', ''}:
            self.stats['unknown']['<EMPTY>'] += 1
            return np.zeros(self.dim, dtype=np.float32)

        if material_str.lower() == 'null':
            self.stats['unknown']['<NULL>'] += 1
            return _null_token_vector(self.dim)

        # 转换常见材料类型到元素
        element_map = self._get_element_mapping(material_str)

        if element_map:
            # 如果有映射，使用映射的元素
            feat = self.magpie.featurize(element_map)
            if not np.all(feat == 0):
                self.stats['single_elements'][element_map] += 1
                return feat

        # 尝试直接作为元素处理
        try:
            feat = self.magpie.featurize(material_str)
            if not np.all(feat == 0):
                self.stats['single_elements'][material_str] += 1
                return feat
        except:
            pass

        # 尝试提取可能的元素符号
        element_pattern = r'\b[A-Z][a-z]?\b'
        possible_elements = re.findall(element_pattern, material_str)

        if possible_elements:
            # 取第一个可能的元素
            first_element = possible_elements[0]
            try:
                feat = self.magpie.featurize(first_element)
                if not np.all(feat == 0):
                    self.stats['single_elements'][first_element] += 1
                    return feat
            except:
                pass

        # 作为材料类型处理
        material_lower = material_str.lower()
        if any(mat in material_lower for mat in ['zeolite', 'mof', 'perovskite', 'alumina', 'silica']):
            self.stats['materials'][material_str] += 1
            return self._get_material_features(material_str)

        # 作为化合物处理（简单处理）
        if self._looks_like_compound(material_str):
            self.stats['compounds'][material_str] += 1
            return self._handle_simple_compound(material_str)

        # 最后手段：作为文本描述处理
        self.stats['text_descriptions'][material_str] += 1
        return self._handle_simple_text(material_str)

    def _get_element_mapping(self, material: str) -> Optional[str]:
        """将常见材料类型映射到主要元素"""
        material_lower = material.lower()

        mapping = {
            'alfum': 'Al',  # 假设AlFum是铝的化合物
            'al_fum': 'Al',
            'al fumarate': 'Al',
            'zeolite': 'Si',  # 沸石主要含硅
            'mof': 'Zn',  # MOF常用锌
            'zif': 'Zn',  # ZIF是锌基MOF
            'zsm': 'Si',  # ZSM沸石
            'alumina': 'Al',
            'silica': 'Si',
            'titania': 'Ti',
            'ceria': 'Ce',
            'zirconia': 'Zr'
        }

        for key, element in mapping.items():
            if key in material_lower:
                return element

        return None

    def _looks_like_compound(self, text: str) -> bool:
        """检查是否看起来像化合物"""
        # 包含数字和字母的组合
        if re.search(r'\d', text):
            return True

        # 多个大写字母（可能是多个元素）
        if len(re.findall(r'[A-Z][a-z]?', text)) > 1:
            return True

        return False

    def _handle_simple_compound(self, formula: str) -> np.ndarray:
        """简单处理化合物：提取所有可能的元素，取平均"""
        # 提取所有可能的元素符号
        elements = re.findall(r'[A-Z][a-z]?', formula)

        features = []
        for el in elements:
            try:
                feat = self.magpie.featurize(el)
                if not np.all(feat == 0):
                    features.append(feat)
            except:
                continue

        if features:
            # 简单平均
            avg_feat = np.mean(features, axis=0)
            # 标记为化合物
            if avg_feat.shape[0] > 0:
                avg_feat[-1] = 0.5
            return avg_feat

        # 返回平均元素特征
        return self._get_mean_element_features()

    def _get_mean_element_features(self) -> np.ndarray:
        """获取常见元素的平均特征"""
        common_elements = ['Cu', 'Zn', 'Al', 'Ni', 'Co', 'Fe', 'Mn', 'Cr']
        features = []

        for el in common_elements:
            try:
                feat = self.magpie.featurize(el)
                if not np.all(feat == 0):
                    features.append(feat)
            except:
                continue

        if features:
            return np.mean(features, axis=0)
        else:
            return np.zeros(self.dim, dtype=np.float32)

    def _get_material_features(self, material_type: str) -> np.ndarray:
        """获取材料类型的特征（简化版）"""
        features = np.zeros(self.dim, dtype=np.float32)

        # 设置一些简单的特征
        material_lower = material_type.lower()

        if 'zeolite' in material_lower:
            features[0] = 0.8  # 高表面积
            features[1] = 0.6  # 酸性
        elif 'mof' in material_lower:
            features[0] = 0.9  # 极高表面积
        elif 'alumina' in material_lower:
            features[2] = 0.7  # 氧化铝特征

        return features

    def _handle_simple_text(self, text: str) -> np.ndarray:
        """简单文本处理：返回特殊标记的特征"""
        features = np.ones(self.dim, dtype=np.float32) * 0.1
        if features.shape[0] > 0:
            features[-1] = 0.8  # 文本标志
        return features

    def print_stats(self):
        """打印统计信息"""
        print("\n材料特征化统计:")
        for category, counter in self.stats.items():
            total = sum(counter.values())
            if total > 0:
                print(f"  {category}: {total} 次")
                if list(counter.items()):
                    examples = list(counter.items())[:5]
                    print(f"    示例: {examples}")


# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="智能数据加载器")
    parser.add_argument("--csv", type=str, default="Main_v1.csv", help="输入CSV文件")
    parser.add_argument("--analyze", action="store_true", help="仅分析数据，不加载")
    parser.add_argument("--mode", type=str, default="advanced", choices=["basic", "advanced"],
                       help="特征器模式: basic=原始Magpie, advanced=智能特征器")
    parser.add_argument("--output", type=str, default="processed_data", help="输出目录")
    args = parser.parse_args()

    if args.analyze:
        # 仅分析数据
        print("分析数据中的元素/材料类型...")
        df = pd.read_csv(args.csv)
        df = _strip_colnames(df)

        for col in ["Promoter 1", "Promoter 2"]:
            if col in df.columns:
                print(f"\n{col} 唯一值:")
                values = df[col].dropna().unique()
                for val in values[:20]:  # 显示前20个
                    print(f"  {val}")
    else:
        # 加载数据
        print(f"使用 {args.mode} 模式加载数据...")

        try:
            # 使用简化版本
            from typing import cast
            (X, Y, numeric_cols_idx, x_col_names, y_col_names,
             observed_combos, observed_value_counts, observed_value_ratios,
             onehot_groups, oh_index_map) = cast(tuple, load_smart_data_simple(
                csv_path=args.csv,
                element_cols=("Promoter 1", "Promoter 2"),
                text_cols=("Type of sysnthesis procedure",),
                element_embedding=args.mode,
                return_dataframe=False
            ))

            print(f"\n数据加载完成:")
            print(f"  X 形状: {X.shape}")
            print(f"  Y 形状: {Y.shape}")
            print(f"  特征数量: {len(x_col_names)}")
            print(f"  目标变量: {y_col_names}")

            # 可选：保存数据
            save_data = True  # 默认保存
            if save_data:
                import os
                os.makedirs(args.output, exist_ok=True)

                # 保存X和Y
                X_df = pd.DataFrame(X, columns=x_col_names)
                Y_df = pd.DataFrame(Y, columns=y_col_names)

                X_df.to_csv(f"{args.output}/X_processed.csv", index=False)
                Y_df.to_csv(f"{args.output}/Y_processed.csv", index=False)
                pd.concat([X_df, Y_df], axis=1).to_csv(f"{args.output}/XY_combined.csv", index=False)

                # 保存元数据
                import json
                metadata = {
                    'original_file': args.csv,
                    'mode': args.mode,
                    'X_shape': list(X.shape),
                    'Y_shape': list(Y.shape),
                    'x_feature_names': x_col_names,
                    'y_target_names': y_col_names,
                    'numeric_features_indices': numeric_cols_idx,
                    'observed_categories': observed_combos,
                    'observed_value_counts': observed_value_counts,
                    'observed_value_ratios': observed_value_ratios,
                    'embedding_groups': onehot_groups,
                    'feature_type_mapping': oh_index_map
                }

                with open(f'{args.output}/metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"数据已保存到 {args.output}/ 目录")

        except Exception as e:
            print(f"加载数据时出错: {e}")
            import traceback
            traceback.print_exc()
