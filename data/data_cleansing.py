import argparse
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd


NULL_TOKEN = "Null"
PROMOTER_COL_1 = "Promoter 1"
PROMOTER_COL_2 = "Promoter 2"
RATIO_COL_1 = "Promoter 1 ratio (Promoter 1:Cu)"
RATIO_COL_2 = "Promoter 2 ratio (Promoter 2:Cu)"


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c for c in df.columns
        if isinstance(c, str) and c.strip().lower().startswith("unnamed")
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _normalize_promoter_token(value) -> str:
    if pd.isna(value):
        return NULL_TOKEN
    text = str(value).strip()
    if not text:
        return NULL_TOKEN
    if text.lower() in {"null", "none", "nan", "na", "n/a"}:
        return NULL_TOKEN
    return text


def _is_real_promoter(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in {"", "null", "none", "nan", "na", "n/a"}


def _build_edges(df: pd.DataFrame, col1: str, col2: str) -> List[Tuple[str, str, int]]:
    edges: List[Tuple[str, str, int]] = []
    for idx, row in df.iterrows():
        p1 = row[col1]
        p2 = row[col2]
        if _is_real_promoter(p1) and _is_real_promoter(p2) and str(p1).strip() != str(p2).strip():
            edges.append((str(p1).strip(), str(p2).strip(), int(idx)))
    return edges


def _find_conflicts(edges: Sequence[Tuple[str, str, int]]) -> Tuple[Set[int], Dict[str, int]]:
    graph: Dict[str, List[str]] = defaultdict(list)
    pair_to_row_ids: Dict[frozenset, List[int]] = defaultdict(list)

    for u, v, row_id in edges:
        graph[u].append(v)
        graph[v].append(u)
        pair_to_row_ids[frozenset((u, v))].append(row_id)

    color: Dict[str, int] = {}
    conflicts: Set[int] = set()

    for node in graph:
        if node in color:
            continue
        queue = deque([node])
        color[node] = 0

        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    conflicts.update(pair_to_row_ids.get(frozenset((u, v)), []))

    return conflicts, color


def _resolve_odd_cycle_conflicts(
    edges: Sequence[Tuple[str, str, int]],
    max_iter: int = 100,
) -> Set[int]:
    remaining_edges = list(edges)
    conflict_indices: Set[int] = set()

    for _ in range(max_iter):
        conflicts, _ = _find_conflicts(remaining_edges)
        if not conflicts:
            break
        conflict_indices.update(conflicts)
        remaining_edges = [e for e in remaining_edges if e[2] not in conflicts]

    return conflict_indices


def _build_final_color_map(df_clean: pd.DataFrame, col1: str, col2: str) -> Dict[str, int]:
    final_graph: Dict[str, List[str]] = defaultdict(list)

    for _, row in df_clean.iterrows():
        p1 = str(row[col1]).strip()
        p2 = str(row[col2]).strip()

        if _is_real_promoter(p1):
            final_graph[p1]
        if _is_real_promoter(p2):
            final_graph[p2]
        if _is_real_promoter(p1) and _is_real_promoter(p2) and p1 != p2:
            final_graph[p1].append(p2)
            final_graph[p2].append(p1)

    color: Dict[str, int] = {}
    for prom in final_graph:
        if prom in color:
            continue
        color[prom] = 0
        queue = deque([prom])
        while queue:
            u = queue.popleft()
            for v in final_graph[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)

    return color


def _apply_orientation(
    df_clean: pd.DataFrame,
    col1: str,
    col2: str,
    ratio1: str,
    ratio2: str,
    color: Dict[str, int],
) -> None:
    has_ratio1 = ratio1 in df_clean.columns
    has_ratio2 = ratio2 in df_clean.columns

    for idx, row in df_clean.iterrows():
        p1 = _normalize_promoter_token(row[col1])
        p2 = _normalize_promoter_token(row[col2])

        r1 = row[ratio1] if has_ratio1 else None
        r2 = row[ratio2] if has_ratio2 else None

        if _is_real_promoter(p1) and _is_real_promoter(p2):
            if color.get(p1, 0) == 1:
                df_clean.at[idx, col1] = p2
                df_clean.at[idx, col2] = p1
                if has_ratio1 and has_ratio2:
                    df_clean.at[idx, ratio1] = r2
                    df_clean.at[idx, ratio2] = r1

        elif p1 == NULL_TOKEN and _is_real_promoter(p2):
            if color.get(p2, 0) == 0:
                df_clean.at[idx, col1] = p2
                df_clean.at[idx, col2] = NULL_TOKEN
                if has_ratio1 and has_ratio2:
                    df_clean.at[idx, ratio1] = r2
                    df_clean.at[idx, ratio2] = r1

        elif p2 == NULL_TOKEN and _is_real_promoter(p1):
            if color.get(p1, 0) == 1:
                df_clean.at[idx, col1] = NULL_TOKEN
                df_clean.at[idx, col2] = p1
                if has_ratio1 and has_ratio2:
                    df_clean.at[idx, ratio1] = r2
                    df_clean.at[idx, ratio2] = r1


def _resolve_default_input(script_dir: Path) -> str:
    candidates = [
        script_dir / "Ea_20260226.csv",
        script_dir / "Main_20260128.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("No default input CSV found. Please provide --input explicitly.")


def cleanse_csv(
    input_path: str,
    output_path: str,
    conflict_output_path: Optional[str] = None,
    enable_odd_cycle: bool = True,
) -> Dict[str, object]:
    df = pd.read_csv(input_path, keep_default_na=False)
    before_rows = int(df.shape[0])

    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df = _drop_unnamed_columns(df)
    df = df.dropna(how="all").reset_index(drop=True)

    col1 = PROMOTER_COL_1
    col2 = PROMOTER_COL_2
    ratio1 = RATIO_COL_1
    ratio2 = RATIO_COL_2

    conflict_rows_saved = 0
    conflict_indices: Set[int] = set()
    overlap_after: Set[str] = set()

    has_promoter_cols = col1 in df.columns and col2 in df.columns
    if enable_odd_cycle and has_promoter_cols:
        df[col1] = df[col1].map(_normalize_promoter_token)
        df[col2] = df[col2].map(_normalize_promoter_token)
        df["row_id"] = df.index

        edges = _build_edges(df, col1, col2)
        conflict_indices = _resolve_odd_cycle_conflicts(edges)

        df_conflict = df[df["row_id"].isin(conflict_indices)].copy()
        df_conflict = df_conflict.drop(columns=["row_id"], errors="ignore")
        conflict_rows_saved = int(df_conflict.shape[0])

        if conflict_output_path:
            df_conflict.to_csv(conflict_output_path, index=False)

        df_clean = df[~df["row_id"].isin(conflict_indices)].copy()
        df_clean.drop(columns=["row_id"], inplace=True, errors="ignore")

        color = _build_final_color_map(df_clean, col1, col2)
        _apply_orientation(df_clean, col1, col2, ratio1, ratio2, color)

        p1_set = {str(v).strip() for v in df_clean[col1] if _is_real_promoter(v)}
        p2_set = {str(v).strip() for v in df_clean[col2] if _is_real_promoter(v)}
        overlap_after = p1_set & p2_set

        df_out = df_clean
    else:
        if has_promoter_cols:
            df[col1] = df[col1].map(_normalize_promoter_token)
            df[col2] = df[col2].map(_normalize_promoter_token)
        df_out = df

    after_rows = int(df_out.shape[0])
    df_out.to_csv(output_path, index=False)

    return {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "removed_rows": int(before_rows - after_rows),
        "conflict_rows_saved": conflict_rows_saved,
        "overlap_after": sorted(overlap_after),
        "odd_cycle_enabled": bool(enable_odd_cycle and has_promoter_cols),
        "has_promoter_cols": has_promoter_cols,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Cleanse dataset and optionally resolve odd-cycle conflicts in unordered promoter pairs."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV path. Default: Ea_20260226.csv if exists, else Main_20260128.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: <input_stem>_cleansed.csv in the same folder.",
    )
    parser.add_argument(
        "--conflict-output",
        type=str,
        default=None,
        help="Conflict rows output path. Default: <input_stem>_conflict_rows.csv in the same folder.",
    )
    parser.add_argument(
        "--disable-odd-cycle",
        action="store_true",
        help="Disable odd-cycle conflict resolving and promoter orientation normalization.",
    )
    args = parser.parse_args()

    input_path = args.input if args.input else _resolve_default_input(script_dir)
    input_path = str(Path(input_path).resolve())

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    base_dir = os.path.dirname(input_path)

    default_output = os.path.join(base_dir, f"{base_name}_cleansed.csv")
    output_path = str(Path(args.output).resolve()) if args.output else default_output

    default_conflict_output = os.path.join(base_dir, f"{base_name}_conflict_rows.csv")
    conflict_output_path = (
        str(Path(args.conflict_output).resolve()) if args.conflict_output else default_conflict_output
    )

    stats = cleanse_csv(
        input_path=input_path,
        output_path=output_path,
        conflict_output_path=conflict_output_path,
        enable_odd_cycle=not args.disable_odd_cycle,
    )

    print(f"[INFO] Input               : {input_path}")
    print(f"[INFO] Output              : {output_path}")
    print(f"[INFO] Conflict rows file  : {conflict_output_path}")
    print(f"[INFO] Rows                : {stats['before_rows']} -> {stats['after_rows']}")
    print(f"[INFO] Conflict rows saved : {stats['conflict_rows_saved']}")
    print(f"[INFO] Removed rows        : {stats['removed_rows']}")

    if stats["odd_cycle_enabled"]:
        overlap = stats["overlap_after"]
        if overlap:
            print(f"[WARN] Promoter overlap remains after resolve: {overlap}")
        else:
            print("[INFO] Promoter 1 and Promoter 2 are mutually exclusive after resolve.")
    else:
        if not stats["has_promoter_cols"]:
            print("[WARN] Promoter columns not found; odd-cycle resolve skipped.")
        else:
            print("[INFO] Odd-cycle resolve disabled by flag.")


if __name__ == "__main__":
    main()
