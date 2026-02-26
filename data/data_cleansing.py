import argparse
import os
from pathlib import Path

import pandas as pd


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c for c in df.columns
        if isinstance(c, str) and c.strip().lower().startswith("unnamed")
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def cleanse_csv(input_path: str, output_path: str) -> tuple[int, int]:
    df = pd.read_csv(input_path, keep_default_na=False)
    before_rows = int(df.shape[0])

    # Normalize column names and drop accidental trailing unnamed columns.
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df = _drop_unnamed_columns(df)

    # Keep all rows except fully empty ones.
    df = df.dropna(how="all").reset_index(drop=True)
    after_rows = int(df.shape[0])

    df.to_csv(output_path, index=False)
    return before_rows, after_rows


def _resolve_default_input(script_dir: Path) -> str:
    candidates = [
        script_dir / "Ea_20260226.csv",
        script_dir / "Main_20260128.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "No default input CSV found. Please provide --input explicitly."
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Cleanse dataset columns and save <name>_cleansed.csv."
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
    args = parser.parse_args()

    input_path = args.input if args.input else _resolve_default_input(script_dir)
    input_path = str(Path(input_path).resolve())
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    default_output = os.path.join(os.path.dirname(input_path), f"{base_name}_cleansed.csv")
    output_path = str(Path(args.output).resolve()) if args.output else default_output

    rows_before, rows_after = cleanse_csv(input_path, output_path)
    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Rows  : {rows_before} -> {rows_after}")
    print("[INFO] Conflict-row checking is disabled by design for this cleanser.")


if __name__ == "__main__":
    main()
