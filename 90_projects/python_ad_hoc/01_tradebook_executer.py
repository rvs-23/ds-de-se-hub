"""
tradebook_processor.py

A script to process Kite tradebook CSVs.

- Discovers CSVs under configured data directories.
- Filters for EQ/MF tradebook files.
- For each file:
  * Loads and validates schema
  * Casts column types
  * Computes trade values
  * Logs date range, per‑type totals, net investment (BUY - SELL)
  * Identifies symbol with max BUY and max SELL values (with amounts)
"""


from pathlib import Path
import re
import sys

import pandas as pd


# ─── Configuration ─────────────────────────────────────────────────────────────

DATA_DIRS = {
    "sample": Path.cwd() / "data" / "sample_data",
    "compute": Path.cwd() / "data" / "compute_data",
}

# Filename patterns for Kite tradebook files
EQ_PATTERN = re.compile(r"^tradebook-[A-Z]+[0-9]+-EQ\.csv$")
MF_PATTERN = re.compile(r"^tradebook-[A-Z]+[0-9]+-MF\.csv$")

# Schema: column → caster (pd.to_datetime for dates)
SCHEMA = {
    "symbol": str,
    "trade_date": pd.to_datetime,
    "trade_type": str,
    "quantity": float,
    "segment": str,
    "price": float,
}

# ─── Functions ─────────────────────────────────────────────────────────────────

def load_csv_paths(source: str) -> list[Path]:
    """
    Return all CSV file paths for a given source key.
    """
    base = DATA_DIRS.get(source)
    if base is None:
        print(f"ERROR: Unknown data source '{source}'\n")
        return []
    return list(base.glob("**/*.csv"))


def filter_tradebooks(paths: list[Path]) -> list[Path]:
    """
    Filter for filenames matching Kite tradebook EQ or MF patterns.
    """
    return [p for p in paths if EQ_PATTERN.match(p.name) or MF_PATTERN.match(p.name)]


def process_tradebook(path: Path) -> pd.DataFrame | None:
    """
    Process one tradebook CSV file:
      - Load and enforce schema
      - Compute trade_value
      - Print detailed summary
    Returns processed DataFrame or None on error.
    """
    print(f"\n==== Processing: {path.name} ====\n")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: Failed to read '{path.name}': {e}\n")
        return None

    print(f"\tRows loaded: {len(df)}")

    # Validate expected columns
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        print(f"\tERROR: Missing columns in '{path.name}': {missing}\n")
        return None

    # Select and cast columns
    df = df[list(SCHEMA.keys())]
    cast_map = {col: typ for col, typ in SCHEMA.items() if typ is not pd.to_datetime}
    df = df.astype(cast_map)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # Compute trade_value
    df["trade_value"] = df["quantity"] * df["price"]

    # Print date range
    start_date = df["trade_date"].min().date()
    end_date = df["trade_date"].max().date()
    print(f"\tTrade dates: {start_date} to {end_date}")

    # Summarize by trade_type
    summary = df.groupby("trade_type")["trade_value"].sum()
    buy_total = summary.get("buy", 0.0)
    sell_total = summary.get("sell", 0.0)
    net = buy_total - sell_total

    print("\n\tInvestment Summary")
    print(f"\t\tBUY total : ₹{buy_total:,.2f}")
    print(f"\t\tSELL total: ₹{sell_total:,.2f}")
    print(f"\n\t\tNET (BUY-SELL): ₹{net:,.2f}")

    # Identify symbol with max cumulative buy value
    buy_summary = df[df["trade_type"] == "buy"].groupby("symbol")["trade_value"].sum()
    if not buy_summary.empty:
        top_buy_sym = buy_summary.idxmax()
        top_buy_val = buy_summary.max()
        print(f"\n\tTop BUY symbol : {top_buy_sym} (₹{top_buy_val:,.2f})")

    # Identify symbol with max cumulative sell value
    sell_summary = df[df["trade_type"] == "sell"].groupby("symbol")["trade_value"].sum()
    if not sell_summary.empty:
        top_sell_sym = sell_summary.idxmax()
        top_sell_val = sell_summary.max()
        print(f"\tTop SELL symbol: {top_sell_sym} (₹{top_sell_val:,.2f})")

    print(f"\n==== Completed: {path.name} ====\n")

    return df


if __name__ == "__main__":
    # Read folder argument ("sample" or "compute")
    if len(sys.argv) < 2:
        print("Usage: python tradebook_processor.py [sample|compute]\n")
        sys.exit(1)

    folder_key = sys.argv[1].strip().lower()
    if folder_key not in DATA_DIRS:
        print(
            f"ERROR: Invalid folder name '{folder_key}'. Must be 'sample' or 'compute'.\n"
        )
        sys.exit(1)

    all_paths = load_csv_paths(folder_key)
    valid_paths = filter_tradebooks(all_paths)
    print(f"Found {len(valid_paths)} tradebook files to process in '{folder_key}'.\n")

    for p in valid_paths:
        process_tradebook(p)
