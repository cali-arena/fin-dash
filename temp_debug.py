import pandas as pd
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.metrics.shared_payload import normalize_base_frame
from app.data.data_gateway import DataGateway
from app.state import FilterState

root = Path(".")
state = FilterState.from_dict({
    "date_start": "2020-01-01",
    "date_end": "2025-12-31"
})
gateway = DataGateway(root)

print("=== DEBUGGING FILTER OPTIONS ===")

# Load selector_frames as in visualisations.py
selector_frames = {}
for name in ["ticker_monthly", "channel_monthly", "segment_monthly", "geo_monthly", "firm_monthly"]:
    df = gateway.run_query(name, state)
    selector_frames[name] = normalize_base_frame(df)

print("\n1. Checking _FRAME_DIM_COLS mapping:")
_FRAME_DIM_COLS = {
    "channel_group": ("channel_group", "channel", "channel_final", "channel_standard"),
    "sub_channel": ("sub_channel",),
    "country": ("country", "src_country", "geo"),
    "sales_focus": ("sales_focus", "uswa_sales_focus_2020"),
    "sub_segment": ("sub_segment", "segment"),
    "product_ticker": ("product_ticker", "ticker"),
}
print(_FRAME_DIM_COLS)

print("\n2. Checking what columns exist in each normalized frame:")
for name, frame in selector_frames.items():
    if frame.empty:
        print(f"{name}: EMPTY")
        continue
    print(f"\n{name}:")
    for dim in ["channel_group", "sub_channel", "country", "sales_focus", "sub_segment", "product_ticker"]:
        for alias in _FRAME_DIM_COLS.get(dim, (dim,)):
            if alias in frame.columns:
                uniq = frame[alias].dropna().astype(str).str.strip().unique()
                non_empty = [v for v in uniq if v and v.lower() not in ("", "nan", "none", "unassigned", "—")]
                print(f"  {dim} (via {alias}): {len(uniq)} unique, {len(non_empty)} non-empty")
                if len(non_empty) <= 5:
                    print(f"    Values: {non_empty}")
                break

print("\n3. Simulating _opts_from_frames for each dimension:")
def _opts_from_frames(col):
    seen = set()
    for frame in selector_frames.values():
        if frame is None or frame.empty:
            continue
        for alias in _FRAME_DIM_COLS.get(col, (col,)):
            if alias not in frame.columns:
                continue
            for v in frame[alias].dropna().astype(str).str.strip().unique():
                if v and v.lower() not in {"", "unassigned", "—", "nan", "none"}:
                    seen.add(v)
            break
    return sorted(seen)

for dim in ["channel_group", "sub_channel", "country", "sales_focus", "sub_segment", "product_ticker"]:
    opts = _opts_from_frames(dim)
    print(f"{dim}: {len(opts)} options -> {opts[:10]}{'...' if len(opts) > 10 else ''}")

print("\n4. Checking raw parquet files for missing columns:")
import pandas as pd
for name in ["channel_monthly", "ticker_monthly", "segment_monthly", "geo_monthly", "firm_monthly"]:
    path = f"data/agg/{name}.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"\n{name} raw columns: {list(df.columns)}")
        # Check for sales_focus column
        if "sales_focus" in df.columns or "uswa_sales_focus_2020" in df.columns:
            print(f"  Has sales_focus column!")
        else:
            print(f"  NO sales_focus column")

print("\n5. Checking data_raw_normalized.parquet for sales_focus:")
path = "data/curated/data_raw_normalized.parquet"
if os.path.exists(path):
    df = pd.read_parquet(path)
    print(f"Columns: {list(df.columns)}")
    if "uswa_sales_focus_2020" in df.columns:
        uniq = df["uswa_sales_focus_2020"].dropna().astype(str).str.strip().unique()
        print(f"uswa_sales_focus_2020 values: {uniq}")
    else:
        print("NO uswa_sales_focus_2020 column in data_raw_normalized.parquet")