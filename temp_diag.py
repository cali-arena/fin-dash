import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Checking parquet files...")
for name in ["channel_monthly", "ticker_monthly", "segment_monthly", "geo_monthly", "firm_monthly"]:
    path = f"data/agg/{name}.parquet"
    if not os.path.exists(path):
        print(f"{path} not found")
        continue
    df = pd.read_parquet(path)
    print(f"\n--- {name} ---")
    print(f"Rows: {len(df)}")
    print("Columns:", list(df.columns))
    # Look for dimension columns
    dim_candidates = ["channel_l1", "channel_l2", "channel_raw", "channel_standard", "channel", "sub_channel", "country", "src_country", "geo", "segment", "sub_segment", "product_ticker", "ticker", "sales_focus", "uswa_sales_focus_2020"]
    for col in dim_candidates:
        if col in df.columns:
            uniq = df[col].dropna().astype(str).str.strip().unique()
            # Count non-empty strings
            non_empty = [v for v in uniq if v and v.lower() not in ("", "nan", "none", "unassigned", "—")]
            print(f"  {col}: {len(uniq)} unique, non-empty: {len(non_empty)}")
            if len(non_empty) <= 20:
                print(f"    {non_empty}")
            else:
                print(f"    first 20: {non_empty[:20]}")
    # Print a few sample rows
    print("Sample rows (first 2):")
    print(df.head(2).to_string())
    print()

# Also check data/curated/data_raw_normalized.parquet
path = "data/curated/data_raw_normalized.parquet"
if os.path.exists(path):
    df = pd.read_parquet(path)
    print(f"\n--- data_raw_normalized ---")
    print(f"Rows: {len(df)}")
    print("Columns:", list(df.columns))
    for col in ["product_ticker", "channel_raw", "channel_standard", "src_country", "uswa_sales_focus_2020", "sub_segment"]:
        if col in df.columns:
            uniq = df[col].dropna().astype(str).str.strip().unique()
            non_empty = [v for v in uniq if v and v.lower() not in ("", "nan", "none", "unassigned", "—")]
            print(f"  {col}: {len(uniq)} unique, non-empty: {len(non_empty)}")
            if len(non_empty) <= 10:
                print(f"    {non_empty}")
else:
    print(f"\n{path} not found")

# Check selector_frames after normalize_base_frame
from app.metrics.shared_payload import normalize_base_frame
from app.data.data_gateway import DataGateway
from app.state import FilterState
import datetime

state = FilterState.from_dict({
    "date_start": "2020-01-01",
    "date_end": "2025-12-31"
})
gateway = DataGateway(".")
frames = {}
for name in ["ticker_monthly", "channel_monthly", "segment_monthly", "geo_monthly", "firm_monthly"]:
    df = gateway.run_query(name, state)
    frames[name] = normalize_base_frame(df)
    print(f"\n--- {name} normalized ---")
    dfn = frames[name]
    if dfn.empty:
        print("Empty")
        continue
    print("Columns:", list(dfn.columns))
    for col in ["channel_group", "sub_channel", "country", "sales_focus", "sub_segment", "product_ticker"]:
        if col in dfn.columns:
            uniq = dfn[col].dropna().astype(str).str.strip().unique()
            non_empty = [v for v in uniq if v and v.lower() not in ("", "nan", "none", "unassigned", "—")]
            print(f"  {col}: {len(uniq)} unique, non-empty: {len(non_empty)}")
            if len(non_empty) <= 10:
                print(f"    {non_empty}")