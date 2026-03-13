import pandas as pd
from pathlib import Path
from app.data.data_gateway import load_dim_lookup, build_dim_lookup_from_frames
from app.metrics.shared_payload import normalize_base_frame
from app.data.data_gateway import DataGateway
from app.state import FilterState

root = Path(".")
print("Loading dim_lookup...")
dl = load_dim_lookup(root)
print(f"dim_lookup shape: {dl.shape}")
print(f"Columns: {list(dl.columns)}")
if not dl.empty:
    for col in dl.columns:
        uniq = dl[col].dropna().astype(str).str.strip().unique()
        non_empty = [v for v in uniq if v and v.lower() not in ("", "nan", "none", "unassigned", "—")]
        print(f"  {col}: {len(uniq)} unique, non-empty: {len(non_empty)}")
        if len(non_empty) <= 10:
            print(f"    {non_empty}")
        else:
            print(f"    first 10: {non_empty[:10]}")
else:
    print("dim_lookup empty")

# Build selector_frames as in visualisations.py
state = FilterState.from_dict({
    "date_start": "2020-01-01",
    "date_end": "2025-12-31"
})
gateway = DataGateway(root)
selector_frames = {}
for name in ["ticker_monthly", "channel_monthly", "segment_monthly", "geo_monthly", "firm_monthly"]:
    df = gateway.run_query(name, state)
    selector_frames[name] = normalize_base_frame(df)
    print(f"\n{name} normalized columns: {list(selector_frames[name].columns)}")
    for col in ["channel_group", "sub_channel", "country", "sales_focus", "sub_segment", "product_ticker"]:
        if col in selector_frames[name].columns:
            uniq = selector_frames[name][col].dropna().astype(str).str.strip().unique()
            non_empty = [v for v in uniq if v and v.lower() not in ("", "nan", "none", "unassigned", "—")]
            print(f"  {col}: {len(uniq)} unique, non-empty: {len(non_empty)}")
            if len(non_empty) <= 5:
                print(f"    {non_empty}")

# Build dim_lookup from frames
print("\nBuilding dim_lookup from frames...")
dl2 = build_dim_lookup_from_frames(selector_frames)
print(f"dim_lookup from frames shape: {dl2.shape}")
if not dl2.empty:
    for col in dl2.columns:
        uniq = dl2[col].dropna().astype(str).str.strip().unique()
        non_empty = [v for v in uniq if v and v.lower() not in ("", "nan", "none", "unassigned", "—")]
        print(f"  {col}: {len(uniq)} unique, non-empty: {len(non_empty)}")
        if len(non_empty) <= 10:
            print(f"    {non_empty}")
else:
    print("dim_lookup from frames empty")