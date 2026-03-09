"""
Debug page: ad-hoc groupby/merge allowed when STRICT_AGG_ONLY is off.
This page does NOT apply the ban_adhoc_agg guardrail — use for exploration only.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Debug (adhoc)", layout="wide")
st.title("Debug: Ad-hoc aggregation")

st.info(
    "This page does not apply the STRICT_AGG_ONLY guardrail. "
    "You may use groupby/merge here for exploration when STRICT_AGG_ONLY is off. "
    "Production views must use agg tables only."
)

# Optional: small demo that uses groupby so users see it's allowed here
import pandas as pd
demo = pd.DataFrame({"x": [1, 1, 2], "y": [10, 20, 30]})
try:
    agg = demo.groupby("x", as_index=False).sum()
    st.write("Sample groupby (allowed on this page):", agg)
except RuntimeError as e:
    st.warning(f"groupby raised (STRICT_AGG_ONLY may be on in this run): {e}")
