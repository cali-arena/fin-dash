"""UI guardrails: ban ad-hoc groupby/merge in Streamlit when STRICT_AGG_ONLY is set."""
from app.guardrails.no_adhoc_agg import ban_adhoc_agg, is_strict_agg_only

__all__ = ["ban_adhoc_agg", "is_strict_agg_only"]
