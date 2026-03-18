"""
Standalone test for _classify_intelligence_question and _autoroute_intel_mode.
Run with: python tests/test_intel_router.py
"""
from __future__ import annotations
import re, sys

DATASET_KEYWORDS = [
    "nnb", "aum", "flow", "flows", "ticker", "tickers", "etf",
    "inflow", "inflows", "outflow", "outflows", "asset", "fund", "funds",
    "month", "channel", "country", "region", "exposure",
    "segment", "sub-segment", "sub segment", "sub_segment",
    "sales focus", "sales_focus", "product", "portfolio",
    "ranking", "top", "bottom", "highest", "lowest", "compare", "distribution",
    "trend", "growth", "performance", "return", "risk", "contributor",
    "net new", "organic growth", "market impact", "fee yield",
]
DATASET_ANALYTICS_VERBS = [
    "show", "list", "rank", "compare", "find", "identify",
    "calculate", "summarize", "break down", "breakdown",
]
GENERAL_QUESTION_INDICATORS = [
    "what is", "what's", "what does", "explain", "define", "meaning of",
    "how does", "why is", "overview of", "tell me about",
    "describe", "elaborate on", "clarify",
]
INTEL_MODE_DATASET_QA = "dataset_qa"
INTEL_MODE_AI_GENERAL = "ai_general"


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _classify_intelligence_question(question: str) -> str:
    q = _normalize_text(question)
    asks_from_data = any(phrase in q for phrase in (
        "from the data", "in the dataset", "based on the data", "in the data",
        "our data", "our dataset",
    ))
    has_dataset_ref = any(keyword in q for keyword in DATASET_KEYWORDS)
    has_analytics_verb = any(verb in q for verb in DATASET_ANALYTICS_VERBS)
    has_any_dataset_signal = has_dataset_ref or has_analytics_verb or asks_from_data
    if not has_any_dataset_signal:
        return "general"
    is_explanatory = any(indicator in q for indicator in GENERAL_QUESTION_INDICATORS)
    if asks_from_data:
        return "dataset"
    if has_analytics_verb:
        return "dataset" if not is_explanatory else "hybrid"
    if has_dataset_ref and not is_explanatory:
        return "dataset"
    if has_dataset_ref and is_explanatory:
        return "hybrid"
    return "general"


def _autoroute_intel_mode(question: str, selected_mode: str) -> tuple[str, str, list[str]]:
    q = _normalize_text(question)
    dataset_terms_found = [k for k in DATASET_KEYWORDS if k in q] + [
        v for v in DATASET_ANALYTICS_VERBS if v in q
    ]
    q_class = _classify_intelligence_question(question)
    if selected_mode != INTEL_MODE_AI_GENERAL:
        return selected_mode, q_class, dataset_terms_found
    if q_class in ("dataset", "hybrid"):
        return INTEL_MODE_DATASET_QA, q_class, dataset_terms_found
    return INTEL_MODE_AI_GENERAL, q_class, dataset_terms_found


CASES = [
    # (question, selected_mode, expected_effective_mode)
    ("What is NLQ?",                         INTEL_MODE_AI_GENERAL, INTEL_MODE_AI_GENERAL),
    ("Show AUM by country",                  INTEL_MODE_AI_GENERAL, INTEL_MODE_DATASET_QA),
    ("Explain AUM and show top tickers",     INTEL_MODE_AI_GENERAL, INTEL_MODE_DATASET_QA),
    ("Compare flows across channels",        INTEL_MODE_AI_GENERAL, INTEL_MODE_DATASET_QA),
    ("Which country has the highest AUM?",   INTEL_MODE_AI_GENERAL, INTEL_MODE_DATASET_QA),
    # dataset_qa button selected — must NEVER downgrade
    ("What is NLQ?",                         INTEL_MODE_DATASET_QA, INTEL_MODE_DATASET_QA),
    ("Explain AUM in simple terms",          INTEL_MODE_AI_GENERAL, INTEL_MODE_DATASET_QA),
    ("List top tickers by NNB",              INTEL_MODE_AI_GENERAL, INTEL_MODE_DATASET_QA),
]


if __name__ == "__main__":
    hdr = f"{'Question':<48} {'selected':<14} {'cls':<9} {'effective':<15} {'expected':<15} PASS?"
    print(hdr)
    print("-" * len(hdr))
    failures = 0
    for q, sel, expected in CASES:
        effective, cls, terms = _autoroute_intel_mode(q, sel)
        ok = effective == expected
        if not ok:
            failures += 1
        status = "PASS" if ok else f"FAIL  (got {effective})"
        print(f"{q:<48} {sel:<14} {cls:<9} {effective:<15} {expected:<15} {status}")
        if terms:
            print(f"  terms={terms}")
    print()
    print(f"Results: {len(CASES) - failures}/{len(CASES)} passed")
    sys.exit(0 if failures == 0 else 1)
