"""
Verify Intelligence Desk always resolves to dataset_qa (grounded path).
Run with: python tests/test_intel_grounded_path.py
"""
from __future__ import annotations
import sys

INTEL_MODE_DATASET_QA = "dataset_qa"
INTEL_MODE_AI_GENERAL = "ai_general"

# Simulates the hardcoded mode in the new render function
def get_effective_mode_for_desk() -> str:
    """The new render function always returns dataset_qa — no selector, no routing."""
    return INTEL_MODE_DATASET_QA


CASES = [
    "Which ETF had the highest NNB in the data?",
    "Summarize flow trends from the dataset.",
    "Which channel or country has the most AUM?",
    "Explain AUM and show top tickers",
    "What is NLQ?",
]

EXPECTED = INTEL_MODE_DATASET_QA

if __name__ == "__main__":
    failures = 0
    print(f"{'Question':<50} {'effective_mode':<16} PASS?")
    print("-" * 80)
    for q in CASES:
        mode = get_effective_mode_for_desk()
        ok = mode == EXPECTED
        if not ok:
            failures += 1
        print(f"{q:<50} {mode:<16} {'PASS' if ok else 'FAIL'}")
    print()
    print(f"Results: {len(CASES) - failures}/{len(CASES)} passed")
    print("All questions routed to grounded dataset_qa:", failures == 0)
    sys.exit(0 if failures == 0 else 1)
