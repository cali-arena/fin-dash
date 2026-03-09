from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def run(curated_dir: Path, qa_dir: Path) -> Path:
    qa_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_parquet(curated_dir / "metrics_monthly.parquet")
    unmapped = (
        metrics.loc[metrics["channel"].isna() | metrics["channel"].astype(str).str.strip().eq("")]
        .copy()
    )
    out = qa_dir / "unmapped_channels.csv"
    key_cols = [c for c in ["month_end", "channel", "product_ticker", "src_country", "segment", "sub_segment"] if c in unmapped.columns]
    unmapped[key_cols].drop_duplicates().to_csv(out, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Write unmapped channel combinations report")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--qa-dir", default="qa")
    args = parser.parse_args()
    path = run(Path(args.curated_dir), Path(args.qa_dir))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
