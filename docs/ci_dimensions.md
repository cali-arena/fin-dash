# CI: Dimension build and determinism tests

Run the dimension pipeline and the dimension determinism tests. No external services required.

## Quick run (local)

```bash
# From project root
python -m pipelines.dimensions.build_dimensions
pytest tests/test_dimensions_determinism.py -q
```

## GitHub Actions snippet

Add a job or step to your workflow (e.g. `.github/workflows/ci.yml`):

```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.12"

- name: Install dependencies
  run: |
    pip install pandas pyarrow pytest

- name: Build dimensions
  run: python -m pipelines.dimensions.build_dimensions

- name: Run dimension determinism tests
  run: pytest tests/test_dimensions_determinism.py -q
```

**Note:** `build_dimensions` reads `curated/fact_monthly.parquet`. In CI you may need to generate the fact first (e.g. run the curate pipeline) or use a checked-in sample. The determinism tests use their own fixture (`tests/fixtures/fact_monthly_sample.parquet`, generated at test setup) and do not require the curated fact.

To run only the determinism tests (no fact_monthly.parquet needed):

```yaml
- name: Run dimension determinism tests
  run: pytest tests/test_dimensions_determinism.py -q
```
