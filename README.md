# leakcheck

**Catch data leakage in a tabular dataset before it flatters your model.** A
library + CLI that scans a pandas DataFrame (or CSV) and flags target leakage,
train/test contamination, and trivial columns - each as a structured,
severity-ranked finding.

## The problem

A model that scores 0.99 offline and collapses in production has usually been
handed the answer during training. The three classic culprits:

- **Target leakage** - a feature that is a copy or downstream function of the
  label (e.g. "is_churned" derived from "cancelled_at"). It predicts the target
  almost perfectly, so cross-validation looks incredible and reality doesn't.
- **Train/test contamination** - the same rows appear in both splits, so the
  test score measures memorization, not generalization.
- **Trivial features** - constant columns (no signal) and ID-like columns
  (unique per row) that leak row identity and never generalize.

leakcheck looks for all three and emits a report you can read, serialize to
JSON, or fail CI on.

## How it works

- **Target leakage:** for every feature, take the larger of (a) its absolute
  Pearson correlation with the numeric-encoded target and (b) its normalized
  mutual information (MI / target entropy for classification). A score near
  1.0 means the feature essentially is the target. MI is deliberately
  distrusted for near-unique discrete columns, where it degenerates to the
  target entropy and would otherwise flag every identifier as a false positive.
- **Contamination:** hash rows over the shared columns and count test rows that
  also occur in train.
- **Trivial features:** constant columns and columns whose values are >=95%
  unique.

Everything is deterministic - mutual-information estimation is seeded - so the
same DataFrame always yields the same report.

## Quickstart

Requires uv (https://docs.astral.sh/uv/). From a fresh clone:

```bash
uv sync
uv run pytest -q          # 21 tests
uv run leakcheck --demo   # scan the built-in synthetic dataset
```

## Usage example

leakcheck ships a deterministic synthetic dataset (make_leaky_dataset) with
one planted leak of each kind: a near-copy of the label (leaky_probe), 30
train rows copied into the test split, a unique customer_id, and a constant
batch_flag. Running the detector against it:

```console
$ uv run leakcheck --demo
leakcheck: 4 finding(s) [critical=1, high=1, medium=1, low=1]
  CRITICAL 30 test row(s) (16.7%) also occur in train -- train/test contamination
           metrics: overlapping_rows=30.0000, overlap_fraction=0.1667, test_rows=180.0000
           fix: Re-split with grouping/deduplication so no record appears in both train and test.
  HIGH     [leaky_probe] feature 'leaky_probe' predicts target 'target' with score=0.978 (likely leakage)
           metrics: leakage_score=0.9778, abs_corr=0.9778, norm_mi=0.9123, raw_mi=0.6321
           fix: Confirm this column is available at prediction time and is not derived from the target; drop it if it is.
  MEDIUM   [customer_id] feature 'customer_id' is ID-like (100% of rows are unique)
           metrics: unique_ratio=1.0000, n_unique=450.0000
           fix: Identifiers leak row identity and don't generalize; drop or hash them out.
  LOW      [batch_flag] feature 'batch_flag' is constant (single value)
           metrics: n_unique=1.0000
           fix: Drop constant columns; they carry no signal.
```

All four planted problems are caught; the genuinely-predictive-but-imperfect
clean_signal column and the pure-noise columns are correctly left alone. Every
number above is reproducible with uv run leakcheck --demo and asserted in
tests/test_end_to_end.py.

### As a library

```python
import pandas as pd
from leakcheck import check

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

report = check(train, target="label", test=test)

print(report.to_text())          # human-readable
print(report.to_json())          # structured
if report.has_blocking():        # any CRITICAL/HIGH finding
    raise SystemExit("leakage detected")
```

### On your own CSVs

```bash
# single frame
leakcheck data.csv --target label

# with a held-out split -> also checks contamination; exit 2 on CRITICAL/HIGH
leakcheck train.csv --target label --test test.csv --json --fail-on-blocking
```

## Development

```bash
uv run ruff check .            # lint  (E,F,I,UP,B,SIM,RUF, line-length 100)
uv run ruff format --check .   # format
uv run mypy src tests          # types (strict)
uv run pytest -q               # tests
```

CI runs all four on every push and PR (.github/workflows/ci.yml).

## What I'd build next

- **Group-aware contamination:** detect near-duplicate (not just exact) rows via
  hashing of normalized/rounded values, and honor a group_id so leakage across
  entities (not just rows) is caught.
- **Time-based leakage:** flag features whose availability post-dates the
  prediction timestamp when an event-time column is supplied.
- **HTML report:** render the JSON report as a shareable page with per-feature
  drill-downs.
- **sklearn hook:** a Pipeline-compatible transformer that raises on blocking
  findings at fit time.

## Maintainer

Lokesh Addanki is a Full Stack Developer with 5 years of experience in finance, insurance, and automotive domains. He specializes in building cloud-native microservices and robust data pipelines. This project is maintained to help developers ensure the integrity of their machine learning datasets.

Contact: codeknight.adloky@gmail.com