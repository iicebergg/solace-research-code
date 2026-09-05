"""Per-item rapid-response thresholds (Phase 1 of the provocation analysis).

Replaces the module-constant RAPID_SECONDS with a per-item threshold: 30% of
that item's own median response time, capped at MAX_THRESHOLD_SECONDS. This
is a median-substituted variant of the Wise & Ma (2012) NT10 normative
threshold method -- median instead of mean (to resist inflation from
students who leave the tab open mid-question), and 30% instead of NT10's
10%, based on Phase 3 sensitivity results: see threshold_validation.py,
whose 10/20/30% sweep is what this default now tracks as primary.
"""
import pandas as pd

from features import parse_test_id

PCT_OF_MEDIAN = 0.30
MAX_THRESHOLD_SECONDS = 10.0


def _item_dims(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (test_id, question_id) with subject/level attached."""
    items = df[["test_id", "question_id"]].drop_duplicates()
    test_ids = df["test_id"].drop_duplicates()
    test_dims = test_ids.apply(parse_test_id).apply(pd.Series)
    test_dims["test_id"] = test_ids.values
    return items.merge(test_dims[["test_id", "subject", "level"]], on="test_id")


def compute_item_thresholds(
    df: pd.DataFrame,
    pct: float = PCT_OF_MEDIAN,
    cap_seconds: float = MAX_THRESHOLD_SECONDS,
) -> pd.DataFrame:
    """One row per (test_id, question_id): subject, level, response_count,
    median_response_time, threshold, cap_bound."""
    stats = (
        df.groupby(["test_id", "question_id"])["time_seconds"]
        .agg(response_count="count", median_response_time="median")
        .reset_index()
    )
    table = stats.merge(_item_dims(df), on=["test_id", "question_id"])

    computed = table["median_response_time"] * pct
    table["cap_bound"] = computed > cap_seconds
    table["threshold"] = computed.clip(upper=cap_seconds)

    return table[[
        "test_id", "question_id", "subject", "level", "response_count",
        "median_response_time", "threshold", "cap_bound",
    ]].sort_values(["subject", "level", "test_id", "question_id"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from load import load_joined

    OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
    OUTPUTS.mkdir(exist_ok=True)

    df = load_joined()
    table = compute_item_thresholds(df)
    table.to_csv(OUTPUTS / "rapid_thresholds.csv", index=False)

    print(f"items: {len(table):,}")
    print(f"items with <30 responses: {(table['response_count'] < 30).sum():,} "
          f"({(table['response_count'] < 30).mean():.1%})")
    print(f"cap bound: {table['cap_bound'].sum():,} ({table['cap_bound'].mean():.1%})")
    print("\nthreshold distribution (seconds):")
    print(table["threshold"].describe())
    print(f"\nwrote rapid_thresholds.csv -> {OUTPUTS}")
