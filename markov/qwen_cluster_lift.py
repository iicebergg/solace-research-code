"""Lift of Qwen-embedding question clusters at the exact item that
coincides with a negative-delta changepoint.

For each subject, this finds every changepoint (see changepoint.py) on that
subject's clean attempts where accuracy dropped across the break
(accuracy_delta < 0). ruptures reports a changepoint as the index `cp` where
the new (post-break) segment starts -- see changepoint.py's `before`/`after`
split -- so the single item that coincides with the drop is `after[0]`, the
first item of the new segment, NOT the last item of the old one. This module
takes only that one item per negative changepoint and asks: relative to how
often each Qwen cluster shows up across all of that subject's answered
items, how over- or under-represented is it at that exact spot?

    lift = P(cluster | item is the one coinciding with an accuracy-drop
             changepoint) / P(cluster) [baseline rate across all answered
             items]
"""
from pathlib import Path

import pandas as pd

from load import DATA
from changepoint import PEN_SCALE, MIN_SEGMENT_SIZE, negative_delta_coincident_items

SUBJECTS = ["math", "reading", "science"]
QWEN_CLUSTER_FILE = "{subject}_qwen_questions_with_clusters.csv"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def load_qwen_cluster_lookup(subject: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / QWEN_CLUSTER_FILE.format(subject=subject))
    df = df.rename(columns={"qid": "question_id", "cluster": "qwen_cluster"})
    df = df[["test_id", "question_id", "qwen_cluster"]]
    return df.drop_duplicates(subset=["test_id", "question_id"])


def qwen_cluster_lift(encoded: pd.DataFrame, attempt_ids, subject: str,
                       pen_scale: float = PEN_SCALE, min_size: int = MIN_SEGMENT_SIZE) -> dict:
    """Per-cluster lift for one subject: rate of being the item that
    coincides with a negative-delta changepoint, divided by the cluster's
    baseline rate across all of that subject's answered items (same
    attempt pool)."""
    clusters = load_qwen_cluster_lookup(subject)

    subject_items = encoded[encoded["attempt_id"].isin(attempt_ids)][["test_id", "question_id"]]
    subject_items = subject_items.merge(clusters, on=["test_id", "question_id"], how="left")

    coincident_items = negative_delta_coincident_items(encoded, attempt_ids, pen_scale, min_size)
    n_negative_changepoints = len(coincident_items)
    coincident_items = coincident_items.merge(clusters, on=["test_id", "question_id"], how="left")

    all_clusters = sorted(clusters["qwen_cluster"].unique())
    baseline_rate = subject_items["qwen_cluster"].value_counts(normalize=True).reindex(all_clusters)
    at_cp_rate = coincident_items["qwen_cluster"].value_counts(normalize=True).reindex(all_clusters)
    n_baseline = subject_items["qwen_cluster"].value_counts().reindex(all_clusters, fill_value=0)
    n_at_cp = coincident_items["qwen_cluster"].value_counts().reindex(all_clusters, fill_value=0)

    result = pd.DataFrame({
        "n_baseline_items": n_baseline,
        "baseline_rate": baseline_rate,
        "n_at_negative_cp_items": n_at_cp,
        "at_negative_cp_rate": at_cp_rate,
    })
    result["lift"] = result["at_negative_cp_rate"] / result["baseline_rate"]
    result = result.rename_axis("qwen_cluster").sort_values("lift", ascending=False)

    return {
        "lift": result,
        "n_negative_delta_changepoints": n_negative_changepoints,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from load import load_joined
    from encode import encode_frame
    from thresholds import compute_item_thresholds
    from clean import clean_attempt_ids
    from features import parse_test_id

    df = load_joined()
    encoded = encode_frame(df, compute_item_thresholds(df))
    clean_ids = clean_attempt_ids(encoded)
    attempt_subject = encoded.groupby("attempt_id")["test_id"].first() \
        .apply(lambda t: parse_test_id(t)["subject"])
    grouped_clean = attempt_subject.loc[clean_ids]

    for subject in SUBJECTS:
        subject_clean_ids = grouped_clean[grouped_clean == subject].index
        result = qwen_cluster_lift(encoded, subject_clean_ids, subject)
        print(f"\n=== {subject} ({len(subject_clean_ids):,} clean attempts) ===")
        print(f"negative-delta changepoints: {result['n_negative_delta_changepoints']:,}")
        print(result["lift"].round(3))

        out_dir = OUTPUTS / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        result["lift"].to_csv(out_dir / "qwen_cluster_lift.csv")
        print(f"wrote qwen_cluster_lift.csv -> {out_dir}")
