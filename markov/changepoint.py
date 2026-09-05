"""Changepoint detection on each attempt's binary correctness series.

Runs ruptures' PELT algorithm (L2 cost) on the sequence of was_correct
values (0/1) for each clean attempt, in question order, to find points where
a student's accuracy shifts -- e.g. settling in after a slow start, or
fatiguing near the end of a long form.

PELT needs a penalty controlling how many changepoints it's willing to
report. There's no single "correct" value; this uses the standard BIC-style
penalty from the ruptures docs, pen = PEN_SCALE * sigma^2 * log(n), where
sigma^2 is the pooled variance of correctness across all included attempts
(a fixed, global estimate, so short and long attempts are penalized on the
same scale) and n is the attempt's own item count. PEN_SCALE is exposed so
it can be tuned without touching the formula.
"""
import numpy as np
import pandas as pd
import ruptures as rpt

PEN_SCALE = 1.0
MIN_SEGMENT_SIZE = 3


def correctness_sequences(encoded: pd.DataFrame, attempt_ids=None) -> dict:
    """Map attempt_id -> ordered array of was_correct as float 0/1.

    encoded must already be sorted by (attempt_id, question_number), which
    load.load_joined guarantees.
    """
    if attempt_ids is not None:
        encoded = encoded[encoded["attempt_id"].isin(attempt_ids)]
    return {aid: grp["was_correct"].astype(float).to_numpy()
            for aid, grp in encoded.groupby("attempt_id", sort=False)}


def attempt_changepoints(series: np.ndarray, pen: float,
                          min_size: int = MIN_SEGMENT_SIZE) -> list:
    """0-indexed changepoint positions within `series` (each marks the start
    of a new segment). Returns [] if the series is too short to split into
    two segments of at least min_size."""
    n = len(series)
    if n < 2 * min_size:
        return []
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(series)
    bkps = algo.predict(pen=pen)
    return bkps[:-1]  # ruptures appends a terminal breakpoint == len(series)


def changepoint_records(encoded: pd.DataFrame, attempt_ids=None,
                         pen_scale: float = PEN_SCALE,
                         min_size: int = MIN_SEGMENT_SIZE) -> pd.DataFrame:
    """One row per detected changepoint: position, position as a fraction of
    attempt length, and the accuracy shift across the break."""
    sequences = correctness_sequences(encoded, attempt_ids)
    if not sequences:
        return pd.DataFrame(columns=["attempt_id", "position", "position_frac",
                                      "n_items", "accuracy_before",
                                      "accuracy_after", "accuracy_delta"])

    pooled_var = float(np.concatenate(list(sequences.values())).var())

    records = []
    for aid, arr in sequences.items():
        n = len(arr)
        pen = pen_scale * pooled_var * np.log(n)
        bkps = attempt_changepoints(arr, pen=pen, min_size=min_size)
        bounds = [0] + bkps + [n]
        for i, cp in enumerate(bkps):
            before = arr[bounds[i]:bounds[i + 1]]
            after = arr[bounds[i + 1]:bounds[i + 2]]
            records.append({
                "attempt_id": aid,
                "position": cp,
                "position_frac": cp / n,
                "n_items": n,
                "accuracy_before": before.mean(),
                "accuracy_after": after.mean(),
                "accuracy_delta": after.mean() - before.mean(),
            })
    return pd.DataFrame.from_records(records)


def summarize(records: pd.DataFrame) -> dict:
    n_attempts_with_cp = records["attempt_id"].nunique() if len(records) else 0
    return {
        "n_changepoints": len(records),
        "n_attempts_with_changepoint": n_attempts_with_cp,
        "position_frac": records["position_frac"].describe() if len(records) else None,
        "accuracy_delta": records["accuracy_delta"].describe() if len(records) else None,
    }


def item_sequences(encoded: pd.DataFrame, attempt_ids=None) -> dict:
    """Map attempt_id -> per-item was_correct/question_id arrays (in
    question order) plus the attempt's test_id.

    encoded must already be sorted by (attempt_id, question_number), which
    load.load_joined guarantees.
    """
    if attempt_ids is not None:
        encoded = encoded[encoded["attempt_id"].isin(attempt_ids)]
    seqs = {}
    for aid, grp in encoded.groupby("attempt_id", sort=False):
        seqs[aid] = {
            "was_correct": grp["was_correct"].astype(float).to_numpy(),
            "question_id": grp["question_id"].to_numpy(),
            "test_id": grp["test_id"].iloc[0],
        }
    return seqs


def negative_delta_coincident_items(encoded: pd.DataFrame, attempt_ids=None,
                                     pen_scale: float = PEN_SCALE,
                                     min_size: int = MIN_SEGMENT_SIZE) -> pd.DataFrame:
    """One row per negative-delta changepoint (accuracy_delta < 0): the
    single item at index `cp`, i.e. the first item of the post-break
    segment -- the item whose response coincides with the detected
    accuracy drop, not the items leading up to it.

    Reruns the exact PELT segmentation from changepoint_records (same pen
    formula, same attempt_changepoints call) but keeps item identity
    instead of collapsing each changepoint to an accuracy summary. Shared
    by the *_lift.py modules (lda_topic_lift.py, qwen_cluster_lift.py,
    length_lift.py), which each join this item's per-item lookup (topic,
    cluster, length, ...) against a subject-wide baseline.
    """
    seqs = item_sequences(encoded, attempt_ids)
    cols = ["attempt_id", "changepoint_idx", "test_id", "question_id"]
    if not seqs:
        return pd.DataFrame(columns=cols)

    pooled_var = float(np.concatenate([s["was_correct"] for s in seqs.values()]).var())

    rows = []
    for aid, s in seqs.items():
        arr = s["was_correct"]
        n = len(arr)
        pen = pen_scale * pooled_var * np.log(n)
        bkps = attempt_changepoints(arr, pen=pen, min_size=min_size)
        bounds = [0] + bkps + [n]
        for i, cp in enumerate(bkps):
            before = arr[bounds[i]:bounds[i + 1]]
            after = arr[bounds[i + 1]:bounds[i + 2]]
            if after.mean() - before.mean() >= 0:
                continue
            rows.append({"attempt_id": aid, "changepoint_idx": i,
                         "test_id": s["test_id"], "question_id": s["question_id"][cp]})
    return pd.DataFrame.from_records(rows, columns=cols)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from load import load_joined
    from encode import encode_frame
    from markov.thresholds import compute_item_thresholds
    from clean import clean_attempt_ids

    df = load_joined()
    encoded = encode_frame(df, compute_item_thresholds(df))
    clean_ids = clean_attempt_ids(encoded)
    records = changepoint_records(encoded, clean_ids)
    stats = summarize(records)

    print(f"clean attempts: {len(clean_ids):,}")
    print(f"changepoints found: {stats['n_changepoints']:,} "
          f"across {stats['n_attempts_with_changepoint']:,} attempts")
    print("\nposition (fraction of attempt length):")
    print(stats["position_frac"])
    print("\naccuracy delta (after - before):")
    print(stats["accuracy_delta"])
