"""Person-level classification of test attempts: clean / replay / rushed_guess.

Three per-attempt features are computed from the five-state encoding (see
encode.py):
    median_mc_item_time  median time_seconds on multiple-choice items
                          (NaN when the attempt has no MC items -- true for
                          about 22% of attempts, mostly all-free-response or
                          all-technology-enhanced science forms)
    rapid_share           fraction of items encoded R (<= RAPID_SECONDS)
    mc_revision_share      V count / MC item count (NaN when no MC items; V
                          can only ever occur on MC items, see encode.py)

"Fast" is a separate, attempt-level pace measure -- total_time_seconds /
item count -- compared within the same test_id, since forms vary hugely in
expected length and a raw cross-test cutoff would just sort by form length.
An attempt is fast if its pace falls at or below the FAST_QUANTILE of pace
for its own test_id.

    replay        fast and score >= REPLAY_SCORE_MIN
                  (finished quickly, but nearly everything right -- looks
                  like the student already knew the material or the test)
    rushed_guess  fast and score < RUSHED_SCORE_MAX
                  (finished quickly with a low score -- looks like clicking
                  through without engaging)
    clean         everything else, including fast attempts with a score in
                  [RUSHED_SCORE_MAX, REPLAY_SCORE_MIN) -- fast alone isn't
                  suspicious without an extreme score to go with it
"""
import pandas as pd

FAST_QUANTILE = 0.10 # Fastest % of test-takers
REPLAY_SCORE_MIN = 0.9
RUSHED_SCORE_MAX = 0.6


def attempt_features(encoded: pd.DataFrame) -> pd.DataFrame:
    """One row per attempt_id with pace, score, and the three Build-Order
    features. `encoded` is the output of encode.encode_frame on load.load_joined."""
    g = encoded.groupby("attempt_id")

    feats = g.agg(
        test_id=("test_id", "first"),
        total_time_seconds=("total_time_seconds", "first"),
        score_correct=("score_correct", "first"),
        score_total=("score_total", "first"),
        n_items=("state", "size"),
        rapid_share=("state", lambda s: (s == "R").mean()),
        n_v=("state", lambda s: (s == "V").sum()),
    )

    is_mc = encoded["question_type"] == "multiple-choice"
    mc = encoded.loc[is_mc]
    feats["n_mc_items"] = mc.groupby("attempt_id").size().reindex(feats.index, fill_value=0)
    feats["median_mc_item_time"] = mc.groupby("attempt_id")["time_seconds"].median()

    feats["mc_revision_share"] = feats["n_v"] / feats["n_mc_items"].replace(0, pd.NA)
    feats["score"] = feats["score_correct"] / feats["score_total"]
    feats["pace_seconds_per_item"] = feats["total_time_seconds"] / feats["n_items"]

    return feats.drop(columns=["n_v"])


def classify_attempts(
    features: pd.DataFrame,
    fast_quantile: float = FAST_QUANTILE,
    replay_score_min: float = REPLAY_SCORE_MIN,
    rushed_score_max: float = RUSHED_SCORE_MAX,
) -> pd.Series:
    """Label each attempt_id 'replay', 'rushed_guess', or 'clean'."""
    pace_threshold = features.groupby("test_id")["pace_seconds_per_item"].transform(
        lambda s: s.quantile(fast_quantile)
    )
    is_fast = features["pace_seconds_per_item"] <= pace_threshold

    label = pd.Series("clean", index=features.index, name="label")
    label[is_fast & (features["score"] >= replay_score_min)] = "replay"
    label[is_fast & (features["score"] < rushed_score_max)] = "rushed_guess"
    return label


def clean_attempt_ids(encoded: pd.DataFrame) -> pd.Index:
    """Attempt ids classified as neither replay nor rushed_guess."""
    features = attempt_features(encoded)
    labels = classify_attempts(features)
    return labels[labels == "clean"].index


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from load import load_joined
    from encode import encode_frame
    from markov.thresholds import compute_item_thresholds

    df = load_joined()
    encoded = encode_frame(df, compute_item_thresholds(df))
    features = attempt_features(encoded)
    labels = classify_attempts(features)
    print(f"attempts: {len(labels):,}")
    print(labels.value_counts())
    print(f"clean fraction: {(labels == 'clean').mean():.1%}")
