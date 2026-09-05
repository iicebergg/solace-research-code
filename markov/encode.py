"""Encode each item response into one of six behavioral states.

This is scoped to multiple-choice only, exactly like the heavy-revision
state V, because answer_changes doesn't carry the same meaning elsewhere:
free-response items are ALWAYS 0 (100% of the export, by construction of
that item type), while drag-drop/multiple-select/point-select regularly
reach nonzero values. So "0 changes" only means "never answered" on MC;
elsewhere it's either uninformative (free-response) or just a normal low
value (the technology-enhanced types).

The blank state is further split by speed: BF (blank fast) is a skip -- the
student never engaged with the item at all -- while BS (blank slow) is
abandonment -- the student worked the item (or at least sat with it open)
and then quit without picking anything. Both still require is_mc and
answer_changes == 0; only the rapid_threshold comparison distinguishes them.

The six states:
    C  correct, engaged
    W  wrong, engaged
    V  heavy revision (multiple-choice items only)
    R  rapid (<= the item's rapid_threshold, see thresholds.py)
    BF blank fast -- skip (multiple-choice only, answer_changes == 0, <= rapid_threshold)
    BS blank slow -- abandonment (multiple-choice only, answer_changes == 0, > rapid_threshold)

The blank check is done before the rapid check: a blank response can still
take a long time (the student stared at the item and never picked anything
-- observed up to the 99th percentile around 200+ seconds), so a blank must
not be swallowed by the rapid check just because it also happened to be
fast. Instead the rapid_threshold comparison happens inside the blank
branch itself, to decide BF vs BS.

The heavy-revision state (V) fires ONLY on multiple-choice items. On a
free-response or technology-enhanced item, answer_changes counts something
other than choice-switching, so it must not be read as the same uncertainty
signal. Unknown item type is treated as non-multiple-choice (conservative):
such items can never be V, BF, or BS.

Non-multiple-choice items can still be C, W, or R; they are never dropped
from the dataset or the sequences, they just can't emit BF, BS, or V -- a
known and accepted limitation of the encoding, not a bug.
"""
import pandas as pd

MC_TYPES = {"multiple-choice", "multiple_choice", "mc", "single_select", "multiplechoice"}

REVISION_MIN = 3


def is_mc(question_type) -> bool:
    """True only for known multiple-choice types. None/NaN -> False."""
    if question_type is None or (isinstance(question_type, float) and pd.isna(question_type)):
        return False
    return str(question_type).strip().lower() in MC_TYPES


def encode_state(time_seconds, answer_changes, was_correct, question_type, rapid_threshold) -> str:
    if is_mc(question_type) and answer_changes == 0:
        return "BF" if time_seconds <= rapid_threshold else "BS"
    if time_seconds <= rapid_threshold:
        return "R"
    if is_mc(question_type) and answer_changes >= REVISION_MIN:
        return "V"
    return "C" if was_correct else "W"


def encode_frame(df: pd.DataFrame, item_thresholds: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        item_thresholds[["test_id", "question_id", "threshold"]]
        .rename(columns={"threshold": "rapid_threshold"}),
        on=["test_id", "question_id"], how="left",
    )
    if df["rapid_threshold"].isna().any():
        n_missing = df.loc[df["rapid_threshold"].isna(), ["test_id", "question_id"]] \
            .drop_duplicates().shape[0]
        raise ValueError(f"missing rapid_threshold for {n_missing} items; "
                          f"item_thresholds must cover every (test_id, question_id) in df")

    states = [
        encode_state(t, ch, c, qt, rt)
        for t, ch, c, qt, rt in zip(
            df["time_seconds"], df["answer_changes"],
            df["was_correct"], df["question_type"], df["rapid_threshold"],
        )
    ]
    return df.assign(state=states).drop(columns=["rapid_threshold"])
