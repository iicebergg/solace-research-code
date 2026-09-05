"""State-transition analysis of clean attempts.

Turns each attempt's encoded state sequence (see encode.py) into a 6x6
Markov transition matrix over states {C, W, V, R, BF, BS} and a lift matrix
showing how much more (or less) likely each next-state is given the current
state, relative to that state's overall base rate across all transitions.
Transitions never cross attempt boundaries.
"""
import numpy as np
import pandas as pd

STATES = ["C", "W", "V", "R", "BF", "BS"]


def state_sequences(encoded: pd.DataFrame, attempt_ids=None) -> dict:
    if attempt_ids is not None:
        encoded = encoded[encoded["attempt_id"].isin(attempt_ids)]
    return {aid: grp["state"].tolist()
            for aid, grp in encoded.groupby("attempt_id", sort=False)}


def transition_counts(sequences: dict) -> pd.DataFrame:
    """len(STATES) x len(STATES) matrix of (current_state -> next_state)
    counts within attempts."""
    counts = pd.DataFrame(0, index=STATES, columns=STATES, dtype=int)
    for seq in sequences.values():
        for cur, nxt in zip(seq, seq[1:]):
            counts.loc[cur, nxt] += 1
    return counts


def transition_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalized transition probabilities: P(next | current)."""
    row_sums = counts.sum(axis=1)
    return counts.div(row_sums.replace(0, np.nan), axis=0)


def baseline_distribution(counts: pd.DataFrame) -> pd.Series:
    """Marginal P(next-state), ignoring current state: column totals over
    all transitions, normalized."""
    col_sums = counts.sum(axis=0)
    return col_sums / col_sums.sum()


def lift_matrix(matrix: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    """lift[i, j] = P(next=j | current=i) / P(next=j).
    > 1: state j more likely to follow i than its base rate. < 1: less likely."""
    return matrix.div(baseline, axis=1)


def run(encoded: pd.DataFrame, attempt_ids=None) -> dict:
    sequences = state_sequences(encoded, attempt_ids)
    counts = transition_counts(sequences)
    matrix = transition_matrix(counts)
    baseline = baseline_distribution(counts)
    lift = lift_matrix(matrix, baseline)
    return {
        "counts": counts,
        "transition_matrix": matrix,
        "baseline": baseline,
        "lift": lift,
        "n_attempts": len(sequences),
        "n_transitions": int(counts.values.sum()),
    }


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
    result = run(encoded, clean_ids)
    print(f"clean attempts: {result['n_attempts']:,}, transitions: {result['n_transitions']:,}")
    print("\ntransition matrix (row = current, col = next):")
    print(result["transition_matrix"].round(3))
    print("\nbaseline next-state distribution:")
    print(result["baseline"].round(3))
    print("\nlift (>1 = more likely than base rate):")
    print(result["lift"].round(2))
