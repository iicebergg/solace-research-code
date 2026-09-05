"""
    uv run src/run_analysis.py               # full run
    uv run src/run_analysis.py --sample       # 1,000-attempt subsample, fast iteration
    uv run src/run_analysis.py --sample 200   # custom subsample size
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd

from load import load_joined
from encode import encode_frame
from thresholds import compute_item_thresholds
from features import parse_test_id
from clean import attempt_features, classify_attempts
from sequential import run as run_sequential
from changepoint import changepoint_records, summarize as summarize_changepoints

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_SAMPLE_SIZE = 1000
SAMPLE_SEED = 0


def log(msg: str) -> None:
    print(f"[run_analysis] {msg}")


def maybe_sample(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    if sample_size is None:
        return df
    attempt_ids = df["attempt_id"].unique()
    if sample_size >= len(attempt_ids):
        log(f"--sample {sample_size} >= {len(attempt_ids)} attempts available; using all of them")
        return df
    rng = pd.Series(attempt_ids).sample(n=sample_size, random_state=SAMPLE_SEED)
    log(f"sampling {sample_size:,} of {len(attempt_ids):,} attempts (seed={SAMPLE_SEED})")
    return df[df["attempt_id"].isin(set(rng))].reset_index(drop=True)


def annotate_heatmap(im, data: pd.DataFrame, valfmt: str = "{:.2f}") -> None:
    """
    Adapted from matplotlib's "Annotated heatmap" gallery example:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    values = data.to_numpy()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if pd.isna(value):
                continue
            color = "white" if abs(im.norm(value) - 0.5) > 0.3 else "black"
            im.axes.text(j, i, valfmt.format(value), ha="center", va="center",
                         color=color, fontsize=8)


def write_sequential_outputs(encoded: pd.DataFrame, attempt_ids, out_dir: Path) -> dict:
    """sequential.py: transition matrix + lift over the given attempts."""
    seq_result = run_sequential(encoded, attempt_ids)
    log(f"sequential: {seq_result['n_attempts']:,} clean attempts, "
        f"{seq_result['n_transitions']:,} transitions")
    seq_result["transition_matrix"].to_csv(out_dir / "transition_matrix.csv", index_label="current_state")
    seq_result["lift"].to_csv(out_dir / "transition_lift.csv", index_label="current_state")

    fig, ax = plt.subplots(figsize=(5, 4))
    lift_norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)
    im = ax.imshow(seq_result["lift"], cmap="RdBu_r", norm=lift_norm)
    annotate_heatmap(im, seq_result["lift"])
    n_states = len(seq_result["lift"])
    ax.set_xticks(range(n_states)); ax.set_xticklabels(seq_result["lift"].columns)
    ax.set_yticks(range(n_states)); ax.set_yticklabels(seq_result["lift"].index)
    ax.set_xlabel("next state"); ax.set_ylabel("current state")
    ax.set_title("state transition lift vs. base rate")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "transition_lift_heatmap.png", dpi=300)
    plt.close(fig)
    log(f"wrote transition_matrix.csv, transition_lift.csv, transition_lift_heatmap.png -> {out_dir}")
    return seq_result


def write_changepoint_outputs(encoded: pd.DataFrame, attempt_ids, out_dir: Path) -> pd.DataFrame:
    """changepoint.py: PELT on the given attempts' correctness series."""
    cp_records = changepoint_records(encoded, attempt_ids)
    cp_stats = summarize_changepoints(cp_records)
    log(f"changepoint: {cp_stats['n_changepoints']:,} changepoints across "
        f"{cp_stats['n_attempts_with_changepoint']:,} of {len(attempt_ids):,} clean attempts")
    cp_records.to_csv(out_dir / "changepoint_records.csv", index=False)

    if len(cp_records):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(cp_records["position_frac"], bins=20)
        axes[0].set_title("changepoint position (fraction of attempt)")
        axes[0].set_xlabel("position"); axes[0].set_ylabel("count")
        axes[1].hist(cp_records["accuracy_delta"], bins=20)
        axes[1].set_title("accuracy delta (after - before)")
        axes[1].set_xlabel("delta"); axes[1].set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "changepoint_distributions.png", dpi=300)
        plt.close(fig)
        log(f"wrote changepoint_records.csv, changepoint_distributions.png -> {out_dir}")
    else:
        log(f"wrote changepoint_records.csv (empty, no changepoints) -> {out_dir}")
    return cp_records


def run_grouped_analyses(encoded: pd.DataFrame, group_by: pd.Series, clean_ids, kind: str) -> None:
    """Rerun sequential/changepoint separately for each distinct value of
    group_by (test_id, level, or subject), restricted to that group's own
    clean attempts, writing results to outputs/<group_value>/.

    group_by must be indexed by attempt_id (e.g. features["test_id"], or a
    subject/level column derived from parse_test_id), covering at least
    clean_ids.
    """
    grouped_clean = group_by.loc[clean_ids]
    group_values = sorted(grouped_clean.unique())
    log(f"per-{kind}: running sequential/changepoint for {len(group_values)} {kind}s")

    for value in group_values:
        group_clean_ids = grouped_clean[grouped_clean == value].index
        out_dir = OUTPUTS / value
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"--- {kind}={value} ({len(group_clean_ids):,} clean attempts) ---")
        try:
            write_sequential_outputs(encoded, group_clean_ids, out_dir)
            write_changepoint_outputs(encoded, group_clean_ids, out_dir)
        except Exception as e:
            log(f"!!! {kind}={value} failed: {e!r}; skipping")


def main(sample_size: int | None) -> None:
    OUTPUTS.mkdir(exist_ok=True)

    df = load_joined()
    log(f"loaded {len(df):,} responses across {df['attempt_id'].nunique():,} attempts")

    df = maybe_sample(df, sample_size)

    item_thresholds = compute_item_thresholds(df)
    item_thresholds.to_csv(OUTPUTS / "rapid_thresholds.csv", index=False)
    thin = (item_thresholds["response_count"] < 30).mean()
    log(f"rapid thresholds: {len(item_thresholds):,} items, "
        f"{thin:.1%} with <30 responses, "
        f"{item_thresholds['cap_bound'].mean():.1%} cap-bound "
        f"(wrote rapid_thresholds.csv)")

    encoded = encode_frame(df, item_thresholds)
    log(f"encoded {len(encoded):,} responses; state counts: "
        f"{encoded['state'].value_counts().to_dict()}")

    # --- clean.py: classify attempts, log exclusion fractions ---
    features = attempt_features(encoded)
    labels = classify_attempts(features)
    label_counts = labels.value_counts()
    total = len(labels)
    log("attempt classification: " + ", ".join(
        f"{name}={count:,} ({count / total:.1%})"
        for name, count in label_counts.items()
    ))
    clean_ids = labels[labels == "clean"].index

    features.join(labels).to_csv(OUTPUTS / "attempt_features.csv")
    log(f"wrote attempt_features.csv ({len(features):,} rows)")

    # --- sequential.py: transition matrix + lift over clean attempts ---
    write_sequential_outputs(encoded, clean_ids, OUTPUTS)

    # --- changepoint.py: PELT on clean attempts' correctness series ---
    write_changepoint_outputs(encoded, clean_ids, OUTPUTS)

    # --- grouped breakdowns: sequential + changepoint, one folder per
    # test_id / grade level / subject ---
    test_dims = features["test_id"].apply(parse_test_id).apply(pd.Series)
    run_grouped_analyses(encoded, features["test_id"], clean_ids, kind="test")
    run_grouped_analyses(encoded, test_dims["level"], clean_ids, kind="level")
    run_grouped_analyses(encoded, test_dims["subject"], clean_ids, kind="subject")

    log(f"done. outputs written to {OUTPUTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=int, nargs="?", const=DEFAULT_SAMPLE_SIZE, default=None,
        help=f"subsample this many attempts for a fast iteration run "
             f"(default {DEFAULT_SAMPLE_SIZE} if flag given with no value)",
    )
    args = parser.parse_args()
    main(args.sample)
