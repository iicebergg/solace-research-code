"""Parse test_id into structural dimensions and convert timestamps to Eastern.

The exported subject and grade_band columns are uninformative ("unknown"),
so all structure comes from parsing test_id, which follows the pattern
{subject}_{level}[_practice]_{year}.
"""
import pandas as pd


def parse_test_id(test_id: str) -> dict:
    parts = test_id.split("_")
    subject = parts[0]
    year = parts[-1]
    form_type = "practice" if "practice" in parts else "released"
    level = "_".join(p for p in parts[1:-1] if p != "practice")
    return {"subject": subject, "level": level,
            "form_type": form_type, "year": year}


def add_test_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["test_id"].apply(parse_test_id).apply(pd.Series)
    return pd.concat([df, parsed], axis=1)


def to_eastern(df: pd.DataFrame, cols=("started_at", "completed_at")) -> pd.DataFrame:
    """Add {col}_edt columns. All timestamps are UTC; the study assumes all
    users are in Virginia, so Eastern is the correct local frame."""
    for c in cols:
        s = pd.to_datetime(df[c], utc=True, format="mixed")
        df[c + "_edt"] = s.dt.tz_convert("America/New_York")
    return df
