from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True)
class CleanerConfig:
    """Runtime options for the 4H CSV cleaner.

    Attributes:
        input_csv: Source CSV path.
        output_csv: Cleaned CSV output path.
        report_json: JSON report output path.
        issues_csv: Row-level issue export path.
        freq_hours: Expected bar interval in hours.
        drop_zero_volume: Whether to remove zero-volume bars.
        zscore_threshold: Absolute z-score used for return outlier flagging.
    """

    input_csv: Path
    output_csv: Path
    report_json: Path
    issues_csv: Path
    freq_hours: int = 4
    drop_zero_volume: bool = False
    zscore_threshold: float = 5.0


def parse_args() -> CleanerConfig:
    """Parse CLI args and return strongly typed cleaner configuration."""

    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "xrp_4h.csv"
    default_output = script_dir / "data" / "xrp_4h_clean.csv"
    default_report = script_dir / "data" / "xrp_4h_clean_report.json"
    default_issues = script_dir / "data" / "xrp_4h_issues.csv"

    parser = argparse.ArgumentParser(description="Clean and QA XRP 4H OHLCV CSV data.")
    parser.add_argument("--input", dest="input_csv", default=str(default_input), help="Input CSV path")
    parser.add_argument("--output", dest="output_csv", default=str(default_output), help="Cleaned CSV output path")
    parser.add_argument("--report", dest="report_json", default=str(default_report), help="JSON report path")
    parser.add_argument("--issues", dest="issues_csv", default=str(default_issues), help="Issues CSV path")
    parser.add_argument("--freq-hours", dest="freq_hours", type=int, default=4, help="Expected bar interval in hours")
    parser.add_argument(
        "--drop-zero-volume",
        action="store_true",
        help="Drop rows where volume is zero or negative",
    )
    parser.add_argument(
        "--zscore-threshold",
        dest="zscore_threshold",
        type=float,
        default=5.0,
        help="Absolute z-score threshold for return outlier flags",
    )

    args = parser.parse_args()
    return CleanerConfig(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        report_json=Path(args.report_json),
        issues_csv=Path(args.issues_csv),
        freq_hours=args.freq_hours,
        drop_zero_volume=bool(args.drop_zero_volume),
        zscore_threshold=float(args.zscore_threshold),
    )


def load_and_standardize(csv_path: Path) -> pd.DataFrame:
    """Load CSV and standardize schema and dtypes.

    Args:
        csv_path: Source CSV containing Date/Open/High/Low/Close/Volume.

    Returns:
        DataFrame indexed by UTC Date with required OHLCV columns.

    Raises:
        ValueError: If required columns are missing.
    """

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def build_issue_mask(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build row-level issue flags and summarized counts.

    Args:
        df: Standardized OHLCV DataFrame indexed by Date.

    Returns:
        Tuple of issue DataFrame and per-issue counts.
    """

    issue_df = pd.DataFrame(index=df.index)

    issue_df["nan_any"] = df[["Open", "High", "Low", "Close", "Volume"]].isna().any(axis=1)
    issue_df["non_positive_price"] = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    issue_df["high_lt_low"] = df["High"] < df["Low"]
    issue_df["high_lt_open_or_close"] = df["High"] < df[["Open", "Close"]].max(axis=1)
    issue_df["low_gt_open_or_close"] = df["Low"] > df[["Open", "Close"]].min(axis=1)
    issue_df["non_positive_volume"] = df["Volume"] <= 0

    returns = df["Close"].pct_change()
    ret_std = float(returns.std(skipna=True))
    if ret_std > 0:
        z = (returns - returns.mean(skipna=True)) / ret_std
        issue_df["return_outlier"] = z.abs() > 5.0
    else:
        issue_df["return_outlier"] = False

    counts = {col: int(issue_df[col].sum()) for col in issue_df.columns}
    return issue_df, counts


def detect_interval_issues(index: pd.DatetimeIndex, freq_hours: int) -> Dict[str, Any]:
    """Detect interval gaps and unexpected timestamp deltas.

    Args:
        index: Datetime index in ascending order.
        freq_hours: Expected interval in hours.

    Returns:
        Dict containing missing timestamp stats and non-standard jump details.
    """

    expected_delta = pd.Timedelta(hours=freq_hours)
    deltas = index.to_series().diff().dropna()
    non_standard = deltas[deltas != expected_delta]

    if len(index) > 0:
        expected_index = pd.date_range(start=index.min(), end=index.max(), freq=f"{freq_hours}h", tz="UTC")
        missing_index = expected_index.difference(index)
    else:
        missing_index = pd.DatetimeIndex([], tz="UTC")

    jumps: List[Dict[str, str]] = []
    for ts, gap in non_standard.items():
        prev_ts = index[index.get_loc(ts) - 1]
        jumps.append(
            {
                "previous_timestamp": prev_ts.isoformat(),
                "current_timestamp": ts.isoformat(),
                "gap": str(gap),
            }
        )

    return {
        "expected_freq_hours": freq_hours,
        "non_standard_delta_count": int(non_standard.shape[0]),
        "missing_bar_count": int(missing_index.shape[0]),
        "missing_timestamps": [ts.isoformat() for ts in missing_index],
        "non_standard_jumps": jumps,
    }


def clean_dataframe(df: pd.DataFrame, issues: pd.DataFrame, cfg: CleanerConfig) -> pd.DataFrame:
    """Apply deterministic cleaning rules and return a cleaned DataFrame.

    Args:
        df: Standardized OHLCV DataFrame.
        issues: Row-level issue flags.
        cfg: Cleaner settings.

    Returns:
        Cleaned DataFrame preserving original OHLCV schema.
    """

    drop_mask = (
        issues["nan_any"]
        | issues["non_positive_price"]
        | issues["high_lt_low"]
        | issues["high_lt_open_or_close"]
        | issues["low_gt_open_or_close"]
    )

    if cfg.drop_zero_volume:
        drop_mask = drop_mask | issues["non_positive_volume"]

    cleaned = df.loc[~drop_mask].copy()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.sort_index()
    return cleaned


def write_outputs(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    issues: pd.DataFrame,
    issue_counts: Dict[str, int],
    interval_report: Dict[str, Any],
    cfg: CleanerConfig,
) -> None:
    """Write cleaned CSV, issues CSV, and JSON QA report to disk."""

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.issues_csv.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.reset_index().to_csv(cfg.output_csv, index=False)

    issue_export = raw_df.copy()
    for col in issues.columns:
        issue_export[col] = issues[col]
    issue_export["has_any_issue"] = issues.any(axis=1)
    issue_export.reset_index().to_csv(cfg.issues_csv, index=False)

    report: Dict[str, Any] = {
        "input_csv": str(cfg.input_csv),
        "output_csv": str(cfg.output_csv),
        "issues_csv": str(cfg.issues_csv),
        "rows_input": int(raw_df.shape[0]),
        "rows_output": int(cleaned_df.shape[0]),
        "rows_removed": int(raw_df.shape[0] - cleaned_df.shape[0]),
        "date_min_input": raw_df.index.min().isoformat() if raw_df.shape[0] else None,
        "date_max_input": raw_df.index.max().isoformat() if raw_df.shape[0] else None,
        "date_min_output": cleaned_df.index.min().isoformat() if cleaned_df.shape[0] else None,
        "date_max_output": cleaned_df.index.max().isoformat() if cleaned_df.shape[0] else None,
        "issue_counts": issue_counts,
        "interval_report": interval_report,
        "drop_zero_volume": cfg.drop_zero_volume,
        "zscore_threshold": cfg.zscore_threshold,
    }

    with cfg.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    """Run CSV cleaning pipeline and print a concise summary."""

    cfg = parse_args()
    raw_df = load_and_standardize(cfg.input_csv)
    issues, issue_counts = build_issue_mask(raw_df)

    # Apply configured outlier threshold after issue frame creation.
    returns = raw_df["Close"].pct_change()
    ret_std = float(returns.std(skipna=True))
    if ret_std > 0:
        z = (returns - returns.mean(skipna=True)) / ret_std
        issues["return_outlier"] = z.abs() > cfg.zscore_threshold
        issue_counts["return_outlier"] = int(issues["return_outlier"].sum())

    cleaned_df = clean_dataframe(raw_df, issues, cfg)
    interval_report = detect_interval_issues(raw_df.index, cfg.freq_hours)

    write_outputs(raw_df, cleaned_df, issues, issue_counts, interval_report, cfg)

    print(f"Input rows:   {raw_df.shape[0]}")
    print(f"Output rows:  {cleaned_df.shape[0]}")
    print(f"Rows removed: {raw_df.shape[0] - cleaned_df.shape[0]}")
    print(f"Report:       {cfg.report_json}")
    print(f"Issues CSV:   {cfg.issues_csv}")
    print(f"Clean CSV:    {cfg.output_csv}")


if __name__ == "__main__":
    main()
