"""
Build a real-GNSS residual baseline for NavIC-SHIELD.

Input:
    results/csv/real_gnss_satellite_residuals.csv

Outputs:
    results/csv/real_gnss_satellite_baseline.csv
    results/csv/real_gnss_temporal_baseline.csv

Purpose:
    Characterize normal GNSS residual behavior before introducing
    spoofing/anomaly detection.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "real_gnss_satellite_residuals.csv"
)

SATELLITE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "real_gnss_satellite_baseline.csv"
)

TEMPORAL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "real_gnss_temporal_baseline.csv"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_OBSERVATIONS = 100

RESIDUAL_THRESHOLD_20M = 20.0
RESIDUAL_THRESHOLD_50M = 50.0


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def mad(series: pd.Series) -> float:
    """
    Median Absolute Deviation.

    MAD = median(|x - median(x)|)
    """
    values = series.dropna().to_numpy(dtype=float)

    if len(values) == 0:
        return np.nan

    median = np.median(values)

    return float(
        np.median(np.abs(values - median))
    )


def longest_consecutive_run(values: pd.Series) -> int:
    """
    Return the longest consecutive run of True values.

    The series must be ordered by epoch.
    """

    values = values.astype(bool).to_numpy()

    longest = 0
    current = 0

    for value in values:

        if value:
            current += 1
            longest = max(longest, current)

        else:
            current = 0

    return longest


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 60)
    print("REAL GNSS BASELINE ANALYSIS")
    print("=" * 60)

    print(f"\nLoading residual data:")
    print(INPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nResidual file not found:\n{INPUT_PATH}\n\n"
            "Run test_real_gnss_position.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "t",
        "satellite_id",
        "Svid",
        "residual_m",
        "abs_residual_m",
        "cn0_db_hz",
        "elevation_deg",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    print(f"\nTotal observations: {len(df):,}")
    print(
        f"Unique satellites: "
        f"{df['satellite_id'].nunique()}"
    )
    print(
        f"Unique epochs: "
        f"{df['t'].nunique():,}"
    )

    # Make sure data is ordered correctly.
    df = df.sort_values(
        ["satellite_id", "t"]
    ).reset_index(drop=True)

    # -----------------------------------------------------------------
    # Add anomaly flags
    # -----------------------------------------------------------------

    df["outlier_20m"] = (
        df["abs_residual_m"]
        > RESIDUAL_THRESHOLD_20M
    )

    df["outlier_50m"] = (
        df["abs_residual_m"]
        > RESIDUAL_THRESHOLD_50M
    )

    # -----------------------------------------------------------------
    # Satellite-specific baseline
    # -----------------------------------------------------------------

    satellite_counts = (
        df.groupby("satellite_id")
        .size()
        .rename("observations")
    )

    valid_satellites = satellite_counts[
        satellite_counts >= MIN_OBSERVATIONS
    ].index

    baseline_df = df[
        df["satellite_id"].isin(valid_satellites)
    ].copy()

    print(
        f"\nSatellites with >= "
        f"{MIN_OBSERVATIONS} observations: "
        f"{len(valid_satellites)}"
    )

    print(
        f"Observations included in baseline: "
        f"{len(baseline_df):,}"
    )

    # -----------------------------------------------------------------
    # Satellite statistics
    # -----------------------------------------------------------------

    satellite_baseline = (
        baseline_df
        .groupby("satellite_id")
        .agg(
            observations=(
                "abs_residual_m",
                "size",
            ),

            mean_residual_m=(
                "residual_m",
                "mean",
            ),

            median_residual_m=(
                "residual_m",
                "median",
            ),

            mean_abs_residual_m=(
                "abs_residual_m",
                "mean",
            ),

            median_abs_residual_m=(
                "abs_residual_m",
                "median",
            ),

            mad_abs_residual_m=(
                "abs_residual_m",
                mad,
            ),

            p95_abs_residual_m=(
                "abs_residual_m",
                lambda x: x.quantile(0.95),
            ),

            p99_abs_residual_m=(
                "abs_residual_m",
                lambda x: x.quantile(0.99),
            ),

            max_abs_residual_m=(
                "abs_residual_m",
                "max",
            ),

            mean_cn0_db_hz=(
                "cn0_db_hz",
                "mean",
            ),

            mean_elevation_deg=(
                "elevation_deg",
                "mean",
            ),

            outlier_20m_count=(
                "outlier_20m",
                "sum",
            ),

            outlier_50m_count=(
                "outlier_50m",
                "sum",
            ),
        )
        .reset_index()
    )

    # Convert counts into percentages.
    satellite_baseline[
        "outlier_rate_20m_pct"
    ] = (
        satellite_baseline["outlier_20m_count"]
        / satellite_baseline["observations"]
        * 100.0
    )

    satellite_baseline[
        "outlier_rate_50m_pct"
    ] = (
        satellite_baseline["outlier_50m_count"]
        / satellite_baseline["observations"]
        * 100.0
    )

    # -----------------------------------------------------------------
    # Temporal persistence per satellite
    # -----------------------------------------------------------------

    persistence_rows = []

    for satellite_id, group in baseline_df.groupby(
        "satellite_id"
    ):

        group = group.sort_values("t").copy()

        # -------------------------------------------------------------
        # Longest consecutive >20 m run
        # -------------------------------------------------------------

        longest_20m = longest_consecutive_run(
            group["outlier_20m"]
        )

        longest_50m = longest_consecutive_run(
            group["outlier_50m"]
        )

        # -------------------------------------------------------------
        # Total abnormal epochs
        # -------------------------------------------------------------

        abnormal_20m_count = int(
            group["outlier_20m"].sum()
        )

        abnormal_50m_count = int(
            group["outlier_50m"].sum()
        )

        # -------------------------------------------------------------
        # Number of abnormal runs
        # -------------------------------------------------------------

        outlier_20 = (
            group["outlier_20m"]
            .astype(int)
        )

        outlier_50 = (
            group["outlier_50m"]
            .astype(int)
        )

        runs_20m = int(
            (
                (outlier_20 == 1)
                & (outlier_20.shift(1, fill_value=0) == 0)
            ).sum()
        )

        runs_50m = int(
            (
                (outlier_50 == 1)
                & (outlier_50.shift(1, fill_value=0) == 0)
            ).sum()
        )

        # -------------------------------------------------------------
        # Temporal statistics
        # -------------------------------------------------------------

        persistence_rows.append({
            "satellite_id": satellite_id,

            "observations": len(group),

            "abnormal_epochs_20m": abnormal_20m_count,
            "abnormal_epochs_50m": abnormal_50m_count,

            "outlier_rate_20m_pct": (
                abnormal_20m_count
                / len(group)
                * 100.0
            ),

            "outlier_rate_50m_pct": (
                abnormal_50m_count
                / len(group)
                * 100.0
            ),

            "number_of_20m_runs": runs_20m,
            "number_of_50m_runs": runs_50m,

            "longest_20m_run_epochs": longest_20m,
            "longest_50m_run_epochs": longest_50m,

            "max_abs_residual_m": (
                group["abs_residual_m"].max()
            ),

            "median_abs_residual_m": (
                group["abs_residual_m"].median()
            ),

            "p95_abs_residual_m": (
                group["abs_residual_m"]
                .quantile(0.95)
            ),
        })

    temporal_baseline = pd.DataFrame(
        persistence_rows
    )

    # -----------------------------------------------------------------
    # Save satellite baseline
    # -----------------------------------------------------------------

    SATELLITE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    satellite_baseline = satellite_baseline.sort_values(
        "mean_abs_residual_m",
        ascending=False,
    )

    satellite_baseline.to_csv(
        SATELLITE_OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Save temporal baseline
    # -----------------------------------------------------------------

    temporal_baseline = temporal_baseline.sort_values(
        "longest_20m_run_epochs",
        ascending=False,
    )

    temporal_baseline.to_csv(
        TEMPORAL_OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SATELLITE BASELINE")
    print("=" * 60)

    print(
        satellite_baseline[
            [
                "satellite_id",
                "observations",
                "mean_abs_residual_m",
                "median_abs_residual_m",
                "mad_abs_residual_m",
                "p95_abs_residual_m",
                "max_abs_residual_m",
                "mean_cn0_db_hz",
                "mean_elevation_deg",
                "outlier_rate_20m_pct",
                "outlier_rate_50m_pct",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("TEMPORAL PERSISTENCE BASELINE")
    print("=" * 60)

    print(
        temporal_baseline[
            [
                "satellite_id",
                "observations",
                "abnormal_epochs_20m",
                "abnormal_epochs_50m",
                "number_of_20m_runs",
                "number_of_50m_runs",
                "longest_20m_run_epochs",
                "longest_50m_run_epochs",
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------------------
    # Global baseline summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 60)
    print("GLOBAL REAL-GNSS BASELINE")
    print("=" * 60)

    print(
        f"Mean |residual|:      "
        f"{df['abs_residual_m'].mean():.2f} m"
    )

    print(
        f"Median |residual|:    "
        f"{df['abs_residual_m'].median():.2f} m"
    )

    print(
        f"95th percentile:      "
        f"{df['abs_residual_m'].quantile(0.95):.2f} m"
    )

    print(
        f"99th percentile:      "
        f"{df['abs_residual_m'].quantile(0.99):.2f} m"
    )

    print(
        f"Maximum:              "
        f"{df['abs_residual_m'].max():.2f} m"
    )

    print(
        f">20 m observations:    "
        f"{df['outlier_20m'].sum():,} "
        f"({df['outlier_20m'].mean() * 100:.2f}%)"
    )

    print(
        f">50 m observations:    "
        f"{df['outlier_50m'].sum():,} "
        f"({df['outlier_50m'].mean() * 100:.2f}%)"
    )

    # -----------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------

    print("\nSaved satellite baseline:")
    print(SATELLITE_OUTPUT_PATH)

    print("\nSaved temporal baseline:")
    print(TEMPORAL_OUTPUT_PATH)


if __name__ == "__main__":
    main()