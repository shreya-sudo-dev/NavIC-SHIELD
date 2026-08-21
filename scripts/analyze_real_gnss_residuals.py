"""
Real GNSS per-satellite residual analysis.

For each GPS L1 observation:

    corrected pseudorange
        -
    predicted pseudorange from solved receiver position
        =
    measurement residual

The analysis identifies:
- residual statistics
- largest residuals
- satellites producing large residuals
- residual behaviour over time
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "real_gnss"
    / "google"
    / "sample"
    / "normalized_gnss.csv"
)

POSITION_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "real_gnss_position_test.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "real_gnss_residuals.csv"
)

OUTPUT_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "real_gnss_residuals.png"
)


def main():

    print(f"Loading GNSS data:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    gps = df[
        (df["ConstellationType"] == 1)
        & (df["SignalType"] == "GPS_L1")
    ].copy()

    print(f"\nGPS L1 observations: {len(gps):,}")
    print(f"GPS L1 epochs:       {gps['t'].nunique():,}")

    # ---------------------------------------------------------------
    # Load solved receiver positions
    # ---------------------------------------------------------------

    if not POSITION_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Position results not found:\n"
            f"{POSITION_RESULTS_PATH}"
        )

    positions = pd.read_csv(
        POSITION_RESULTS_PATH
    )

    required_position_columns = [
        "t",
        "solved_x",
        "solved_y",
        "solved_z",
    ]

    missing = [
        c for c in required_position_columns
        if c not in positions.columns
    ]

    if missing:
        raise ValueError(
            "Position results are missing columns: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------------
    # Merge solved position with measurements
    # ---------------------------------------------------------------

    gps = gps.merge(
        positions[required_position_columns],
        on="t",
        how="inner",
    )

    print(
        f"\nMatched observations with solved positions: "
        f"{len(gps):,}"
    )

    # ---------------------------------------------------------------
    # Compute predicted range
    # ---------------------------------------------------------------

    receiver = gps[
        [
            "solved_x",
            "solved_y",
            "solved_z",
        ]
    ].to_numpy(dtype=float)

    satellite = gps[
        [
            "x",
            "y",
            "z",
        ]
    ].to_numpy(dtype=float)

    predicted_range = np.linalg.norm(
        satellite - receiver,
        axis=1,
    )

    gps["predicted_range_m"] = predicted_range

    gps["residual_m"] = (
        gps["pseudorange_m"]
        - gps["predicted_range_m"]
    )

    gps["abs_residual_m"] = (
        gps["residual_m"].abs()
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("REAL GNSS PER-SATELLITE RESIDUAL ANALYSIS")
    print("=" * 60)

    print(
        f"Residual mean:       "
        f"{gps['residual_m'].mean():.3f} m"
    )

    print(
        f"Residual median:     "
        f"{gps['residual_m'].median():.3f} m"
    )

    print(
        f"Mean |residual|:     "
        f"{gps['abs_residual_m'].mean():.3f} m"
    )

    print(
        f"95th percentile:     "
        f"{gps['abs_residual_m'].quantile(0.95):.3f} m"
    )

    print(
        f"99th percentile:     "
        f"{gps['abs_residual_m'].quantile(0.99):.3f} m"
    )

    print(
        f"Maximum |residual|:  "
        f"{gps['abs_residual_m'].max():.3f} m"
    )

    # ---------------------------------------------------------------
    # Largest individual residuals
    # ---------------------------------------------------------------

    print("\nLargest absolute residuals:")

    largest = gps.nlargest(
        15,
        "abs_residual_m",
    )

    print(
        largest[
            [
                "t",
                "Svid",
                "satellite_id",
                "residual_m",
                "abs_residual_m",
                "cn0_db_hz",
                "elevation_deg",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Per-satellite statistics
    # ---------------------------------------------------------------

    satellite_stats = (
        gps.groupby("satellite_id")
        .agg(
            observations=("residual_m", "size"),
            mean_residual_m=("residual_m", "mean"),
            median_residual_m=("residual_m", "median"),
            mean_abs_residual_m=(
                "abs_residual_m",
                "mean",
            ),
            max_abs_residual_m=(
                "abs_residual_m",
                "max",
            ),
            p95_abs_residual_m=(
                "abs_residual_m",
                lambda x: x.quantile(0.95),
            ),
            mean_cn0_db_hz=(
                "cn0_db_hz",
                "mean",
            ),
            mean_elevation_deg=(
                "elevation_deg",
                "mean",
            ),
        )
        .sort_values(
            "mean_abs_residual_m",
            ascending=False,
        )
    )

    print("\nPer-satellite residual statistics:")

    print(
        satellite_stats.head(15).to_string()
    )

    # ---------------------------------------------------------------
    # Count potentially large residuals
    # ---------------------------------------------------------------

    thresholds = [
        10,
        20,
        50,
        100,
    ]

    print("\nResidual threshold counts:")

    for threshold in thresholds:

        count = (
            gps["abs_residual_m"] > threshold
        ).sum()

        percentage = (
            count / len(gps) * 100
        )

        print(
            f"  > {threshold:3d} m: "
            f"{count:6d} "
            f"({percentage:.2f}%)"
        )

    # ---------------------------------------------------------------
    # Save detailed residual dataset
    # ---------------------------------------------------------------

    output_columns = [
        "t",
        "Svid",
        "satellite_id",

        "x",
        "y",
        "z",

        "solved_x",
        "solved_y",
        "solved_z",

        "pseudorange_m",
        "predicted_range_m",
        "residual_m",
        "abs_residual_m",

        "cn0_db_hz",
        "elevation_deg",
        "azimuth_deg",
    ]

    output_columns = [
        c for c in output_columns
        if c in gps.columns
    ]

    residual_output = gps[
        output_columns
    ].copy()

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    residual_output.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------------
    # Plot residual magnitude over time
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    time_minutes = gps["t"] / 60.0

    ax.scatter(
        time_minutes,
        gps["abs_residual_m"],
        s=3,
        alpha=0.35,
    )

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("|Pseudorange residual| (m)")
    ax.set_title(
        "Real GPS Per-Satellite Pseudorange Residuals"
    )

    ax.grid(True)

    fig.tight_layout()

    OUTPUT_FIGURE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"\nSaved residual data:\n{OUTPUT_CSV}"
    )

    print(
        f"Saved figure:\n{OUTPUT_FIGURE}"
    )


if __name__ == "__main__":
    main()