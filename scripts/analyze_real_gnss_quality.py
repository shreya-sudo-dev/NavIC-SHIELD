"""
Real GNSS measurement-quality analysis.

Analyzes whether C/N0, elevation, and satellite count are related
to the observed real-GNSS positioning error.
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
    / "real_gnss_quality.csv"
)

OUTPUT_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "real_gnss_quality_vs_error.png"
)


def main():

    print(f"Loading GNSS data:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # GPS L1 only for a clean comparison with the position solver.
    gps = df[
        (df["ConstellationType"] == 1)
        & (df["SignalType"] == "GPS_L1")
    ].copy()

    print(f"\nGPS L1 observations: {len(gps):,}")
    print(f"GPS L1 epochs:       {gps['t'].nunique():,}")

    # ---------------------------------------------------------------
    # Per-epoch measurement quality
    # ---------------------------------------------------------------

    rows = []

    for t, group in gps.groupby("t", sort=True):

        cn0 = group["cn0_db_hz"].dropna()
        elevation = group["elevation_deg"].dropna()

        rows.append({
            "t": t,

            "satellite_count": len(group),

            "mean_cn0_db_hz": cn0.mean(),
            "median_cn0_db_hz": cn0.median(),
            "min_cn0_db_hz": cn0.min(),

            "mean_elevation_deg": elevation.mean(),
            "median_elevation_deg": elevation.median(),
            "min_elevation_deg": elevation.min(),

            "weak_signal_count": int(
                (cn0 < 20).sum()
            ),

            "very_weak_signal_count": int(
                (cn0 < 15).sum()
            ),

            "low_elevation_count": int(
                (elevation < 15).sum()
            ),

            "very_low_elevation_count": int(
                (elevation < 10).sum()
            ),
        })

    quality = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Merge position error
    # ---------------------------------------------------------------

    if not POSITION_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Position results not found:\n"
            f"{POSITION_RESULTS_PATH}"
        )

    position = pd.read_csv(
        POSITION_RESULTS_PATH
    )

    quality = quality.merge(
        position[["t", "position_error_m"]],
        on="t",
        how="left",
    )

    # ---------------------------------------------------------------
    # Correlations
    # ---------------------------------------------------------------

    correlation_columns = [
        "position_error_m",
        "satellite_count",
        "mean_cn0_db_hz",
        "min_cn0_db_hz",
        "mean_elevation_deg",
        "min_elevation_deg",
        "weak_signal_count",
        "very_weak_signal_count",
        "low_elevation_count",
        "very_low_elevation_count",
    ]

    correlations = (
        quality[correlation_columns]
        .corr()["position_error_m"]
        .drop("position_error_m")
        .sort_values()
    )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("REAL GNSS MEASUREMENT QUALITY ANALYSIS")
    print("=" * 60)

    print(f"Epochs analysed: {len(quality):,}")

    print("\nMeasurement quality:")
    print(
        f"  Mean C/N0:              "
        f"{quality['mean_cn0_db_hz'].mean():.2f} dB-Hz"
    )
    print(
        f"  Mean elevation:         "
        f"{quality['mean_elevation_deg'].mean():.2f} deg"
    )
    print(
        f"  Mean satellites/epoch:  "
        f"{quality['satellite_count'].mean():.2f}"
    )

    print("\nWeak observations:")
    print(
        f"  Mean weak (<20 dB-Hz):  "
        f"{quality['weak_signal_count'].mean():.2f}"
    )
    print(
        f"  Mean very weak (<15):    "
        f"{quality['very_weak_signal_count'].mean():.2f}"
    )

    print("\nLow-elevation observations:")
    print(
        f"  Mean <15 deg:           "
        f"{quality['low_elevation_count'].mean():.2f}"
    )
    print(
        f"  Mean <10 deg:           "
        f"{quality['very_low_elevation_count'].mean():.2f}"
    )

    print("\nCorrelation with position error:")

    for name, value in correlations.items():
        print(
            f"  {name:28s}: {value:+.3f}"
        )

    # ---------------------------------------------------------------
    # Plot 1: position error + mean C/N0
    # ---------------------------------------------------------------

    fig, ax1 = plt.subplots(
        figsize=(12, 6)
    )

    time_minutes = quality["t"] / 60.0

    ax1.plot(
        time_minutes,
        quality["position_error_m"],
    )

    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Position error (m)")
    ax1.grid(True)

    ax2 = ax1.twinx()

    ax2.plot(
        time_minutes,
        quality["mean_cn0_db_hz"],
        linestyle="--",
    )

    ax2.set_ylabel("Mean C/N0 (dB-Hz)")

    ax1.set_title(
        "Real GPS Position Error vs Signal Quality"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"\nSaved figure:\n{OUTPUT_FIGURE}"
    )

    print(
        f"\nSaved quality data:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()