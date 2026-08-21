"""
Real GNSS satellite geometry analysis.

Computes GDOP / PDOP / HDOP / VDOP for the real Google GPS L1 dataset
and compares satellite geometry with the existing position error.

This is a diagnostic layer only. It does not modify the position solver.
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
    / "real_gnss_geometry.csv"
)

OUTPUT_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "real_gnss_geometry_vs_error.png"
)


def calculate_dop(group):
    """
    Calculate GNSS DOP values for one epoch.

    Uses the standard geometry matrix:

        H = [-los_x, -los_y, -los_z, 1]

    where LOS is the unit vector from receiver to satellite.

    Returns:
        GDOP, PDOP, HDOP, VDOP
    """

    if len(group) < 4:
        return np.nan, np.nan, np.nan, np.nan

    receiver = np.array([
        group["receiver_x_true_m"].iloc[0],
        group["receiver_y_true_m"].iloc[0],
        group["receiver_z_true_m"].iloc[0],
    ])

    satellites = group[
        ["x", "y", "z"]
    ].to_numpy(dtype=float)

    # Satellite -> receiver geometry
    vectors = satellites - receiver

    ranges = np.linalg.norm(vectors, axis=1)

    # Remove invalid geometry
    valid = np.isfinite(ranges) & (ranges > 0)

    if valid.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan

    vectors = vectors[valid]
    ranges = ranges[valid]

    # Line-of-sight unit vectors
    los = vectors / ranges[:, None]

    # Geometry matrix
    H = np.column_stack([
        -los[:, 0],
        -los[:, 1],
        -los[:, 2],
        np.ones(len(los)),
    ])

    try:
        Q = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan

    # Position DOP
    pdop = np.sqrt(
        Q[0, 0] +
        Q[1, 1] +
        Q[2, 2]
    )

    # Horizontal DOP
    hdop = np.sqrt(
        Q[0, 0] +
        Q[1, 1]
    )

    # Vertical DOP
    vdop = np.sqrt(Q[2, 2])

    # Geometric DOP includes receiver clock
    gdop = np.sqrt(
        Q[0, 0] +
        Q[1, 1] +
        Q[2, 2] +
        Q[3, 3]
    )

    return gdop, pdop, hdop, vdop


def main():

    print(f"Loading GNSS data:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # ---------------------------------------------------------------
    # First experiment: GPS L1 only
    # ---------------------------------------------------------------

    gps = df[
        (df["ConstellationType"] == 1)
        & (df["SignalType"] == "GPS_L1")
    ].copy()

    print(f"\nGPS L1 observations: {len(gps):,}")
    print(f"GPS L1 epochs:       {gps['t'].nunique():,}")

    geometry_rows = []

    for t, group in gps.groupby("t", sort=True):

        gdop, pdop, hdop, vdop = calculate_dop(group)

        geometry_rows.append({
            "t": t,
            "n_satellites": len(group),
            "gdop": gdop,
            "pdop": pdop,
            "hdop": hdop,
            "vdop": vdop,
            "mean_cn0_db_hz": group["cn0_db_hz"].mean(),
            "min_elevation_deg": group["elevation_deg"].min(),
            "mean_elevation_deg": group["elevation_deg"].mean(),
        })

    geometry = pd.DataFrame(geometry_rows)

    # ---------------------------------------------------------------
    # Merge with existing position error
    # ---------------------------------------------------------------

    if POSITION_RESULTS_PATH.exists():

        position_results = pd.read_csv(
            POSITION_RESULTS_PATH
        )

        geometry = geometry.merge(
            position_results[
                ["t", "position_error_m"]
            ],
            on="t",
            how="left",
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    geometry.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    valid = geometry.dropna(
        subset=["pdop", "position_error_m"]
    )

    print("\n" + "=" * 60)
    print("REAL GNSS GEOMETRY ANALYSIS")
    print("=" * 60)

    print(f"Epochs analysed:     {len(geometry):,}")
    print(f"Valid DOP epochs:    {geometry['pdop'].notna().sum():,}")

    print("\nDOP statistics:")

    for name in ["gdop", "pdop", "hdop", "vdop"]:
        print(
            f"  {name.upper():5s} "
            f"mean={geometry[name].mean():.3f}, "
            f"median={geometry[name].median():.3f}, "
            f"max={geometry[name].max():.3f}"
        )

    if len(valid) > 10:

        correlation = valid[
            ["pdop", "position_error_m"]
        ].corr().iloc[0, 1]

        print(
            f"\nPDOP ↔ position-error correlation: "
            f"{correlation:.3f}"
        )

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------

    if "position_error_m" in geometry.columns:

        fig, ax1 = plt.subplots(
            figsize=(12, 6)
        )

        time_minutes = geometry["t"] / 60.0

        ax1.plot(
            time_minutes,
            geometry["position_error_m"],
            label="Position error",
        )

        ax1.set_xlabel("Time (minutes)")
        ax1.set_ylabel("Position error (m)")
        ax1.grid(True)

        ax2 = ax1.twinx()

        ax2.plot(
            time_minutes,
            geometry["pdop"],
            label="PDOP",
        )

        ax2.set_ylabel("PDOP")

        ax1.set_title(
            "Real GPS Position Error vs Satellite Geometry"
        )

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
            f"\nSaved figure:\n{OUTPUT_FIGURE}"
        )

    print(
        f"\nSaved geometry data:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()