"""
Cross-satellite consistency analysis for real GNSS data.

Purpose
-------
Analyze whether multiple satellites show abnormal residual behavior
at the same epoch.

This is a baseline-analysis step, NOT the final spoofing detector.

Inputs
------
results/csv/real_gnss_satellite_residuals.csv

Outputs
-------
results/csv/real_gnss_cross_satellite.csv

results/figures/
    real_gnss_cross_satellite.png
    real_gnss_cross_satellite_sign.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def longest_true_run(values):
    """
    Return the longest consecutive run of True values.
    """

    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def main():

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------

    residual_path = (
        PROJECT_ROOT
        / "results"
        / "csv"
        / "real_gnss_satellite_residuals.csv"
    )

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "csv"
    )

    figure_dir = (
        PROJECT_ROOT
        / "results"
        / "figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading residual data:")
    print(residual_path)

    df = pd.read_csv(residual_path)

    # ---------------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------------

    required_columns = [
        "t",
        "satellite_id",
        "residual_m",
        "abs_residual_m",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    print(
        f"\nTotal observations: {len(df):,}"
    )

    print(
        f"Unique satellites: "
        f"{df['satellite_id'].nunique():,}"
    )

    print(
        f"Unique epochs: "
        f"{df['t'].nunique():,}"
    )

    # ---------------------------------------------------------------
    # Per-satellite normal baseline
    #
    # We use the median absolute residual of each satellite.
    #
    # This is important because GPS-2 and GPS-12 do not have
    # exactly the same normal residual behavior.
    # ---------------------------------------------------------------

    satellite_baseline = (
        df.groupby("satellite_id")["abs_residual_m"]
        .median()
        .rename("satellite_median_abs_residual_m")
    )

    df = df.merge(
        satellite_baseline,
        on="satellite_id",
        how="left",
    )

    # Avoid division by zero.
    baseline = (
        df["satellite_median_abs_residual_m"]
        .clip(lower=0.1)
    )

    df["normalized_residual"] = (
        df["abs_residual_m"] / baseline
    )

    # ---------------------------------------------------------------
    # Abnormality definitions
    # ---------------------------------------------------------------

    # Absolute residual threshold.
    df["abnormal_20m"] = (
        df["abs_residual_m"] > 20.0
    )

    df["abnormal_50m"] = (
        df["abs_residual_m"] > 50.0
    )

    # Satellite-relative abnormality.
    #
    # 5x the satellite's normal median residual is treated as
    # unusually large for this baseline experiment.
    df["relative_abnormal"] = (
        df["normalized_residual"] > 5.0
    )

    # ---------------------------------------------------------------
    # Epoch-level cross-satellite analysis
    # ---------------------------------------------------------------

    epoch_rows = []

    for t, group in df.groupby("t", sort=True):

        n_satellites = len(group)

        abs_residual = group["abs_residual_m"].to_numpy()
        signed_residual = group["residual_m"].to_numpy()

        abnormal_20 = group["abnormal_20m"].to_numpy()
        abnormal_50 = group["abnormal_50m"].to_numpy()
        relative_abnormal = group[
            "relative_abnormal"
        ].to_numpy()

        n_abnormal_20 = int(abnormal_20.sum())
        n_abnormal_50 = int(abnormal_50.sum())
        n_relative_abnormal = int(
            relative_abnormal.sum()
        )

        # -----------------------------------------------------------
        # Sign consistency
        #
        # Among satellites with non-zero residuals, determine
        # whether most residuals point in the same direction.
        #
        # +1  -> positive residual
        # -1  -> negative residual
        #
        # A value close to 1 means strong same-sign agreement.
        # -----------------------------------------------------------

        positive_count = int(
            np.sum(signed_residual > 0)
        )

        negative_count = int(
            np.sum(signed_residual < 0)
        )

        total_signed = (
            positive_count
            + negative_count
        )

        if total_signed > 0:
            sign_consistency = (
                max(
                    positive_count,
                    negative_count,
                )
                / total_signed
            )
        else:
            sign_consistency = np.nan

        # -----------------------------------------------------------
        # Mean signed residual
        # -----------------------------------------------------------

        mean_signed_residual = np.mean(
            signed_residual
        )

        median_signed_residual = np.median(
            signed_residual
        )

        # -----------------------------------------------------------
        # Abnormal-satellite fractions
        # -----------------------------------------------------------

        abnormal_fraction_20 = (
            n_abnormal_20 / n_satellites
        )

        abnormal_fraction_50 = (
            n_abnormal_50 / n_satellites
        )

        relative_abnormal_fraction = (
            n_relative_abnormal
            / n_satellites
        )

        # -----------------------------------------------------------
        # Strong multi-satellite event
        #
        # At least 50% of satellites abnormal.
        # -----------------------------------------------------------

        multi_satellite_event = (
            abnormal_fraction_20 >= 0.5
        )

        # -----------------------------------------------------------
        # Strong coordinated event
        #
        # At least 50% abnormal AND strong sign agreement.
        #
        # This is NOT called spoofing.
        # It is only a candidate coordinated anomaly.
        # -----------------------------------------------------------

        coordinated_event = (
            multi_satellite_event
            and sign_consistency >= 0.75
        )

        epoch_rows.append(
            {
                "t": t,

                "n_satellites": n_satellites,

                "n_abnormal_20m": n_abnormal_20,
                "n_abnormal_50m": n_abnormal_50,
                "n_relative_abnormal": (
                    n_relative_abnormal
                ),

                "abnormal_fraction_20m": (
                    abnormal_fraction_20
                ),

                "abnormal_fraction_50m": (
                    abnormal_fraction_50
                ),

                "relative_abnormal_fraction": (
                    relative_abnormal_fraction
                ),

                "mean_abs_residual_m": (
                    np.mean(abs_residual)
                ),

                "median_abs_residual_m": (
                    np.median(abs_residual)
                ),

                "max_abs_residual_m": (
                    np.max(abs_residual)
                ),

                "mean_signed_residual_m": (
                    mean_signed_residual
                ),

                "median_signed_residual_m": (
                    median_signed_residual
                ),

                "sign_consistency": (
                    sign_consistency
                ),

                "multi_satellite_event": (
                    multi_satellite_event
                ),

                "coordinated_event": (
                    coordinated_event
                ),
            }
        )

    epoch_df = pd.DataFrame(epoch_rows)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("REAL GNSS CROSS-SATELLITE CONSISTENCY")
    print("=" * 60)

    print(
        f"Epochs analysed:       "
        f"{len(epoch_df):,}"
    )

    print(
        f"Mean satellites/epoch: "
        f"{epoch_df['n_satellites'].mean():.2f}"
    )

    print("\nAbnormal satellite behavior:")

    print(
        f"Mean >20 m satellites: "
        f"{epoch_df['n_abnormal_20m'].mean():.3f}"
    )

    print(
        f"Maximum >20 m satellites: "
        f"{epoch_df['n_abnormal_20m'].max():,}"
    )

    print(
        f"Mean >50 m satellites: "
        f"{epoch_df['n_abnormal_50m'].mean():.3f}"
    )

    print(
        f"Maximum >50 m satellites: "
        f"{epoch_df['n_abnormal_50m'].max():,}"
    )

    print("\nCross-satellite fractions:")

    print(
        f"Mean >20 m fraction: "
        f"{epoch_df['abnormal_fraction_20m'].mean():.4f}"
    )

    print(
        f"95th percentile >20 m fraction: "
        f"{epoch_df['abnormal_fraction_20m'].quantile(0.95):.4f}"
    )

    print(
        f"Maximum >20 m fraction: "
        f"{epoch_df['abnormal_fraction_20m'].max():.4f}"
    )

    print("\nSign consistency:")

    print(
        f"Mean: "
        f"{epoch_df['sign_consistency'].mean():.3f}"
    )

    print(
        f"Median: "
        f"{epoch_df['sign_consistency'].median():.3f}"
    )

    print(
        f"95th percentile: "
        f"{epoch_df['sign_consistency'].quantile(0.95):.3f}"
    )

    print("\nCandidate coordinated events:")

    print(
        f"Multi-satellite events: "
        f"{epoch_df['multi_satellite_event'].sum():,}"
    )

    print(
        f"Coordinated events: "
        f"{epoch_df['coordinated_event'].sum():,}"
    )

    # ---------------------------------------------------------------
    # Show strongest epochs
    # ---------------------------------------------------------------

    strongest = (
        epoch_df
        .sort_values(
            [
                "abnormal_fraction_20m",
                "sign_consistency",
            ],
            ascending=False,
        )
        .head(15)
    )

    print("\nStrongest cross-satellite events:")

    print(
        strongest[
            [
                "t",
                "n_satellites",
                "n_abnormal_20m",
                "abnormal_fraction_20m",
                "mean_abs_residual_m",
                "max_abs_residual_m",
                "sign_consistency",
                "multi_satellite_event",
                "coordinated_event",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Save epoch-level data
    # ---------------------------------------------------------------

    output_path = (
        output_dir
        / "real_gnss_cross_satellite.csv"
    )

    epoch_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved cross-satellite data:\n"
        f"{output_path}"
    )

    # ---------------------------------------------------------------
    # Figure 1
    #
    # Cross-satellite abnormal fraction + residual magnitude
    # ---------------------------------------------------------------

    fig, ax1 = plt.subplots(
        figsize=(16, 8)
    )

    time_minutes = (
        epoch_df["t"] / 60.0
    )

    ax1.plot(
        time_minutes,
        epoch_df["abnormal_fraction_20m"],
        label="Fraction of satellites >20 m",
    )

    ax1.set_xlabel(
        "Time (minutes)"
    )

    ax1.set_ylabel(
        "Abnormal satellite fraction"
    )

    ax1.set_ylim(
        0,
        1.05,
    )

    ax2 = ax1.twinx()

    ax2.plot(
        time_minutes,
        epoch_df["median_abs_residual_m"],
        label="Median |residual|",
    )

    ax2.set_ylabel(
        "Median |residual| (m)"
    )

    ax1.set_title(
        "Real GNSS Cross-Satellite Consistency"
    )

    fig.tight_layout()

    figure_path = (
        figure_dir
        / "real_gnss_cross_satellite.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Figure 2
    #
    # Sign consistency
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    ax.plot(
        time_minutes,
        epoch_df["sign_consistency"],
        label="Residual sign consistency",
    )

    ax.axhline(
        0.75,
        linestyle="--",
        label="Candidate threshold = 0.75",
    )

    ax.set_xlabel(
        "Time (minutes)"
    )

    ax.set_ylabel(
        "Sign consistency"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_title(
        "Real GNSS Residual Sign Consistency"
    )

    ax.legend()

    fig.tight_layout()

    sign_figure_path = (
        figure_dir
        / "real_gnss_cross_satellite_sign.png"
    )

    fig.savefig(
        sign_figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure:\n"
        f"{figure_path}"
    )

    print(
        f"Saved sign-consistency figure:\n"
        f"{sign_figure_path}"
    )


if __name__ == "__main__":
    main()