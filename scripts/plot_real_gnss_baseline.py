"""
Plot real-GNSS residual baseline for NavIC-SHIELD.

Inputs:
    results/csv/real_gnss_satellite_baseline.csv
    results/csv/real_gnss_temporal_baseline.csv
    results/csv/real_gnss_satellite_residuals.csv

Outputs:
    results/figures/real_gnss_satellite_residual_baseline.png
    results/figures/real_gnss_outlier_rates.png
    results/figures/real_gnss_temporal_persistence.png
    results/figures/real_gnss_residual_distribution.png
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = PROJECT_ROOT / "results"
CSV_DIR = RESULTS_DIR / "csv"
FIGURE_DIR = RESULTS_DIR / "figures"

SATELLITE_BASELINE = (
    CSV_DIR / "real_gnss_satellite_baseline.csv"
)

TEMPORAL_BASELINE = (
    CSV_DIR / "real_gnss_temporal_baseline.csv"
)

RESIDUAL_DATA = (
    CSV_DIR / "real_gnss_satellite_residuals.csv"
)


def check_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


def main():

    print("=" * 60)
    print("REAL GNSS BASELINE VISUALIZATION")
    print("=" * 60)

    check_file(SATELLITE_BASELINE)
    check_file(TEMPORAL_BASELINE)
    check_file(RESIDUAL_DATA)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    satellite = pd.read_csv(
        SATELLITE_BASELINE
    )

    temporal = pd.read_csv(
        TEMPORAL_BASELINE
    )

    residuals = pd.read_csv(
        RESIDUAL_DATA
    )

    print(
        f"\nSatellite baseline rows: "
        f"{len(satellite)}"
    )

    print(
        f"Temporal baseline rows: "
        f"{len(temporal)}"
    )

    print(
        f"Residual observations: "
        f"{len(residuals):,}"
    )

    # ================================================================
    # 1. Satellite residual baseline
    # ================================================================

    plot_data = satellite.sort_values(
        "mean_abs_residual_m",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    x = np.arange(len(plot_data))

    ax.plot(
        x,
        plot_data["median_abs_residual_m"],
        marker="o",
        linewidth=2,
        label="Median |residual|",
    )

    ax.plot(
        x,
        plot_data["mean_abs_residual_m"],
        marker="o",
        linewidth=2,
        label="Mean |residual|",
    )

    ax.plot(
        x,
        plot_data["p95_abs_residual_m"],
        marker="o",
        linewidth=2,
        label="P95 |residual|",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_data["satellite_id"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Residual magnitude (m)"
    )

    ax.set_xlabel(
        "Satellite"
    )

    ax.set_title(
        "Real GNSS Satellite Residual Baseline"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        FIGURE_DIR
        / "real_gnss_satellite_residual_baseline.png"
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved:\n{output}"
    )

    # ================================================================
    # 2. Natural outlier rates
    # ================================================================

    plot_data = satellite.sort_values(
        "outlier_rate_20m_pct",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    x = np.arange(len(plot_data))
    width = 0.36

    ax.bar(
        x - width / 2,
        plot_data["outlier_rate_20m_pct"],
        width,
        label=">20 m",
    )

    ax.bar(
        x + width / 2,
        plot_data["outlier_rate_50m_pct"],
        width,
        label=">50 m",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_data["satellite_id"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Outlier rate (%)"
    )

    ax.set_xlabel(
        "Satellite"
    )

    ax.set_title(
        "Natural Real-GNSS Residual Outlier Rates"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        FIGURE_DIR
        / "real_gnss_outlier_rates.png"
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved:\n{output}"
    )

    # ================================================================
    # 3. Temporal persistence
    # ================================================================

    plot_data = temporal.sort_values(
        "longest_20m_run_epochs",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    x = np.arange(len(plot_data))

    ax.bar(
        x,
        plot_data["longest_20m_run_epochs"],
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_data["satellite_id"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Longest consecutive epochs"
    )

    ax.set_xlabel(
        "Satellite"
    )

    ax.set_title(
        "Temporal Persistence of >20 m Residuals"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output = (
        FIGURE_DIR
        / "real_gnss_temporal_persistence.png"
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved:\n{output}"
    )

    # ================================================================
    # 4. Global residual distribution
    # ================================================================

    values = (
        residuals["abs_residual_m"]
        .dropna()
        .to_numpy()
    )

    p95 = np.percentile(
        values,
        95,
    )

    p99 = np.percentile(
        values,
        99,
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    # Limit the histogram to the useful bulk of the
    # distribution so that the natural tail remains visible.
    histogram_max = min(
        np.percentile(values, 99.9),
        100.0,
    )

    histogram_values = values[
        values <= histogram_max
    ]

    ax.hist(
        histogram_values,
        bins=50,
        alpha=0.8,
    )

    ax.axvline(
        p95,
        linestyle="--",
        linewidth=2,
        label=f"P95 = {p95:.2f} m",
    )

    ax.axvline(
        p99,
        linestyle="--",
        linewidth=2,
        label=f"P99 = {p99:.2f} m",
    )

    ax.axvline(
        20,
        linestyle=":",
        linewidth=2,
        label="20 m threshold",
    )

    ax.axvline(
        50,
        linestyle=":",
        linewidth=2,
        label="50 m threshold",
    )

    ax.set_xlabel(
        "Absolute residual (m)"
    )

    ax.set_ylabel(
        "Observation count"
    )

    ax.set_title(
        "Real GNSS Absolute Residual Distribution"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        FIGURE_DIR
        / "real_gnss_residual_distribution.png"
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved:\n{output}"
    )

    # ================================================================
    # Final summary
    # ================================================================

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

    print(
        "\nGenerated figures:"
    )

    print(
        "1. real_gnss_satellite_residual_baseline.png"
    )

    print(
        "2. real_gnss_outlier_rates.png"
    )

    print(
        "3. real_gnss_temporal_persistence.png"
    )

    print(
        "4. real_gnss_residual_distribution.png"
    )


if __name__ == "__main__":
    main()