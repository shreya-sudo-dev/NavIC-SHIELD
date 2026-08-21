"""
Compare clean and attacked GNSS solutions on real Google GNSS data.

Experiment
----------
Real GPS L1 observations
        |
        +---- clean pseudorange ----> position solver
        |
        +---- attacked pseudorange -> position solver
                                      |
                                      v
                              compare the two

For each epoch and attack level we calculate:

    clean position
    attacked position
    clean clock bias
    attacked clock bias

    delta clock bias
    delta position
    delta mean residual
    delta max residual

Purpose
-------
Determine whether a common-mode pseudorange attack is primarily
absorbed by the receiver clock state.

This is a controlled experiment, not the final attack detector.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Project setup
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navigation.position_solver import solve_position_single_epoch


# -------------------------------------------------------------------
# Attack levels
# -------------------------------------------------------------------

ATTACK_BIASES = [
    0.0,
    10.0,
    50.0,
    100.0,
    500.0,
    1000.0,
]


# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------

def solve_epoch(
    sat_positions,
    pseudoranges,
    initial_guess,
):
    """
    Solve one GNSS epoch.
    """

    return solve_position_single_epoch(
        sat_positions=sat_positions,
        pseudoranges=pseudoranges,
        initial_guess=initial_guess.copy(),
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------

    data_path = (
        PROJECT_ROOT
        / "data"
        / "real_gnss"
        / "google"
        / "sample"
        / "normalized_gnss.csv"
    )

    print(
        f"Loading:\n{data_path}"
    )

    df = pd.read_csv(data_path)

    # ----------------------------------------------------------------
    # GPS L1 only
    # ----------------------------------------------------------------

    gps = df[
        (df["ConstellationType"] == 1)
        & (df["SignalType"] == "GPS_L1")
    ].copy()

    print(
        f"\nGPS L1 observations: "
        f"{len(gps):,}"
    )

    print(
        f"GPS L1 epochs:       "
        f"{gps['t'].nunique():,}"
    )

    # ----------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------

    results = []

    # ----------------------------------------------------------------
    # Process every epoch
    # ----------------------------------------------------------------

    for t, group in gps.groupby(
        "t",
        sort=True,
    ):

        if len(group) < 4:
            continue

        sat_positions = group[
            ["x", "y", "z"]
        ].to_numpy(
            dtype=float
        )

        clean_pseudoranges = group[
            "pseudorange_m"
        ].to_numpy(
            dtype=float
        )

        # ------------------------------------------------------------
        # Ground truth
        # ------------------------------------------------------------

        true_position = np.array(
            [
                group[
                    "receiver_x_true_m"
                ].iloc[0],

                group[
                    "receiver_y_true_m"
                ].iloc[0],

                group[
                    "receiver_z_true_m"
                ].iloc[0],
            ],
            dtype=float,
        )

        # ------------------------------------------------------------
        # Use true receiver position as the initial guess.
        #
        # We deliberately use the SAME initial guess for clean and
        # attacked solutions so the comparison isolates the effect
        # of the injected measurement bias.
        # ------------------------------------------------------------

        initial_guess = true_position.copy()

        # ------------------------------------------------------------
        # CLEAN SOLUTION
        # ------------------------------------------------------------

        clean_solution = solve_epoch(
            sat_positions=sat_positions,
            pseudoranges=clean_pseudoranges,
            initial_guess=initial_guess,
        )

        if clean_solution["position"] is None:
            continue

        clean_position = np.asarray(
            clean_solution["position"],
            dtype=float,
        )

        clean_clock_bias = float(
            clean_solution["clock_bias"]
        )

        clean_residuals = np.asarray(
            clean_solution["residuals"],
            dtype=float,
        )

        clean_mean_abs_residual = float(
            np.mean(
                np.abs(clean_residuals)
            )
        )

        clean_max_abs_residual = float(
            np.max(
                np.abs(clean_residuals)
            )
        )

        clean_position_error = float(
            np.linalg.norm(
                clean_position
                - true_position
            )
        )

        # ------------------------------------------------------------
        # ATTACK EXPERIMENTS
        # ------------------------------------------------------------

        for attack_bias in ATTACK_BIASES:

            attacked_pseudoranges = (
                clean_pseudoranges
                + attack_bias
            )

            attacked_solution = solve_epoch(
                sat_positions=sat_positions,
                pseudoranges=attacked_pseudoranges,
                initial_guess=initial_guess,
            )

            if attacked_solution["position"] is None:
                continue

            attacked_position = np.asarray(
                attacked_solution["position"],
                dtype=float,
            )

            attacked_clock_bias = float(
                attacked_solution["clock_bias"]
            )

            attacked_residuals = np.asarray(
                attacked_solution["residuals"],
                dtype=float,
            )

            attacked_mean_abs_residual = float(
                np.mean(
                    np.abs(attacked_residuals)
                )
            )

            attacked_max_abs_residual = float(
                np.max(
                    np.abs(attacked_residuals)
                )
            )

            attacked_position_error = float(
                np.linalg.norm(
                    attacked_position
                    - true_position
                )
            )

            # --------------------------------------------------------
            # DELTAS
            # --------------------------------------------------------

            delta_position_vector = (
                attacked_position
                - clean_position
            )

            delta_position_m = float(
                np.linalg.norm(
                    delta_position_vector
                )
            )

            delta_clock_bias_m = (
                attacked_clock_bias
                - clean_clock_bias
            )

            delta_mean_abs_residual_m = (
                attacked_mean_abs_residual
                - clean_mean_abs_residual
            )

            delta_max_abs_residual_m = (
                attacked_max_abs_residual
                - clean_max_abs_residual
            )

            # --------------------------------------------------------
            # Save
            # --------------------------------------------------------

            results.append(
                {
                    "t": t,

                    "attack_bias_m":
                        attack_bias,

                    # Clean
                    "clean_x_m":
                        clean_position[0],

                    "clean_y_m":
                        clean_position[1],

                    "clean_z_m":
                        clean_position[2],

                    "clean_clock_bias_m":
                        clean_clock_bias,

                    "clean_position_error_m":
                        clean_position_error,

                    "clean_mean_abs_residual_m":
                        clean_mean_abs_residual,

                    "clean_max_abs_residual_m":
                        clean_max_abs_residual,

                    # Attacked
                    "attacked_x_m":
                        attacked_position[0],

                    "attacked_y_m":
                        attacked_position[1],

                    "attacked_z_m":
                        attacked_position[2],

                    "attacked_clock_bias_m":
                        attacked_clock_bias,

                    "attacked_position_error_m":
                        attacked_position_error,

                    "attacked_mean_abs_residual_m":
                        attacked_mean_abs_residual,

                    "attacked_max_abs_residual_m":
                        attacked_max_abs_residual,

                    # Differences
                    "delta_x_m":
                        delta_position_vector[0],

                    "delta_y_m":
                        delta_position_vector[1],

                    "delta_z_m":
                        delta_position_vector[2],

                    "delta_position_m":
                        delta_position_m,

                    "delta_clock_bias_m":
                        delta_clock_bias_m,

                    "delta_mean_abs_residual_m":
                        delta_mean_abs_residual_m,

                    "delta_max_abs_residual_m":
                        delta_max_abs_residual_m,

                    "n_satellites":
                        len(group),

                    "clean_converged":
                        clean_solution["converged"],

                    "attacked_converged":
                        attacked_solution["converged"],
                }
            )

    # ----------------------------------------------------------------
    # Dataframe
    # ----------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        print("\nNo successful solutions.")
        return

    print("\n" + "=" * 65)
    print("REAL GNSS CLEAN vs ATTACK DELTA ANALYSIS")
    print("=" * 65)

    print(
        f"Rows:              "
        f"{len(results_df):,}"
    )

    print(
        f"Epochs:            "
        f"{results_df['t'].nunique():,}"
    )

    print(
        f"Attack levels:     "
        f"{results_df['attack_bias_m'].nunique()}"
    )

    # ----------------------------------------------------------------
    # Summary by attack level
    # ----------------------------------------------------------------

    summary = (
        results_df
        .groupby("attack_bias_m")
        .agg(
            mean_delta_clock_m=(
                "delta_clock_bias_m",
                "mean",
            ),

            median_delta_clock_m=(
                "delta_clock_bias_m",
                "median",
            ),

            mean_delta_position_m=(
                "delta_position_m",
                "mean",
            ),

            median_delta_position_m=(
                "delta_position_m",
                "median",
            ),

            p95_delta_position_m=(
                "delta_position_m",
                lambda x: np.percentile(
                    x,
                    95,
                ),
            ),

            mean_delta_residual_m=(
                "delta_mean_abs_residual_m",
                "mean",
            ),

            median_delta_residual_m=(
                "delta_mean_abs_residual_m",
                "median",
            ),

            max_delta_residual_m=(
                "delta_max_abs_residual_m",
                "max",
            ),
        )
        .reset_index()
    )

    print("\n" + "=" * 65)
    print("ATTACK-LEVEL SUMMARY")
    print("=" * 65)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # ----------------------------------------------------------------
    # Save raw results
    # ----------------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "csv"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "real_gnss_attack_delta.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved detailed results:\n"
        f"{output_path}"
    )

    # ----------------------------------------------------------------
    # Save summary
    # ----------------------------------------------------------------

    summary_path = (
        output_dir
        / "real_gnss_attack_delta_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"Saved summary:\n"
        f"{summary_path}"
    )

    # ----------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------

    figure_dir = (
        PROJECT_ROOT
        / "results"
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ================================================================
    # Figure 1
    # Attack bias vs delta clock bias
    # ================================================================

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.plot(
        summary["attack_bias_m"],
        summary["mean_delta_clock_m"],
        marker="o",
        label="Mean Δ clock bias",
    )

    ax.plot(
        summary["attack_bias_m"],
        summary["attack_bias_m"],
        linestyle="--",
        label="Ideal Δ clock = attack bias",
    )

    ax.set_xlabel(
        "Injected common bias (m)"
    )

    ax.set_ylabel(
        "Change in receiver clock bias (m)"
    )

    ax.set_title(
        "Common Attack vs Change in Receiver Clock Bias"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        figure_dir
        / "real_gnss_attack_delta_clock.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure:\n{path}"
    )

    # ================================================================
    # Figure 2
    # Attack bias vs delta position
    # ================================================================

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.plot(
        summary["attack_bias_m"],
        summary["mean_delta_position_m"],
        marker="o",
        label="Mean Δ position",
    )

    ax.plot(
        summary["attack_bias_m"],
        summary["p95_delta_position_m"],
        marker="o",
        label="95th percentile Δ position",
    )

    ax.set_xlabel(
        "Injected common bias (m)"
    )

    ax.set_ylabel(
        "Change in position (m)"
    )

    ax.set_title(
        "Common Attack vs Change in Position"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        figure_dir
        / "real_gnss_attack_delta_position.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure:\n{path}"
    )

    # ================================================================
    # Figure 3
    # Attack bias vs delta residual
    # ================================================================

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.plot(
        summary["attack_bias_m"],
        summary["mean_delta_residual_m"],
        marker="o",
        label="Mean Δ |residual|",
    )

    ax.plot(
        summary["attack_bias_m"],
        summary["max_delta_residual_m"],
        marker="o",
        label="Maximum Δ |residual|",
    )

    ax.axhline(
        0.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Injected common bias (m)"
    )

    ax.set_ylabel(
        "Change in residual (m)"
    )

    ax.set_title(
        "Common Attack vs Change in Residuals"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        figure_dir
        / "real_gnss_attack_delta_residual.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure:\n{path}"
    )

    # ================================================================
    # Figure 4
    # Time response for the largest attack
    # ================================================================

    largest_attack = (
        results_df["attack_bias_m"].max()
    )

    largest = results_df[
        results_df["attack_bias_m"]
        == largest_attack
    ].copy()

    largest["time_min"] = (
        largest["t"] / 60.0
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        largest["time_min"],
        largest["delta_clock_bias_m"],
        label="Δ clock bias",
    )

    ax.axhline(
        largest_attack,
        linestyle="--",
        label="Injected bias",
    )

    ax.set_xlabel(
        "Time (minutes)"
    )

    ax.set_ylabel(
        "Clock bias change (m)"
    )

    ax.set_title(
        f"Temporal Response to +{largest_attack:.0f} m "
        "Common Attack"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        figure_dir
        / "real_gnss_attack_delta_clock_time.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure:\n{path}"
    )


if __name__ == "__main__":
    main()