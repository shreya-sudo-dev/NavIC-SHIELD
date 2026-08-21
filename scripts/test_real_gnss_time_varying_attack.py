"""
Time-varying common pseudorange attack on real Google GNSS data.

Experiment
----------
Real GPS L1 observations
        ->
inject a common pseudorange bias that changes with time
        ->
position solver
        ->
track:

    attack bias
    receiver clock bias
    position error
    mean residual
    maximum residual

Purpose
-------
Understand how a common-mode attack appears in the receiver
clock state over time.

This is an experiment, NOT the final spoofing detector.
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
# Attack profile
# -------------------------------------------------------------------

def attack_bias(t):
    """
    Time-varying common pseudorange bias.

    0 - 600 s:
        Clean

    600 - 1200 s:
        Linear ramp: 0 -> 100 m

    1200 - 1800 s:
        Linear ramp: 100 -> 300 m

    1800 s onward:
        Constant 300 m
    """

    if t < 600:
        return 0.0

    elif t < 1200:
        return (
            (t - 600.0)
            / 600.0
            * 100.0
        )

    elif t < 1800:
        return (
            100.0
            + (t - 1200.0)
            / 600.0
            * 200.0
        )

    else:
        return 300.0


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
    # Process epochs
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

        # Ground truth
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
        # Attack
        # ------------------------------------------------------------

        bias_m = attack_bias(t)

        attacked_pseudoranges = (
            clean_pseudoranges
            + bias_m
        )

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # Start from the TRUE position for each epoch.
        #
        # This keeps the experiment focused on the measurement
        # response instead of allowing an accumulated solver error
        # to contaminate later epochs.
        # ------------------------------------------------------------

        initial_guess = (
            true_position.copy()
        )

        # ------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------

        solution = solve_position_single_epoch(
            sat_positions=sat_positions,
            pseudoranges=attacked_pseudoranges,
            initial_guess=initial_guess,
        )

        if solution["position"] is None:
            continue

        estimated_position = (
            solution["position"]
        )

        residuals = np.asarray(
            solution["residuals"],
            dtype=float,
        )

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------

        position_error_m = np.linalg.norm(
            estimated_position
            - true_position
        )

        mean_abs_residual_m = np.mean(
            np.abs(residuals)
        )

        max_abs_residual_m = np.max(
            np.abs(residuals)
        )

        results.append(
            {
                "t": t,

                "attack_bias_m": bias_m,

                "estimated_x_m":
                    estimated_position[0],

                "estimated_y_m":
                    estimated_position[1],

                "estimated_z_m":
                    estimated_position[2],

                "true_x_m":
                    true_position[0],

                "true_y_m":
                    true_position[1],

                "true_z_m":
                    true_position[2],

                "position_error_m":
                    position_error_m,

                "clock_bias_m":
                    solution["clock_bias"],

                "mean_abs_residual_m":
                    mean_abs_residual_m,

                "max_abs_residual_m":
                    max_abs_residual_m,

                "n_satellites":
                    len(group),

                "converged":
                    solution["converged"],

                "n_iterations":
                    solution["n_iterations"],
            }
        )

    # ----------------------------------------------------------------
    # Results dataframe
    # ----------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print("TIME-VARYING COMMON ATTACK")
    print("=" * 60)

    print(
        f"Solved epochs: "
        f"{len(results_df):,}"
    )

    print(
        f"Converged epochs: "
        f"{results_df['converged'].sum():,}"
        f" / {len(results_df):,}"
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------

    print("\nAttack bias:")
    print(
        f"  Minimum: "
        f"{results_df['attack_bias_m'].min():.2f} m"
    )

    print(
        f"  Maximum: "
        f"{results_df['attack_bias_m'].max():.2f} m"
    )

    print("\nPosition error:")
    print(
        f"  Mean: "
        f"{results_df['position_error_m'].mean():.2f} m"
    )

    print(
        f"  Median: "
        f"{results_df['position_error_m'].median():.2f} m"
    )

    print(
        f"  Maximum: "
        f"{results_df['position_error_m'].max():.2f} m"
    )

    print("\nClock bias:")
    print(
        f"  Minimum: "
        f"{results_df['clock_bias_m'].min():.2f} m"
    )

    print(
        f"  Maximum: "
        f"{results_df['clock_bias_m'].max():.2f} m"
    )

    print("\nMean |residual|:")
    print(
        f"  Mean: "
        f"{results_df['mean_abs_residual_m'].mean():.2f} m"
    )

    print(
        f"  Maximum: "
        f"{results_df['mean_abs_residual_m'].max():.2f} m"
    )

    # ----------------------------------------------------------------
    # Show selected checkpoints
    # ----------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SELECTED ATTACK CHECKPOINTS")
    print("=" * 60)

    checkpoint_times = [
        0,
        600,
        900,
        1200,
        1500,
        1800,
        2400,
        3000,
    ]

    for checkpoint in checkpoint_times:

        if len(results_df) == 0:
            break

        index = (
            results_df["t"]
            - checkpoint
        ).abs().idxmin()

        row = results_df.loc[index]

        print(
            f"\nt = {row['t']:.0f} s"
        )

        print(
            f"  Attack bias:     "
            f"{row['attack_bias_m']:.2f} m"
        )

        print(
            f"  Clock bias:      "
            f"{row['clock_bias_m']:.2f} m"
        )

        print(
            f"  Position error:  "
            f"{row['position_error_m']:.2f} m"
        )

        print(
            f"  Mean residual:   "
            f"{row['mean_abs_residual_m']:.2f} m"
        )

    # ----------------------------------------------------------------
    # Save CSV
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
        / "real_gnss_time_varying_attack.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved results:\n"
        f"{output_path}"
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

    time_minutes = (
        results_df["t"] / 60.0
    )

    # ---------------------------------------------------------------
    # Figure 1: Attack bias vs clock bias
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        time_minutes,
        results_df["attack_bias_m"],
        label="Injected attack bias",
    )

    ax.plot(
        time_minutes,
        results_df["clock_bias_m"],
        label="Estimated receiver clock bias",
    )

    ax.set_xlabel(
        "Time (minutes)"
    )

    ax.set_ylabel(
        "Bias (m)"
    )

    ax.set_title(
        "Common GNSS Attack vs Receiver Clock Bias"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    figure_path = (
        figure_dir
        / "real_gnss_time_varying_clock_bias.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure:\n"
        f"{figure_path}"
    )

    # ---------------------------------------------------------------
    # Figure 2: Attack bias vs position error
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        time_minutes,
        results_df["attack_bias_m"],
        label="Injected attack bias",
    )

    ax.plot(
        time_minutes,
        results_df["position_error_m"],
        label="Position error",
    )

    ax.set_xlabel(
        "Time (minutes)"
    )

    ax.set_ylabel(
        "Distance (m)"
    )

    ax.set_title(
        "Common GNSS Attack vs Position Error"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    figure_path = (
        figure_dir
        / "real_gnss_time_varying_position_error.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure:\n"
        f"{figure_path}"
    )

    # ---------------------------------------------------------------
    # Figure 3: Residual response
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        time_minutes,
        results_df["mean_abs_residual_m"],
        label="Mean |residual|",
    )

    ax.plot(
        time_minutes,
        results_df["max_abs_residual_m"],
        label="Maximum |residual|",
    )

    ax.set_xlabel(
        "Time (minutes)"
    )

    ax.set_ylabel(
        "Residual (m)"
    )

    ax.set_title(
        "Residual Response to Common GNSS Attack"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    figure_path = (
        figure_dir
        / "real_gnss_time_varying_residual_response.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure:\n"
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()