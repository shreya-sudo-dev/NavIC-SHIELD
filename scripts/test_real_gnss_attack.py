"""
Controlled pseudorange attack experiment on real Google GNSS data.

Experiment
----------
Take one real GPS L1 epoch and progressively inject the SAME
pseudorange bias into every satellite:

    0 m
    10 m
    50 m
    100 m
    500 m
    1000 m

Then solve the receiver position and measure:

    - position error
    - receiver clock bias
    - mean absolute residual
    - maximum absolute residual

Purpose
-------
Establish a controlled baseline for later spoofing/attack experiments.

This is NOT an attack detector yet.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Project setup
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navigation.position_solver import solve_position_single_epoch


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BIAS_LEVELS_M = [
    0,
    10,
    50,
    100,
    500,
    1000,
]

# Use the first complete GPS L1 epoch.
TEST_EPOCH = 0.0


def main():

    # ----------------------------------------------------------------
    # Load normalized real GNSS data
    # ----------------------------------------------------------------

    data_path = (
        PROJECT_ROOT
        / "data"
        / "real_gnss"
        / "google"
        / "sample"
        / "normalized_gnss.csv"
    )

    print(f"Loading: {data_path}")

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
    # Select test epoch
    # ----------------------------------------------------------------

    group = gps[
        gps["t"] == TEST_EPOCH
    ].copy()

    if len(group) < 4:

        raise ValueError(
            f"Epoch {TEST_EPOCH} has only "
            f"{len(group)} satellites. "
            f"At least 4 are required."
        )

    print(
        f"\nTest epoch:          "
        f"{TEST_EPOCH:.1f} s"
    )

    print(
        f"Satellites:          "
        f"{len(group)}"
    )

    # ----------------------------------------------------------------
    # Satellite positions
    # ----------------------------------------------------------------

    sat_positions = group[
        ["x", "y", "z"]
    ].to_numpy(
        dtype=float
    )

    # ----------------------------------------------------------------
    # Clean pseudoranges
    # ----------------------------------------------------------------

    clean_pseudoranges = group[
        "pseudorange_m"
    ].to_numpy(
        dtype=float
    )

    # ----------------------------------------------------------------
    # Ground-truth receiver position
    # ----------------------------------------------------------------

    true_position = np.array(
        [
            group["receiver_x_true_m"].iloc[0],
            group["receiver_y_true_m"].iloc[0],
            group["receiver_z_true_m"].iloc[0],
        ],
        dtype=float,
    )

    # ----------------------------------------------------------------
    # Initial position estimate
    #
    # Same strategy as the clean real-GNSS experiment:
    # use the true position ONLY as the initial numerical guess.
    #
    # It is NOT used to calculate the attacked pseudoranges.
    # ----------------------------------------------------------------

    initial_guess = true_position.copy()

    # ----------------------------------------------------------------
    # Display satellites
    # ----------------------------------------------------------------

    print("\nSatellites used:")

    print(
        group[
            [
                "Svid",
                "satellite_id",
                "cn0_db_hz",
                "elevation_deg",
                "pseudorange_m",
            ]
        ].to_string(index=False)
    )

    # ----------------------------------------------------------------
    # Run controlled attack experiment
    # ----------------------------------------------------------------

    results = []

    print("\n" + "=" * 60)
    print("CONTROLLED PSEUDORANGE ATTACK EXPERIMENT")
    print("=" * 60)

    for bias_m in BIAS_LEVELS_M:

        # ------------------------------------------------------------
        # Inject common pseudorange bias
        #
        # Positive bias means every satellite appears farther away
        # by the same amount.
        # ------------------------------------------------------------

        attacked_pseudoranges = (
            clean_pseudoranges + bias_m
        )

        # ------------------------------------------------------------
        # Solve position
        # ------------------------------------------------------------

        solution = solve_position_single_epoch(
            sat_positions=sat_positions,
            pseudoranges=attacked_pseudoranges,
            initial_guess=initial_guess,
        )

        # ------------------------------------------------------------
        # Check solver result
        # ------------------------------------------------------------

        if solution["position"] is None:

            print(
                f"\nBias {bias_m:>6.1f} m -> "
                f"solver failed"
            )

            results.append(
                {
                    "attack_bias_m": bias_m,
                    "position_error_m": np.nan,
                    "clock_bias_m": np.nan,
                    "mean_abs_residual_m": np.nan,
                    "max_abs_residual_m": np.nan,
                    "converged": False,
                    "n_iterations": solution.get(
                        "n_iterations",
                        np.nan,
                    ),
                    "n_satellites": len(group),
                }
            )

            continue

        estimated_position = solution[
            "position"
        ]

        # ------------------------------------------------------------
        # Position error
        # ------------------------------------------------------------

        position_error_m = np.linalg.norm(
            estimated_position
            - true_position
        )

        # ------------------------------------------------------------
        # Residual statistics
        # ------------------------------------------------------------

        residuals = np.asarray(
            solution["residuals"],
            dtype=float,
        )

        mean_abs_residual_m = np.mean(
            np.abs(residuals)
        )

        max_abs_residual_m = np.max(
            np.abs(residuals)
        )

        # ------------------------------------------------------------
        # Store result
        # ------------------------------------------------------------

        results.append(
            {
                "attack_bias_m": bias_m,

                "estimated_x_m": (
                    estimated_position[0]
                ),

                "estimated_y_m": (
                    estimated_position[1]
                ),

                "estimated_z_m": (
                    estimated_position[2]
                ),

                "position_error_m": (
                    position_error_m
                ),

                "clock_bias_m": (
                    solution["clock_bias"]
                ),

                "mean_abs_residual_m": (
                    mean_abs_residual_m
                ),

                "max_abs_residual_m": (
                    max_abs_residual_m
                ),

                "converged": (
                    solution["converged"]
                ),

                "n_iterations": (
                    solution["n_iterations"]
                ),

                "n_satellites": len(group),
            }
        )

        print(
            f"\nAttack bias:          "
            f"{bias_m:8.1f} m"
        )

        print(
            f"Position error:       "
            f"{position_error_m:8.3f} m"
        )

        print(
            f"Clock bias:           "
            f"{solution['clock_bias']:8.3f} m"
        )

        print(
            f"Mean |residual|:      "
            f"{mean_abs_residual_m:8.3f} m"
        )

        print(
            f"Max |residual|:       "
            f"{max_abs_residual_m:8.3f} m"
        )

        print(
            f"Iterations:           "
            f"{solution['n_iterations']}"
        )

        print(
            f"Converged:            "
            f"{solution['converged']}"
        )

    # ----------------------------------------------------------------
    # Results dataframe
    # ----------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # ----------------------------------------------------------------
    # Print final table
    # ----------------------------------------------------------------

    print("\n" + "=" * 60)
    print("ATTACK EXPERIMENT SUMMARY")
    print("=" * 60)

    print(
        results_df[
            [
                "attack_bias_m",
                "position_error_m",
                "clock_bias_m",
                "mean_abs_residual_m",
                "max_abs_residual_m",
                "converged",
            ]
        ].to_string(index=False)
    )

    # ----------------------------------------------------------------
    # Save results
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
        / "real_gnss_attack_single_epoch.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved attack experiment results to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()