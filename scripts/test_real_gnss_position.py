"""
Test NavIC-SHIELD position solver on real Google GNSS data.

First real-data navigation experiment:
    Google GPS_L1 observations
        -> existing position solver
        -> compare solved ECEF with ground truth
        -> save per-satellite post-fit residuals
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navigation.position_solver import solve_position_single_epoch


def main():

    project_root = Path(__file__).resolve().parents[1]

    data_path = (
        project_root
        / "data"
        / "real_gnss"
        / "google"
        / "sample"
        / "normalized_gnss.csv"
    )

    print(f"Loading: {data_path}")

    df = pd.read_csv(data_path)

    # ---------------------------------------------------------------
    # First experiment: GPS L1 only
    # ---------------------------------------------------------------

    gps = df[
        (df["ConstellationType"] == 1)
        & (df["SignalType"] == "GPS_L1")
    ].copy()

    print(f"\nGPS L1 observations: {len(gps):,}")
    print(f"GPS L1 epochs:       {gps['t'].nunique():,}")

    # ---------------------------------------------------------------
    # Initial receiver position
    #
    # Use ground-truth position only to initialize the first solve.
    # The solver itself estimates position and receiver clock bias.
    # ---------------------------------------------------------------

    first = gps.iloc[0]

    initial_guess = np.array([
        first["receiver_x_true_m"],
        first["receiver_y_true_m"],
        first["receiver_z_true_m"],
    ])

    errors = []
    solved_rows = []
    residual_rows = []

    # ---------------------------------------------------------------
    # Solve every epoch
    # ---------------------------------------------------------------

    for t, group in gps.groupby("t", sort=True):

        if len(group) < 4:
            continue

        sat_positions = group[
            ["x", "y", "z"]
        ].to_numpy()

        pseudoranges = group[
            "pseudorange_m"
        ].to_numpy()

        solution = solve_position_single_epoch(
            sat_positions=sat_positions,
            pseudoranges=pseudoranges,
            initial_guess=initial_guess,
        )

        if solution["position"] is None:
            continue

        estimated = solution["position"]

        # -----------------------------------------------------------
        # Ground truth for this epoch
        # -----------------------------------------------------------

        true_position = np.array([
            group["receiver_x_true_m"].iloc[0],
            group["receiver_y_true_m"].iloc[0],
            group["receiver_z_true_m"].iloc[0],
        ])

        error_m = np.linalg.norm(
            estimated - true_position
        )

        errors.append(error_m)

        # -----------------------------------------------------------
        # Save epoch-level position result
        # -----------------------------------------------------------

        solved_rows.append({
            "t": t,

            "solved_x": estimated[0],
            "solved_y": estimated[1],
            "solved_z": estimated[2],

            "true_x": true_position[0],
            "true_y": true_position[1],
            "true_z": true_position[2],

            "position_error_m": error_m,

            # Receiver clock bias estimated by the solver.
            "clock_bias_m": solution["clock_bias"],

            "n_satellites": len(group),

            "converged": solution["converged"],

            "mean_abs_residual_m": np.mean(
                np.abs(solution["residuals"])
            ),
        })

        # -----------------------------------------------------------
        # Save individual satellite post-fit residuals
        #
        # solution["residuals"] corresponds to the satellites in
        # the same order as sat_positions / pseudoranges.
        # -----------------------------------------------------------

        for idx, (_, sat_row) in enumerate(group.iterrows()):

            residual_m = float(
                solution["residuals"][idx]
            )

            residual_rows.append({
                "t": t,

                "satellite_id": sat_row["satellite_id"],
                "Svid": sat_row["Svid"],

                "residual_m": residual_m,
                "abs_residual_m": abs(residual_m),

                "cn0_db_hz": sat_row["cn0_db_hz"],
                "elevation_deg": sat_row["elevation_deg"],
            })

        # Carry the previous solution into the next epoch.
        initial_guess = estimated

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    results = pd.DataFrame(solved_rows)

    residual_results = pd.DataFrame(residual_rows)

    print("\n" + "=" * 60)
    print("REAL GNSS POSITION SOLVER TEST")
    print("=" * 60)

    print(f"Solved epochs:       {len(results):,}")

    if len(results) == 0:
        print("\nNo epochs were successfully solved.")
        return

    print(
        f"Converged epochs:    "
        f"{results['converged'].sum():,} / {len(results):,}"
    )

    print("\nPosition error:")

    print(
        f"  Mean:              "
        f"{results['position_error_m'].mean():.2f} m"
    )

    print(
        f"  Median:            "
        f"{results['position_error_m'].median():.2f} m"
    )

    print(
        f"  95th percentile:   "
        f"{results['position_error_m'].quantile(0.95):.2f} m"
    )

    print(
        f"  Maximum:            "
        f"{results['position_error_m'].max():.2f} m"
    )

    print("\nSolver residual:")

    print(
        f"  Mean |residual|:    "
        f"{results['mean_abs_residual_m'].mean():.2f} m"
    )

    print("\nReceiver clock bias:")

    print(
        f"  Mean:              "
        f"{results['clock_bias_m'].mean():.2f} m"
    )

    print(
        f"  Median:            "
        f"{results['clock_bias_m'].median():.2f} m"
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    output_dir = (
        project_root
        / "results"
        / "csv"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Epoch-level position results
    output_path = (
        output_dir
        / "real_gnss_position_test.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # Per-satellite post-fit residuals
    residual_output_path = (
        output_dir
        / "real_gnss_satellite_residuals.csv"
    )

    residual_results.to_csv(
        residual_output_path,
        index=False,
    )

    print(
        f"\nSaved results to:\n"
        f"{output_path}"
    )

    print(
        f"\nSaved satellite residuals to:\n"
        f"{residual_output_path}"
    )


if __name__ == "__main__":
    main()