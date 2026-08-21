"""
Test selective / satellite-specific pseudorange attacks
on real Google GNSS data.

Experiment:

    Real GPS L1 observations
            |
            +----> Clean position solution
            |
            +----> Selective pseudorange attack
                       |
                       +----> Attacked position solution
            |
            v
    Compare clean vs attacked

Unlike a common-mode attack, only selected satellites receive
the injected pseudorange bias. This should prevent the receiver
clock term from completely absorbing the attack.

Outputs:
    results/csv/real_gnss_selective_attack.csv
    results/csv/real_gnss_selective_attack_summary.csv

Figures:
    results/figures/real_gnss_selective_attack_position.png
    results/figures/real_gnss_selective_attack_clock.png
    results/figures/real_gnss_selective_attack_residual.png
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navigation.position_solver import solve_position_single_epoch


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ATTACK_BIASES_M = [20.0, 50.0, 100.0, 500.0]

# Number of satellites to attack.
ATTACK_COUNTS = [1, 2, 3]

# Fractional attack experiment.
# 50% means approximately half of the available baseline satellites.
INCLUDE_HALF_ATTACK = True

MIN_OBSERVATIONS_PER_SATELLITE = 100


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def solve_epoch(
    group,
    pseudoranges,
    initial_guess,
):
    """
    Solve one GNSS epoch.
    """

    sat_positions = group[
        ["x", "y", "z"]
    ].to_numpy(dtype=float)

    solution = solve_position_single_epoch(
        sat_positions=sat_positions,
        pseudoranges=np.asarray(pseudoranges, dtype=float),
        initial_guess=initial_guess,
    )

    return solution


def get_true_position(group):
    """
    Get receiver ground-truth ECEF position.
    """

    return np.array(
        [
            group["receiver_x_true_m"].iloc[0],
            group["receiver_y_true_m"].iloc[0],
            group["receiver_z_true_m"].iloc[0],
        ],
        dtype=float,
    )


def choose_attack_satellites(gps):
    """
    Select satellites with enough observations to form a stable
    selective-attack experiment.

    Satellites are ranked by observation count.
    """

    counts = (
        gps.groupby("satellite_id")
        .size()
        .sort_values(ascending=False)
    )

    candidates = counts[
        counts >= MIN_OBSERVATIONS_PER_SATELLITE
    ]

    satellites = list(candidates.index)

    return satellites


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

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

    print(f"Loading:")
    print(data_path)

    df = pd.read_csv(data_path)

    # ---------------------------------------------------------------
    # GPS L1 only
    # ---------------------------------------------------------------

    gps = df[
        (df["ConstellationType"] == 1)
        & (df["SignalType"] == "GPS_L1")
    ].copy()

    gps = gps.sort_values("t")

    print(
        f"\nGPS L1 observations: {len(gps):,}"
    )

    print(
        f"GPS L1 epochs:       {gps['t'].nunique():,}"
    )

    # ---------------------------------------------------------------
    # Select stable satellites
    # ---------------------------------------------------------------

    candidate_satellites = choose_attack_satellites(gps)

    print(
        "\nSatellites eligible for attack:"
    )

    for sat in candidate_satellites:
        n = int(
            (gps["satellite_id"] == sat).sum()
        )
        print(f"  {sat}: {n} observations")

    if len(candidate_satellites) < 3:

        raise RuntimeError(
            "Not enough stable satellites for selective attack."
        )

    # ---------------------------------------------------------------
    # Build attack configurations
    # ---------------------------------------------------------------

    attack_configs = []

    max_count = min(
        max(ATTACK_COUNTS),
        len(candidate_satellites),
    )

    for n_attack in ATTACK_COUNTS:

        if n_attack > len(candidate_satellites):
            continue

        attacked = candidate_satellites[
            :n_attack
        ]

        attack_configs.append(
            (
                f"{n_attack}_satellites",
                attacked,
            )
        )

    if INCLUDE_HALF_ATTACK:

        half_count = max(
            1,
            len(candidate_satellites) // 2,
        )

        attacked = candidate_satellites[
            :half_count
        ]

        attack_configs.append(
            (
                "50_percent",
                attacked,
            )
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTIVE ATTACK CONFIGURATIONS"
    )

    print(
        "=" * 70
    )

    for name, satellites in attack_configs:

        print(
            f"{name:15s}: "
            f"{len(satellites)} satellites"
        )

        print(
            " " * 17
            + ", ".join(satellites)
        )

    # ---------------------------------------------------------------
    # Initial receiver position
    # ---------------------------------------------------------------

    first = gps.iloc[0]

    initial_guess_clean = np.array(
        [
            first["receiver_x_true_m"],
            first["receiver_y_true_m"],
            first["receiver_z_true_m"],
        ],
        dtype=float,
    )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    detailed_rows = []

    # ---------------------------------------------------------------
    # Run each attack configuration
    # ---------------------------------------------------------------

    for attack_name, attacked_satellites in attack_configs:

        attacked_set = set(
            attacked_satellites
        )

        for attack_bias in ATTACK_BIASES_M:

            print(
                "\n" + "=" * 70
            )

            print(
                f"Attack configuration: {attack_name}"
            )

            print(
                f"Attack bias:          {attack_bias:.1f} m"
            )

            print(
                f"Attacked satellites:  "
                f"{len(attacked_satellites)}"
            )

            print(
                "=" * 70
            )

            # Separate initial guesses.
            #
            # Both start from the same true receiver position so that
            # the comparison measures the effect of the attack rather
            # than different initialization.
            clean_initial_guess = (
                initial_guess_clean.copy()
            )

            attack_initial_guess = (
                initial_guess_clean.copy()
            )

            for t, group in gps.groupby(
                "t",
                sort=True,
            ):

                if len(group) < 4:
                    continue

                group = group.reset_index(
                    drop=True
                )

                # ---------------------------------------------------
                # Original measurements
                # ---------------------------------------------------

                clean_pseudoranges = (
                    group[
                        "pseudorange_m"
                    ]
                    .to_numpy(dtype=float)
                )

                # ---------------------------------------------------
                # Inject selective attack
                # ---------------------------------------------------

                attack_pseudoranges = (
                    clean_pseudoranges.copy()
                )

                attacked_mask = (
                    group["satellite_id"]
                    .isin(attacked_set)
                    .to_numpy()
                )

                attack_pseudoranges[
                    attacked_mask
                ] += attack_bias

                # ---------------------------------------------------
                # Clean solution
                # ---------------------------------------------------

                clean_solution = solve_epoch(
                    group,
                    clean_pseudoranges,
                    clean_initial_guess,
                )

                if (
                    clean_solution["position"]
                    is None
                ):
                    continue

                # ---------------------------------------------------
                # Attacked solution
                # ---------------------------------------------------

                attack_solution = solve_epoch(
                    group,
                    attack_pseudoranges,
                    attack_initial_guess,
                )

                if (
                    attack_solution["position"]
                    is None
                ):
                    continue

                # ---------------------------------------------------
                # Positions
                # ---------------------------------------------------

                clean_position = (
                    clean_solution["position"]
                )

                attack_position = (
                    attack_solution["position"]
                )

                true_position = (
                    get_true_position(group)
                )

                clean_position_error = (
                    np.linalg.norm(
                        clean_position
                        - true_position
                    )
                )

                attack_position_error = (
                    np.linalg.norm(
                        attack_position
                        - true_position
                    )
                )

                delta_position = (
                    np.linalg.norm(
                        attack_position
                        - clean_position
                    )
                )

                # ---------------------------------------------------
                # Clock bias
                # ---------------------------------------------------

                clean_clock = float(
                    clean_solution["clock_bias"]
                )

                attack_clock = float(
                    attack_solution["clock_bias"]
                )

                delta_clock = (
                    attack_clock
                    - clean_clock
                )

                # ---------------------------------------------------
                # Residuals
                # ---------------------------------------------------

                clean_residuals = np.asarray(
                    clean_solution["residuals"],
                    dtype=float,
                )

                attack_residuals = np.asarray(
                    attack_solution["residuals"],
                    dtype=float,
                )

                clean_abs_residual = (
                    np.abs(clean_residuals)
                )

                attack_abs_residual = (
                    np.abs(attack_residuals)
                )

                clean_mean_residual = (
                    np.mean(
                        clean_abs_residual
                    )
                )

                attack_mean_residual = (
                    np.mean(
                        attack_abs_residual
                    )
                )

                clean_max_residual = (
                    np.max(
                        clean_abs_residual
                    )
                )

                attack_max_residual = (
                    np.max(
                        attack_abs_residual
                    )
                )

                # ---------------------------------------------------
                # Residual changes
                # ---------------------------------------------------

                residual_change = (
                    attack_abs_residual
                    - clean_abs_residual
                )

                mean_delta_residual = (
                    np.mean(
                        np.abs(
                            residual_change
                        )
                    )
                )

                # ---------------------------------------------------
                # Abnormal satellite counts
                # ---------------------------------------------------

                n_abnormal_20 = int(
                    np.sum(
                        attack_abs_residual
                        > 20.0
                    )
                )

                n_abnormal_50 = int(
                    np.sum(
                        attack_abs_residual
                        > 50.0
                    )
                )

                # ---------------------------------------------------
                # Sign consistency
                # ---------------------------------------------------

                nonzero = (
                    attack_residuals[
                        np.abs(
                            attack_residuals
                        ) > 1e-9
                    ]
                )

                if len(nonzero) > 0:

                    positive_fraction = (
                        np.mean(
                            nonzero > 0
                        )
                    )

                    negative_fraction = (
                        np.mean(
                            nonzero < 0
                        )
                    )

                    sign_consistency = max(
                        positive_fraction,
                        negative_fraction,
                    )

                else:

                    sign_consistency = 0.0

                # ---------------------------------------------------
                # Save epoch result
                # ---------------------------------------------------

                detailed_rows.append(
                    {
                        "t": t,

                        "attack_configuration":
                            attack_name,

                        "attack_bias_m":
                            attack_bias,

                        "n_attacked_satellites":
                            len(
                                attacked_satellites
                            ),

                        "attack_fraction":
                            len(
                                attacked_satellites
                            )
                            / len(group),

                        "clean_position_error_m":
                            clean_position_error,

                        "attack_position_error_m":
                            attack_position_error,

                        "delta_position_m":
                            delta_position,

                        "clean_clock_bias_m":
                            clean_clock,

                        "attack_clock_bias_m":
                            attack_clock,

                        "delta_clock_bias_m":
                            delta_clock,

                        "clean_mean_abs_residual_m":
                            clean_mean_residual,

                        "attack_mean_abs_residual_m":
                            attack_mean_residual,

                        "delta_mean_abs_residual_m":
                            mean_delta_residual,

                        "clean_max_abs_residual_m":
                            clean_max_residual,

                        "attack_max_abs_residual_m":
                            attack_max_residual,

                        "n_abnormal_20m":
                            n_abnormal_20,

                        "n_abnormal_50m":
                            n_abnormal_50,

                        "sign_consistency":
                            sign_consistency,

                        "clean_converged":
                            clean_solution[
                                "converged"
                            ],

                        "attack_converged":
                            attack_solution[
                                "converged"
                            ],
                    }
                )

                # Carry solutions forward.
                clean_initial_guess = (
                    clean_position
                )

                attack_initial_guess = (
                    attack_position
                )

    # -----------------------------------------------------------------
    # Create dataframe
    # -----------------------------------------------------------------

    results = pd.DataFrame(
        detailed_rows
    )

    if results.empty:

        print(
            "\nNo successful attack experiments."
        )

        return

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    summary = (
        results
        .groupby(
            [
                "attack_configuration",
                "attack_bias_m",
            ]
        )
        .agg(
            epochs=(
                "t",
                "count",
            ),

            n_attacked_satellites=(
                "n_attacked_satellites",
                "first",
            ),

            mean_attack_fraction=(
                "attack_fraction",
                "mean",
            ),

            mean_clean_position_error_m=(
                "clean_position_error_m",
                "mean",
            ),

            mean_attack_position_error_m=(
                "attack_position_error_m",
                "mean",
            ),

            mean_delta_position_m=(
                "delta_position_m",
                "mean",
            ),

            p95_delta_position_m=(
                "delta_position_m",
                lambda x: x.quantile(0.95),
            ),

            max_delta_position_m=(
                "delta_position_m",
                "max",
            ),

            mean_delta_clock_m=(
                "delta_clock_bias_m",
                "mean",
            ),

            median_delta_clock_m=(
                "delta_clock_bias_m",
                "median",
            ),

            mean_attack_residual_m=(
                "attack_mean_abs_residual_m",
                "mean",
            ),

            p95_attack_residual_m=(
                "attack_mean_abs_residual_m",
                lambda x: x.quantile(0.95),
            ),

            max_attack_residual_m=(
                "attack_max_abs_residual_m",
                "max",
            ),

            mean_abnormal_20m=(
                "n_abnormal_20m",
                "mean",
            ),

            max_abnormal_20m=(
                "n_abnormal_20m",
                "max",
            ),

            mean_abnormal_50m=(
                "n_abnormal_50m",
                "mean",
            ),

            max_abnormal_50m=(
                "n_abnormal_50m",
                "max",
            ),

            mean_sign_consistency=(
                "sign_consistency",
                "mean",
            ),

            attack_convergence_rate=(
                "attack_converged",
                "mean",
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "REAL GNSS SELECTIVE ATTACK SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Save CSV files
    # -----------------------------------------------------------------

    csv_dir = (
        project_root
        / "results"
        / "csv"
    )

    csv_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = (
        csv_dir
        / "real_gnss_selective_attack.csv"
    )

    summary_path = (
        csv_dir
        / "real_gnss_selective_attack_summary.csv"
    )

    results.to_csv(
        detailed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nSaved detailed results:"
    )

    print(detailed_path)

    print(
        f"\nSaved summary:"
    )

    print(summary_path)

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------

    figure_dir = (
        project_root
        / "results"
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Figure 1: Position impact
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    for attack_name in summary[
        "attack_configuration"
    ].unique():

        subset = summary[
            summary[
                "attack_configuration"
            ]
            == attack_name
        ]

        plt.plot(
            subset["attack_bias_m"],
            subset["mean_delta_position_m"],
            marker="o",
            label=attack_name,
        )

    plt.xlabel(
        "Injected selective attack bias (m)"
    )

    plt.ylabel(
        "Mean change in position (m)"
    )

    plt.title(
        "Selective GNSS Attack vs Position Change"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    position_path = (
        figure_dir
        / "real_gnss_selective_attack_position.png"
    )

    plt.savefig(
        position_path,
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Figure 2: Clock absorption
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    for attack_name in summary[
        "attack_configuration"
    ].unique():

        subset = summary[
            summary[
                "attack_configuration"
            ]
            == attack_name
        ]

        plt.plot(
            subset["attack_bias_m"],
            subset["mean_delta_clock_m"],
            marker="o",
            label=attack_name,
        )

    plt.plot(
        ATTACK_BIASES_M,
        ATTACK_BIASES_M,
        "--",
        label="Ideal common-mode response",
    )

    plt.xlabel(
        "Injected selective attack bias (m)"
    )

    plt.ylabel(
        "Mean change in receiver clock bias (m)"
    )

    plt.title(
        "Selective GNSS Attack vs Receiver Clock Response"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    clock_path = (
        figure_dir
        / "real_gnss_selective_attack_clock.png"
    )

    plt.savefig(
        clock_path,
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Figure 3: Residual response
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    for attack_name in summary[
        "attack_configuration"
    ].unique():

        subset = summary[
            summary[
                "attack_configuration"
            ]
            == attack_name
        ]

        plt.plot(
            subset["attack_bias_m"],
            subset["mean_attack_residual_m"],
            marker="o",
            label=attack_name,
        )

    plt.axhline(
        20.0,
        linestyle="--",
        label="20 m threshold",
    )

    plt.xlabel(
        "Injected selective attack bias (m)"
    )

    plt.ylabel(
        "Mean |residual| (m)"
    )

    plt.title(
        "Selective GNSS Attack vs Residual Response"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    residual_path = (
        figure_dir
        / "real_gnss_selective_attack_residual.png"
    )

    plt.savefig(
        residual_path,
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Figure 4: Abnormal satellite count
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    for attack_name in summary[
        "attack_configuration"
    ].unique():

        subset = summary[
            summary[
                "attack_configuration"
            ]
            == attack_name
        ]

        plt.plot(
            subset["attack_bias_m"],
            subset["mean_abnormal_20m"],
            marker="o",
            label=attack_name,
        )

    plt.xlabel(
        "Injected selective attack bias (m)"
    )

    plt.ylabel(
        "Mean satellites with |residual| > 20 m"
    )

    plt.title(
        "Selective GNSS Attack vs Abnormal Satellite Count"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    abnormal_path = (
        figure_dir
        / "real_gnss_selective_attack_abnormal_satellites.png"
    )

    plt.savefig(
        abnormal_path,
        dpi=200,
    )

    plt.close()

    print(
        "\nSaved figures:"
    )

    print(position_path)
    print(clock_path)
    print(residual_path)
    print(abnormal_path)


if __name__ == "__main__":
    main()