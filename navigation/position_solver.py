"""
navigation/position_solver.py

Standard GNSS position solve: given pseudoranges from >=4 visible
satellites at one timestep, solve for receiver ECEF position (x, y, z)
and clock bias (b) via iterative weighted least squares.

This is what a real receiver does continuously. Nothing upstream in
this project (temporal/spatial features) computes an actual position --
they work on per-satellite residuals for detection. This module is what
turns "satellite X looks spoofed" into "the receiver's computed
position is now off by N meters", which is the actual quantity
navigation/kalman_fallback.py needs to correct.

Model: pseudorange_i = |x_receiver - x_sat_i| + clock_bias + noise
Linearized around a current estimate, iterated (Gauss-Newton) a few
times until the position update is small.
"""

import numpy as np
import pandas as pd

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def solve_position_single_epoch(sat_positions: np.ndarray, pseudoranges: np.ndarray,
                                 initial_guess: np.ndarray, max_iter: int = 10,
                                 tol: float = 1e-3) -> dict:
    """
    Solve receiver ECEF position + clock bias from one epoch's
    pseudoranges via iterative weighted least squares (Gauss-Newton).

    Parameters
    ----------
    sat_positions : np.ndarray, shape (n_sats, 3)
        ECEF positions of the visible satellites at this timestep.
    pseudoranges : np.ndarray, shape (n_sats,)
        Measured pseudorange to each satellite (meters). May include
        spoofed values -- the solver has no way to know that, which is
        the point: this reproduces what a real receiver would compute.
    initial_guess : np.ndarray, shape (3,)
        Starting ECEF position estimate. In practice, the previous
        epoch's solved position (or the known receiver site for the
        very first epoch) works well since position changes little
        between consecutive 30s samples for a stationary/slow receiver.
    max_iter : int
        Maximum Gauss-Newton iterations.
    tol : float
        Convergence threshold on position update magnitude (meters).

    Returns
    -------
    dict with:
        position : np.ndarray, shape (3,) -- solved ECEF position
        clock_bias : float -- solved clock bias (meters, i.e. c * dt)
        residuals : np.ndarray, shape (n_sats,) -- final pseudorange
            residuals (measured - predicted), useful as a solve-quality
            diagnostic
        converged : bool
        n_iterations : int
        n_satellites : int
    """
    n_sats = len(pseudoranges)
    if n_sats < 4:
        return {
            "position": None, "clock_bias": None, "residuals": None,
            "converged": False, "n_iterations": 0, "n_satellites": n_sats,
        }

    x = np.array(initial_guess, dtype=float)
    b = 0.0  # clock bias, meters

    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        ranges = np.linalg.norm(sat_positions - x, axis=1)
        predicted_pr = ranges + b
        residuals = pseudoranges - predicted_pr

        # Jacobian: d(pseudorange)/d(x,y,z,b) = [-unit_vector_to_sat, 1]
        unit_vectors = (x - sat_positions) / ranges[:, None]
        H = np.hstack([unit_vectors, np.ones((n_sats, 1))])

        # Least-squares solve for [dx, dy, dz, db]
        delta, _, _, _ = np.linalg.lstsq(H, residuals, rcond=None)

        x = x + delta[:3]
        b = b + delta[3]

        if np.linalg.norm(delta[:3]) < tol:
            converged = True
            break

    final_ranges = np.linalg.norm(sat_positions - x, axis=1)
    final_residuals = pseudoranges - (final_ranges + b)

    return {
        "position": x, "clock_bias": b, "residuals": final_residuals,
        "converged": converged, "n_iterations": n_iter, "n_satellites": n_sats,
    }


def solve_position_timeseries(df: pd.DataFrame, receiver_ecef_guess: np.ndarray) -> pd.DataFrame:
    """
    Run solve_position_single_epoch for every timestep in df, using
    each epoch's visible satellites. Carries the previous epoch's
    solution forward as the next epoch's initial guess for faster
    convergence (falls back to receiver_ecef_guess if no prior solve
    succeeded yet).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain t, satellite_id, visible, x, y, z (satellite ECEF
        position), pseudorange_m.
    receiver_ecef_guess : np.ndarray, shape (3,)
        Initial position guess for the very first epoch (use the known
        receiver site -- realistic, since a receiver has at least a
        rough prior position, e.g. from its last known fix).

    Returns
    -------
    pd.DataFrame with one row per timestep (not per satellite):
        t, solved_x, solved_y, solved_z, clock_bias_m, n_satellites_used,
        converged, mean_abs_residual_m
    """
    results = []
    current_guess = np.array(receiver_ecef_guess, dtype=float)

    for t_val, group in df.groupby("t", sort=True):
        vis = group[group["visible"]]
        if len(vis) < 4:
            results.append({
                "t": t_val, "solved_x": np.nan, "solved_y": np.nan, "solved_z": np.nan,
                "clock_bias_m": np.nan, "n_satellites_used": len(vis),
                "converged": False, "mean_abs_residual_m": np.nan,
            })
            continue

        sat_positions = vis[["x", "y", "z"]].values
        pseudoranges = vis["pseudorange_m"].values

        solution = solve_position_single_epoch(sat_positions, pseudoranges, current_guess)

        if solution["position"] is not None:
            current_guess = solution["position"]  # carry forward for next epoch
            results.append({
                "t": t_val,
                "solved_x": solution["position"][0],
                "solved_y": solution["position"][1],
                "solved_z": solution["position"][2],
                "clock_bias_m": solution["clock_bias"],
                "n_satellites_used": solution["n_satellites"],
                "converged": solution["converged"],
                "mean_abs_residual_m": np.mean(np.abs(solution["residuals"])),
            })
        else:
            results.append({
                "t": t_val, "solved_x": np.nan, "solved_y": np.nan, "solved_z": np.nan,
                "clock_bias_m": np.nan, "n_satellites_used": len(vis),
                "converged": False, "mean_abs_residual_m": np.nan,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Sanity check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import config
    from simulator.constellation import propagate_constellation
    from simulator.terrain import satellite_visibility
    from simulator.observation import generate_observations
    from simulator.receiver import _DEFAULT_RECEIVER_ECEF

    duration = 3 * 3600
    dt = config.SIM_TIMESTEP_S
    df = propagate_constellation(0, duration, dt)

    az, el, mask, vis, reason = [], [], [], [], []
    for _, row in df.iterrows():
        r = satellite_visibility(np.array([row.x, row.y, row.z]))
        az.append(r["azimuth_deg"]); el.append(r["elevation_deg"])
        mask.append(r["mask_angle_deg"]); vis.append(r["visible"])
        reason.append(r["visibility_reason"])
    df["azimuth_deg"], df["elevation_deg"] = az, el
    df["mask_angle_deg"], df["visible"], df["visibility_reason"] = mask, vis, reason

    df = generate_observations(df)

    print("Solving position timeseries on CLEAN (unattacked) data...")
    solved = solve_position_timeseries(df, _DEFAULT_RECEIVER_ECEF)

    valid = solved.dropna(subset=["solved_x"])
    print(f"\nSolved {len(valid)} / {len(solved)} epochs "
          f"(rest had <4 visible satellites)")

    # Compare solved position against the TRUE receiver position -- on
    # clean data this error should be small (just noise from pseudorange
    # measurement error, nothing structural).
    true_pos = _DEFAULT_RECEIVER_ECEF
    errors = np.linalg.norm(
        valid[["solved_x", "solved_y", "solved_z"]].values - true_pos, axis=1
    )
    print(f"\nPosition error vs true receiver location (clean data, no attacks):")
    print(f"  mean: {errors.mean():.2f} m")
    print(f"  max:  {errors.max():.2f} m")
    print(f"  (expect a few meters -- roughly matching the pseudorange noise "
          f"floor from observation.py, NOT hundreds of meters. If this is "
          f"large, something's wrong with the solver, not the data.)")

    print(f"\nMean |residual| across solved epochs: "
          f"{valid['mean_abs_residual_m'].mean():.2f} m")
    print(f"Satellites used per epoch: min={valid['n_satellites_used'].min()}, "
          f"max={valid['n_satellites_used'].max()}")