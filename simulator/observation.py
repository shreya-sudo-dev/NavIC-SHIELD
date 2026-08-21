"""
simulator/observation.py

Generates per-satellite observations (pseudorange, C/N0, Doppler) for every
row where the satellite is visible (per terrain.py). Grounded in the ICD's
received-power range from config.py rather than arbitrary noise levels.

Expects to operate on a dataframe that already has the columns produced by
constellation.py (satellite_id, t, x, y, z) plus terrain.py
(azimuth_deg, elevation_deg, visible, visibility_reason).
"""

import numpy as np
import pandas as pd

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from simulator.receiver import _DEFAULT_RECEIVER_ECEF


def elevation_to_cn0(elevation_deg: np.ndarray) -> np.ndarray:
    """
    Map elevation angle to C/N0 (dB-Hz), interpolating received power
    between the ICD's minimum (at 5 deg) and max (near zenith) values,
    then converting power -> C/N0 via the receiver noise floor.
    """
    elevation_clipped = np.clip(elevation_deg, config.MIN_ELEVATION_DEG, 90.0)
    frac = (elevation_clipped - config.MIN_ELEVATION_DEG) / (90.0 - config.MIN_ELEVATION_DEG)

    rx_power_dbw = (
        config.NAVIC_NOMINAL_RX_POWER_DBW
        + frac * (config.NAVIC_MAX_RX_POWER_DBW - config.NAVIC_NOMINAL_RX_POWER_DBW)
    )

    cn0_db_hz = rx_power_dbw - config.NOISE_FLOOR_DBW_HZ
    return cn0_db_hz


def _elevation_dependent_range_noise_std(elevation_deg: np.ndarray) -> np.ndarray:
    """
    Pseudorange noise std (meters), worse at low elevation (more multipath/
    atmospheric residual error), better near zenith. Simplification of the
    true atmospheric delay curve -- adequate for a detection-algorithm
    testbed, not a precision-positioning study.
    """
    elevation_clipped = np.clip(elevation_deg, config.MIN_ELEVATION_DEG, 90.0)
    frac = (elevation_clipped - config.MIN_ELEVATION_DEG) / (90.0 - config.MIN_ELEVATION_DEG)
    return 3.0 - 2.0 * frac  # ~3m at 5 deg, ~1m at 90 deg


def generate_observations(df: pd.DataFrame, receiver_ecef: np.ndarray = None,
                           clock_bias_m: float = 2.5,
                           seed: int = 42) -> pd.DataFrame:
    """
    Add pseudorange_m, cn0_db_hz, doppler_hz, true_range_m columns to df.
    Only rows where visible == True get real observation values; all other
    rows get NaN in these columns (consistent with them not being tracked).

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain satellite_id, t, x, y, z, azimuth_deg,
        elevation_deg, visible (i.e. constellation.py + terrain.py output
        merged together).
    receiver_ecef : np.ndarray, optional
        Defaults to config.RECEIVER_SITE via receiver.py.
    clock_bias_m : float
        Fixed receiver clock bias for this simulation run.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame, same as input plus true_range_m, pseudorange_m,
    cn0_db_hz, doppler_hz columns.
    """
    if receiver_ecef is None:
        receiver_ecef = _DEFAULT_RECEIVER_ECEF

    rng = np.random.default_rng(seed)

    df = df.copy()
    df["true_range_m"] = np.nan
    df["pseudorange_m"] = np.nan
    df["cn0_db_hz"] = np.nan
    df["doppler_hz"] = np.nan

    # True range can be computed for every row (visible or not) -- it's
    # pure geometry, needed for Doppler continuity across gaps too.
    diffs = df[["x", "y", "z"]].values - receiver_ecef
    df["true_range_m"] = np.linalg.norm(diffs, axis=1)

    # Process satellite by satellite so Doppler is computed against the
    # correct previous *visible* sample, not just the previous row in the
    # file (which may belong to a different satellite or a gap).
    output_frames = []
    for sat_id, group in df.groupby("satellite_id", sort=False):
        group = group.sort_values("t").copy()
        vis_mask = group["visible"].values

        n = len(group)
        pseudorange = np.full(n, np.nan)
        cn0 = np.full(n, np.nan)
        doppler = np.full(n, np.nan)

        vis_idx = np.where(vis_mask)[0]
        if len(vis_idx) > 0:
            el_vis = group["elevation_deg"].values[vis_idx]
            range_vis = group["true_range_m"].values[vis_idx]
            t_vis = group["t"].values[vis_idx]

            # C/N0
            cn0_mean = elevation_to_cn0(el_vis)
            cn0[vis_idx] = cn0_mean + rng.normal(0, 0.5, size=len(vis_idx))

            # Pseudorange
            range_noise_std = _elevation_dependent_range_noise_std(el_vis)
            pseudorange[vis_idx] = (
                range_vis + clock_bias_m
                + rng.normal(0, range_noise_std, size=len(vis_idx))
            )

            # Doppler: finite difference against the PREVIOUS VISIBLE
            # sample only (not the previous row overall), using the
            # actual elapsed time between those two visible samples.
            # Uses the L5 carrier -- NavIC's civilian SPS signal (NOT L1,
            # which is a GPS band NavIC does not broadcast on).
            for i in range(1, len(vis_idx)):
                dt = t_vis[i] - t_vis[i - 1]
                if dt <= 0:
                    continue
                range_rate = (range_vis[i] - range_vis[i - 1]) / dt
                doppler_hz = -(range_rate / config.SPEED_OF_LIGHT) * config.NAVIC_L5_FREQ
                doppler_hz += rng.normal(0, 2.0)
                doppler[vis_idx[i]] = doppler_hz

        group["pseudorange_m"] = pseudorange
        group["cn0_db_hz"] = cn0
        group["doppler_hz"] = doppler
        output_frames.append(group)

    result = pd.concat(output_frames, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Sanity checks (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from simulator.constellation import propagate_constellation
    from simulator.terrain import satellite_visibility

    period = config.SIDEREAL_DAY_SECONDS
    dt = config.SIM_TIMESTEP_S

    print("Propagating constellation...")
    df = propagate_constellation(0, period, dt)

    print("Computing terrain visibility for each row...")
    az_list, el_list, mask_list, vis_list, reason_list = [], [], [], [], []
    for _, row in df.iterrows():
        sat_ecef = np.array([row.x, row.y, row.z])
        result = satellite_visibility(sat_ecef)
        az_list.append(result["azimuth_deg"])
        el_list.append(result["elevation_deg"])
        mask_list.append(result["mask_angle_deg"])
        vis_list.append(result["visible"])
        reason_list.append(result["visibility_reason"])

    df["azimuth_deg"] = az_list
    df["elevation_deg"] = el_list
    df["mask_angle_deg"] = mask_list
    df["visible"] = vis_list
    df["visibility_reason"] = reason_list

    print("Generating observations...")
    df = generate_observations(df)

    visible_df = df[df["visible"]]
    print(f"\nTotal rows: {len(df)}, visible rows: {len(visible_df)}")

    print(f"\nC/N0 range (visible obs): "
          f"{visible_df['cn0_db_hz'].min():.1f} to {visible_df['cn0_db_hz'].max():.1f} dB-Hz")
    print("(expect roughly 40-55 dB-Hz -- if wildly off, check power/unit conversions)")

    print(f"\nPseudorange noise implied (visible obs, std of pseudorange - true_range - clock_bias): "
          f"{(visible_df['pseudorange_m'] - visible_df['true_range_m'] - 2.5).std():.2f} m")

    doppler_by_type = visible_df.dropna(subset=["doppler_hz"]).groupby(
        df.loc[visible_df.dropna(subset=["doppler_hz"]).index, "type"]
    )["doppler_hz"].agg(["mean", "std", "min", "max"])
    print("\nDoppler stats by satellite type (GEO should be near-zero/stable, "
          "GSO should show real variation):")
    print(doppler_by_type)

    # Sanity plot: C/N0 vs elevation across all visible obs
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(visible_df["elevation_deg"], visible_df["cn0_db_hz"], s=2, alpha=0.3)
    ax.set_xlabel("Elevation (deg)")
    ax.set_ylabel("C/N0 (dB-Hz)")
    ax.set_title("C/N0 vs elevation (should show clear positive trend)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("cn0_vs_elevation.png", dpi=120)
    print("\nSaved cn0_vs_elevation.png")