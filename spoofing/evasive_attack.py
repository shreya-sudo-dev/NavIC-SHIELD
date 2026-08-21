"""
spoofing/evasive_attack.py

The threat-model novelty of this project. Reaches the same kind of
eventual offset as a drift attack, but shaped specifically to stay
under the radar of a naive temporal detector: each individual step is
kept small relative to the natural pseudorange noise floor, and the
ramp is smooth (sigmoid-shaped) rather than constant-rate, avoiding a
sharp, easily-thresholded derivative.

Rule-based only -- no optimization/RL against the detector (out of
scope for the 5-day build).
"""

import numpy as np
import pandas as pd

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from spoofing.step_attack import _ensure_label_columns

# Pseudorange noise floor from observation.py's elevation-dependent model
# is roughly 1-3 m (see _elevation_dependent_range_noise_std). Cap the
# per-step delta at a small multiple of that so no single timestep looks
# like an outlier to a naive per-sample or rolling-derivative check.
_MAX_STEP_DELTA_M = 2.5  # meters per SIM_TIMESTEP_S interval


def _sigmoid_ramp(elapsed: np.ndarray, ramp_duration_s: float) -> np.ndarray:
    """
    Smooth 0->1 ramp over ramp_duration_s, centered at the midpoint of
    the ramp window. Using a sigmoid instead of a straight line means
    the attack starts and ends gently (near-zero rate of change at both
    ends) with the fastest rate of change in the middle -- there is no
    sharp on/off transition anywhere for a derivative-threshold detector
    to catch, unlike the step attack's instantaneous jump.
    """
    midpoint = ramp_duration_s / 2.0
    # Steepness chosen so the sigmoid goes from ~0.01 to ~0.99 across
    # the full ramp_duration_s window.
    steepness = 10.0 / ramp_duration_s
    return 1.0 / (1.0 + np.exp(-steepness * (elapsed - midpoint)))


def apply_evasive_attack(df: pd.DataFrame, satellite_id: str, start_time: float,
                          target_offset_m: float, ramp_duration_s: float,
                          seed: int = None) -> pd.DataFrame:
    """
    Smoothly ramp pseudorange_m up to target_offset_m over
    ramp_duration_s, capping the per-timestep delta so it stays within
    noise-like bounds for a naive temporal detector.

    Parameters
    ----------
    df : pd.DataFrame
        observation.py output (satellite_id, t, visible, pseudorange_m).
    satellite_id : str
        Which satellite to attack.
    start_time : float
        Simulation time (seconds) the attack begins.
    target_offset_m : float
        Final offset the attack ramps toward, meters.
    ramp_duration_s : float
        How long the ramp takes. Longer ramp = smaller per-step delta
        for the same target_offset_m, i.e. more evasive but slower to
        reach full effect. Choose this so target_offset_m /
        (ramp_duration_s / SIM_TIMESTEP_S) stays under _MAX_STEP_DELTA_M
        -- the function will warn (not fail) if it doesn't.
    seed : int, optional
        Unused (deterministic attack); kept for signature consistency.

    Returns
    -------
    pd.DataFrame
        Copy of df with pseudorange_m modified and is_spoofed/attack_type
        labels set for the attacked rows.
    """
    df = df.copy()
    df = _ensure_label_columns(df)

    mask = (
        (df["satellite_id"] == satellite_id)
        & (df["t"] >= start_time)
        & (df["visible"])
    )

    t_vals = df.loc[mask, "t"].values
    elapsed = t_vals - start_time
    elapsed_clipped = np.clip(elapsed, 0, ramp_duration_s)

    ramp_fraction = _sigmoid_ramp(elapsed_clipped, ramp_duration_s)
    # Beyond ramp_duration_s, hold at target_offset_m (sigmoid asymptotes
    # near 1.0 already, this just makes the intent explicit)
    ramp_fraction = np.where(elapsed > ramp_duration_s, 1.0, ramp_fraction)

    offset = target_offset_m * ramp_fraction

    # Sanity warning: is the per-step delta actually staying small?
    if len(offset) > 1:
        max_delta = np.max(np.abs(np.diff(offset)))
        if max_delta > _MAX_STEP_DELTA_M:
            print(f"[evasive_attack] WARNING: max per-step delta "
                  f"{max_delta:.2f} m exceeds the {_MAX_STEP_DELTA_M} m "
                  f"evasiveness target -- consider a longer ramp_duration_s "
                  f"or smaller target_offset_m for satellite {satellite_id}.")

    df.loc[mask, "pseudorange_m"] = df.loc[mask, "pseudorange_m"] + offset
    df.loc[mask, "is_spoofed"] = True
    df.loc[mask, "attack_type"] = "evasive"

    return df


# ---------------------------------------------------------------------------
# Sanity check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    import config
    from simulator.constellation import propagate_constellation
    from simulator.terrain import satellite_visibility
    from simulator.observation import generate_observations
    from spoofing.drift_attack import apply_drift_attack

    duration = 4 * 3600
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

    target_sat = df[df["visible"]].groupby("satellite_id").size().idxmax()
    start_time = duration / 4
    ramp_duration = 2 * 3600  # 2 hour ramp -> gentle per-step deltas

    evasive = apply_evasive_attack(df, target_sat, start_time,
                                    target_offset_m=150.0,
                                    ramp_duration_s=ramp_duration)
    drift_cmp = apply_drift_attack(df, target_sat, start_time,
                                    drift_rate_m_per_s=1.0)

    ev_df = evasive[(evasive["satellite_id"] == target_sat) & (evasive["visible"])]
    dr_df = drift_cmp[(drift_cmp["satellite_id"] == target_sat) & (drift_cmp["visible"])]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(ev_df["t"] / 60, ev_df["pseudorange_m"] - ev_df["true_range_m"])
    axes[0].axvline(start_time / 60, color="red", linestyle="--")
    axes[0].set_title(f"Evasive attack: sigmoid ramp to +150 m over {ramp_duration/60:.0f} min")
    axes[0].set_ylabel("pseudorange - true_range (m)")
    axes[0].grid(True)

    axes[1].plot(dr_df["t"] / 60, dr_df["pseudorange_m"] - dr_df["true_range_m"])
    axes[1].axvline(start_time / 60, color="red", linestyle="--")
    axes[1].set_title("Drift attack: constant 1.0 m/s (for comparison)")
    axes[1].set_xlabel("time (minutes)")
    axes[1].set_ylabel("pseudorange - true_range (m)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("evasive_vs_drift_check.png", dpi=120)

    ev_deltas = (ev_df.sort_values("t")["pseudorange_m"]
                 - ev_df.sort_values("t")["true_range_m"]).diff().dropna()
    print(f"Target satellite: {target_sat}")
    print(f"Evasive attack max per-step |delta| (pseudorange - true_range, "
          f"isolates the injected offset from real orbital range-rate): "
          f"{ev_deltas.abs().max():.2f} m")
    print(f"(target: stay within a few m of the {_MAX_STEP_DELTA_M} m cap, "
          f"plus pseudorange noise on top -- do NOT compare raw pseudorange_m "
          f"deltas directly, GSO satellites have real orbital range-rate that "
          f"can be tens to hundreds of m/s and will dominate that number)")
    print(f"Evasive attack final offset reached: {ev_df['pseudorange_m'].iloc[-1] - ev_df['true_range_m'].iloc[-1]:.1f} m")
    print("Saved evasive_vs_drift_check.png -- evasive curve should look "
          "visually much closer to noise than the drift ramp, despite "
          "reaching a comparable eventual offset")