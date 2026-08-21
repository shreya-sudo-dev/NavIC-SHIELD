"""
spoofing/step_attack.py

Simplest spoofing pattern: a sudden constant offset added to a target
satellite's pseudorange, starting at a fixed time. This should be the
easiest attack for a temporal detector to catch -- it's the sanity-check
attack, not the interesting one.
"""

import numpy as np
import pandas as pd


def _ensure_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure is_spoofed / attack_type columns exist before attacks
    are layered on, so multiple attacks can be applied to the same df
    without clobbering each other's labels."""
    if "is_spoofed" not in df.columns:
        df["is_spoofed"] = False
    if "attack_type" not in df.columns:
        df["attack_type"] = "none"
    return df


def apply_step_attack(df: pd.DataFrame, satellite_id: str, start_time: float,
                       step_size_m: float, duration_s: float = None,
                       seed: int = None) -> pd.DataFrame:
    """
    Add a constant offset to pseudorange_m for the target satellite, for
    all visible rows with t >= start_time (and t < start_time +
    duration_s, if duration_s is given -- otherwise the attack runs for
    the rest of the simulation, unbounded).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain satellite_id, t, visible, pseudorange_m (i.e.
        observation.py output).
    satellite_id : str
        Which satellite to attack.
    start_time : float
        Simulation time (seconds) the attack begins.
    step_size_m : float
        Constant offset added to pseudorange, meters. Positive = satellite
        appears farther away than it really is.
    duration_s : float, optional
        How long the attack lasts, in seconds. None (default) = runs
        for the rest of the simulation, unbounded (Day 2 behavior).
        Given a value, the attack automatically ends after that many
        seconds and pseudorange_m/is_spoofed/attack_type revert to
        normal for rows after that.
    seed : int, optional
        Unused here (attack is deterministic) but accepted for a
        consistent function signature across step/drift/evasive.

    Returns
    -------
    pd.DataFrame
        Copy of df with pseudorange_m modified and is_spoofed/attack_type
        labels set for the attacked rows.
    """
    df = df.copy()
    df = _ensure_label_columns(df)

    end_time = start_time + duration_s if duration_s is not None else np.inf

    mask = (
        (df["satellite_id"] == satellite_id)
        & (df["t"] >= start_time)
        & (df["t"] < end_time)
        & (df["visible"])
    )

    df.loc[mask, "pseudorange_m"] = df.loc[mask, "pseudorange_m"] + step_size_m
    df.loc[mask, "is_spoofed"] = True
    df.loc[mask, "attack_type"] = "step"

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

    # Build a short dataset for a quick visual check
    duration = 3 * 3600  # 3 hours is enough to see a step clearly
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

    # Pick whichever satellite has the most visible samples for a clean plot
    target_sat = (
        df[df["visible"]].groupby("satellite_id").size().idxmax()
    )
    start_time = duration / 2

    attacked = apply_step_attack(df, target_sat, start_time, step_size_m=200.0)

    sat_df = attacked[(attacked["satellite_id"] == target_sat) & (attacked["visible"])]

    plt.figure(figsize=(10, 5))
    plt.plot(sat_df["t"] / 60, sat_df["pseudorange_m"] - sat_df["true_range_m"])
    plt.axvline(start_time / 60, color="red", linestyle="--", label="attack start")
    plt.xlabel("time (minutes)")
    plt.ylabel("pseudorange - true_range (m)")
    plt.title(f"Step attack on {target_sat} (+200 m at t={start_time/60:.0f} min)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step_attack_check.png", dpi=120)

    n_spoofed = sat_df["is_spoofed"].sum()
    print(f"Target satellite: {target_sat}")
    print(f"Spoofed rows: {n_spoofed} / {len(sat_df)}")
    print("Saved step_attack_check.png -- expect a clean vertical jump at the red line")