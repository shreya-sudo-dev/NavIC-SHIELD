"""
spoofing/drift_attack.py

Slow, linearly-growing offset rather than a sudden jump. Still simple
and still meant to be catchable by a temporal detector (via the
first-difference / delta features), but visually distinct from the
step attack -- a ramp rather than a cliff.
"""

import numpy as np
import pandas as pd

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from spoofing.step_attack import _ensure_label_columns


def apply_drift_attack(df: pd.DataFrame, satellite_id: str, start_time: float,
                        drift_rate_m_per_s: float, duration_s: float = None,
                        seed: int = None) -> pd.DataFrame:
    """
    Add a linearly growing offset to pseudorange_m for the target
    satellite, for all visible rows with t >= start_time (and
    t < start_time + duration_s if duration_s is given -- otherwise
    unbounded, growing for the rest of the simulation).

    offset(t) = drift_rate_m_per_s * (t - start_time)

    Parameters
    ----------
    df : pd.DataFrame
        observation.py output (satellite_id, t, visible, pseudorange_m).
    satellite_id : str
        Which satellite to attack.
    start_time : float
        Simulation time (seconds) the attack begins.
    drift_rate_m_per_s : float
        Rate the fake offset grows, meters/second. Keep this slow
        (~0.5-2 m/s) so it looks like a plausible drift rather than an
        obvious jump.
    duration_s : float, optional
        How long the attack lasts, in seconds. None (default) = grows
        unbounded for the rest of the simulation (Day 2 behavior).
        Given a value, the attack switches off after that many seconds
        -- pseudorange_m/is_spoofed/attack_type revert to normal, not
        held at the peak offset -- consistent with step_attack.py's
        bounded-duration behavior.
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

    end_time = start_time + duration_s if duration_s is not None else np.inf

    mask = (
        (df["satellite_id"] == satellite_id)
        & (df["t"] >= start_time)
        & (df["t"] < end_time)
        & (df["visible"])
    )

    elapsed = df.loc[mask, "t"] - start_time
    offset = drift_rate_m_per_s * elapsed

    df.loc[mask, "pseudorange_m"] = df.loc[mask, "pseudorange_m"] + offset
    df.loc[mask, "is_spoofed"] = True
    df.loc[mask, "attack_type"] = "drift"

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

    target_sat = df[df["visible"]].groupby("satellite_id").size().idxmax()
    start_time = duration / 3

    attacked = apply_drift_attack(df, target_sat, start_time, drift_rate_m_per_s=1.0)
    sat_df = attacked[(attacked["satellite_id"] == target_sat) & (attacked["visible"])]

    plt.figure(figsize=(10, 5))
    plt.plot(sat_df["t"] / 60, sat_df["pseudorange_m"] - sat_df["true_range_m"])
    plt.axvline(start_time / 60, color="red", linestyle="--", label="attack start")
    plt.xlabel("time (minutes)")
    plt.ylabel("pseudorange - true_range (m)")
    plt.title(f"Drift attack on {target_sat} (1.0 m/s from t={start_time/60:.0f} min)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("drift_attack_check.png", dpi=120)

    n_spoofed = sat_df["is_spoofed"].sum()
    print(f"Target satellite: {target_sat}")
    print(f"Spoofed rows: {n_spoofed} / {len(sat_df)}")
    print("Saved drift_attack_check.png -- expect a clean upward ramp after the red line")