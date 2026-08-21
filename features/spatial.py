"""
features/spatial.py

Cross-satellite consensus features: at each timestep, does a given
satellite's pseudorange residual agree with what the REST of the
currently-visible constellation implies?

Unlike temporal features (features/temporal.py), which operate within
one satellite's own history and fade back to normal once a rolling
window re-centers around a new baseline, spatial features compare
satellites against each other at a single instant. A spoofed satellite
stays inconsistent with the rest of the constellation for as long as
the attack continues -- this is what should give Day 3's fusion a real
advantage over the Day 2 temporal-only baseline on sustained attacks,
not just transitions.
"""

import numpy as np
import pandas as pd


def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add constellation_median_residual and spatial_deviation columns.

    For each timestep t, across all satellites visible at t:
      residual_i          = pseudorange_m_i - true_range_m_i
      constellation_median = median(residual_i) across all visible i at t
      spatial_deviation_i  = abs(residual_i - constellation_median)

    The median (not mean) is used deliberately: a single spoofed
    satellite pulling its own residual far from the pack should NOT
    drag the consensus value with it. With 2-7 visible satellites and
    at most one attacked at a time in this project's scope, the median
    stays anchored to the honest majority even when one satellite lies.

    Timesteps with fewer than 2 visible satellites get NaN for both
    columns -- there is no "rest of the constellation" to compare
    against with only one (or zero) satellites visible, and a
    deviation of 0 in that case would misleadingly look like perfect
    agreement rather than "not computable."

    Parameters
    ----------
    df : pd.DataFrame
        Must contain t, satellite_id, visible, pseudorange_m,
        true_range_m (i.e. observation.py output, optionally with
        spoofing applied).

    Returns
    -------
    pd.DataFrame
        Copy of df with constellation_median_residual and
        spatial_deviation columns added. Non-visible rows get NaN in
        both, same convention as temporal.py.
    """
    df = df.copy()
    df["residual_m"] = df["pseudorange_m"] - df["true_range_m"]
    df["constellation_median_residual"] = np.nan
    df["spatial_deviation"] = np.nan

    output_frames = []
    for t_val, group in df.groupby("t", sort=False):
        vis = group[group["visible"]]

        if len(vis) < 2:
            # Not enough visible satellites to form a consensus at this
            # timestep -- leave NaN, don't default to 0.
            output_frames.append(group)
            continue

        median_residual = vis["residual_m"].median()
        deviation = (vis["residual_m"] - median_residual).abs()

        group = group.copy()
        group.loc[vis.index, "constellation_median_residual"] = median_residual
        group.loc[vis.index, "spatial_deviation"] = deviation

        output_frames.append(group)

    result = pd.concat(output_frames, ignore_index=True)
    return result


SPATIAL_FEATURE_COLUMNS = ["spatial_deviation"]


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
    from spoofing.step_attack import apply_step_attack
    from spoofing.drift_attack import apply_drift_attack
    from spoofing.evasive_attack import apply_evasive_attack
    from features.temporal import compute_temporal_features

    print("Propagating full-day constellation...")
    duration = config.SIDEREAL_DAY_SECONDS
    dt = config.SIM_TIMESTEP_S
    df = propagate_constellation(0, duration, dt)

    print("Computing terrain visibility...")
    az, el, mask, vis, reason = [], [], [], [], []
    for _, row in df.iterrows():
        r = satellite_visibility(np.array([row.x, row.y, row.z]))
        az.append(r["azimuth_deg"]); el.append(r["elevation_deg"])
        mask.append(r["mask_angle_deg"]); vis.append(r["visible"])
        reason.append(r["visibility_reason"])
    df["azimuth_deg"], df["elevation_deg"] = az, el
    df["mask_angle_deg"], df["visible"], df["visibility_reason"] = mask, vis, reason

    print("Generating observations...")
    df = generate_observations(df)

    sat_ids = df[df["visible"]].groupby("satellite_id").size().sort_values(ascending=False).index.tolist()
    step_sat, drift_sat, evasive_sat = sat_ids[0], sat_ids[1], sat_ids[2]

    # IMPORTANT: attacks are staggered, not simultaneous. Three attacks
    # starting at the same time on three different satellites means
    # 50%+ of the visible constellation can be spoofed at once (verified:
    # 55% of post-attack timesteps hit this with a 7-satellite fleet) --
    # that breaks the median's core assumption (robust only while
    # spoofed satellites are a MINORITY), and contaminates the
    # consensus reference for genuinely clean satellites too. A single
    # attacker spoofing one satellite at a time is also the more
    # realistic threat model for this project's scope.
    step_start = duration * 0.4
    drift_start = step_start + 4 * 3600
    evasive_start = drift_start + 4 * 3600

    print(f"Attacks (staggered): step->{step_sat} @ {step_start/3600:.1f}h, "
          f"drift->{drift_sat} @ {drift_start/3600:.1f}h, "
          f"evasive->{evasive_sat} @ {evasive_start/3600:.1f}h")
    
    STEP_DURATION_S = 7200
    DRIFT_DURATION_S = 7200
    df = apply_step_attack(df, step_sat, step_start, step_size_m=200.0,
                            duration_s=STEP_DURATION_S)
    df = apply_drift_attack(df, drift_sat, drift_start, drift_rate_m_per_s=1.0,
                            duration_s=DRIFT_DURATION_S)

    df = apply_evasive_attack(df, evasive_sat, evasive_start,
                               target_offset_m=150.0, ramp_duration_s=2 * 3600)

    print("Computing spatial features...")
    df = compute_spatial_features(df)

    print("Computing temporal features (for side-by-side comparison)...")
    df = compute_temporal_features(df)

    # --- Check 1: how many visible satellites per timestep? (context
    #     for interpreting NaN rates below) ---
    vis_counts = df[df["visible"]].groupby("t").size()
    print(f"\nVisible satellites per timestep: min={vis_counts.min()}, "
          f"median={vis_counts.median()}, max={vis_counts.max()}")
    print(f"Timesteps with <2 visible (spatial_deviation will be NaN): "
          f"{(vis_counts < 2).sum()} / {len(vis_counts)}")

    # --- Check 2: spatial_deviation for the step-attacked satellite,
    #     sustained across the WHOLE attack, vs. temporal pr_zscore
    #     which we already know fades after the transition ---
    step_df = df[(df["satellite_id"] == step_sat) & (df["visible"])].sort_values("t")
    step_df = step_df[step_df["t"] > step_start - 600]  # window around attack

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(step_df["t"] / 60, step_df["spatial_deviation"])
    axes[0].axvline(step_start / 60, color="red", linestyle="--", label="attack start")
    axes[0].set_title(f"spatial_deviation for {step_sat} (step attack) -- "
                       f"should STAY elevated, not fade")
    axes[0].set_ylabel("spatial_deviation (m)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(step_df["t"] / 60, step_df["pr_zscore"])
    axes[1].axvline(step_start / 60, color="red", linestyle="--")
    axes[1].set_title(f"pr_zscore for {step_sat} (same attack) -- "
                       f"fades back to normal after the transition (Day 2 finding)")
    axes[1].set_xlabel("time (minutes)")
    axes[1].set_ylabel("pr_zscore")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("spatial_vs_temporal_check.png", dpi=120)

    pre_attack = step_df[step_df["t"] < step_start]["spatial_deviation"].mean()
    during_attack = step_df[step_df["t"] > step_start + 300]["spatial_deviation"].mean()
    print(f"\nMean spatial_deviation, {step_sat}, before attack: {pre_attack:.2f} m")
    print(f"Mean spatial_deviation, {step_sat}, well into attack: {during_attack:.2f} m")
    print("(expect during_attack >> pre_attack, and to STAY elevated, unlike pr_zscore)")

    # --- Check 3: a clean satellite's spatial_deviation, restricted to
    #     BEFORE any attack starts anywhere in the constellation. Using
    #     a whole-day average here would be misleading: after step_start
    #     (9.6h), some satellite is spoofed for most of the rest of the
    #     day, and even a genuinely clean satellite's deviation from the
    #     median rises somewhat when a minority of the constellation it's
    #     being compared against is compromised. That's an expected,
    #     bounded side effect of a mostly-honest-majority median, not
    #     a bug -- but the fair comparison is pre-any-attack vs.
    #     pre-any-attack, not pre-attack vs. whole-day-average.
    clean_sat = sat_ids[5] if len(sat_ids) > 5 else sat_ids[-1]
    clean_df = df[(df["satellite_id"] == clean_sat) & (df["visible"])
                  & (df["t"] < step_start)]
    print(f"\nMean spatial_deviation, clean satellite {clean_sat}, "
          f"pre-any-attack window: {clean_df['spatial_deviation'].mean():.2f} m "
          f"(expect small, noise-level, comparable to {step_sat}'s own "
          f"pre-attack value above)")

    print("\nSaved spatial_vs_temporal_check.png")