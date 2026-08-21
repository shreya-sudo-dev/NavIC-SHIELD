"""
features/temporal.py

Turns raw per-timestep observations into rolling-window features a
classifier can use to flag spoofing. Operates per-satellite so rolling
windows never mix rows from different satellites.

Features are built on the RESIDUAL (pseudorange_m - true_range_m), not
raw pseudorange_m. Raw pseudorange for a GSO satellite changes by
thousands of meters per sample purely from real orbital motion, which
would swamp any injected attack offset. A real receiver knows the
satellite's ephemeris and expected range and would work from the
residual too -- this isn't a simplification, it's the physically
correct basis for these features.
"""

import numpy as np
import pandas as pd

# Rolling window size, in number of samples (not seconds) -- with
# SIM_TIMESTEP_S = 30s, a window of 8 covers ~4 minutes.
DEFAULT_WINDOW = 8


def compute_temporal_features(df: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """
    Add rolling-window temporal features to df, computed independently
    per satellite_id and only over visible rows (invisible rows have no
    observation to feature-ize).

    Added columns:
      pr_delta          : pseudorange_m - previous visible sample's pseudorange_m
      pr_roll_mean       : rolling mean of pseudorange_m over `window` visible samples
      pr_roll_std        : rolling std of pseudorange_m over `window` visible samples
      pr_zscore          : (pseudorange_m - pr_roll_mean) / pr_roll_std
                           (measures how anomalous this sample is relative
                           to its own recent history)
      cn0_delta          : cn0_db_hz - previous visible sample's cn0_db_hz
      cn0_roll_slope     : rolling linear-fit slope of cn0_db_hz over
                           `window` visible samples (trend indicator --
                           spoofing that manipulates signal power, or
                           masks its C/N0 signature, shows up here)

    Rows that are not visible get NaN in all feature columns -- they
    should be dropped before training, not imputed, since there's no
    real observation there.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain satellite_id, t, visible, pseudorange_m, cn0_db_hz
        (i.e. observation.py output, optionally with spoofing applied).
    window : int
        Rolling window size in samples.

    Returns
    -------
    pd.DataFrame
        Copy of df with the feature columns above added.
    """
    df = df.copy()
    feature_cols = ["pr_delta", "pr_roll_mean", "pr_roll_std", "pr_zscore",
                     "cn0_delta", "cn0_roll_slope"]
    for c in feature_cols:
        df[c] = np.nan

    output_frames = []
    for sat_id, group in df.groupby("satellite_id", sort=False):
        group = group.sort_values("t").copy()
        vis_mask = group["visible"].values
        vis_idx = np.where(vis_mask)[0]

        if len(vis_idx) == 0:
            output_frames.append(group)
            continue

        pr = group["pseudorange_m"].values[vis_idx]
        true_range = group["true_range_m"].values[vis_idx]
        cn0 = group["cn0_db_hz"].values[vis_idx]
        t_vis = group["t"].values[vis_idx]

        # IMPORTANT: features are built on the RESIDUAL (pseudorange -
        # true_range), not raw pseudorange. Raw pseudorange for a GSO
        # satellite changes by thousands of meters per sample from real
        # orbital motion alone (verified: ~2900 m/sample mean, spikes
        # far higher) -- an injected attack offset of a few meters to a
        # few hundred meters would be completely invisible against that.
        # A real receiver knows the satellite's ephemeris and expected
        # range, so working from the residual (which isolates clock
        # bias + noise + any attack offset) is both more realistic and
        # the only way this feature set is usable for GSO satellites.
        residual = pr - true_range

        pr_series = pd.Series(residual)
        cn0_series = pd.Series(cn0)

        pr_delta = pr_series.diff().values
        pr_roll_mean = pr_series.rolling(window, min_periods=2).mean().values
        pr_roll_std = pr_series.rolling(window, min_periods=2).std().values
        # Avoid divide-by-zero when a window happens to have ~zero variance
        pr_zscore = np.where(
            pr_roll_std > 1e-6,
            (residual - pr_roll_mean) / pr_roll_std,
            0.0,
        )

        cn0_delta = cn0_series.diff().values

        # Rolling slope of C/N0 via simple linear fit per window
        cn0_roll_slope = np.full(len(vis_idx), np.nan)
        for i in range(len(vis_idx)):
            lo = max(0, i - window + 1)
            if i - lo < 2:
                continue
            x = t_vis[lo:i + 1]
            y = cn0[lo:i + 1]
            slope = np.polyfit(x - x[0], y, 1)[0]
            cn0_roll_slope[i] = slope

        col_values = {
            "pr_delta": pr_delta,
            "pr_roll_mean": pr_roll_mean,
            "pr_roll_std": pr_roll_std,
            "pr_zscore": pr_zscore,
            "cn0_delta": cn0_delta,
            "cn0_roll_slope": cn0_roll_slope,
        }
        for col, values in col_values.items():
            full_col = np.full(len(group), np.nan)
            full_col[vis_idx] = values
            group[col] = full_col

        output_frames.append(group)

    result = pd.concat(output_frames, ignore_index=True)
    return result


FEATURE_COLUMNS = ["pr_delta", "pr_roll_mean", "pr_roll_std", "pr_zscore",
                    "cn0_delta", "cn0_roll_slope"]


# ---------------------------------------------------------------------------
# Sanity check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    import config
    from simulator.constellation import propagate_constellation
    from simulator.terrain import satellite_visibility
    from simulator.observation import generate_observations
    from spoofing.step_attack import apply_step_attack

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
    df = apply_step_attack(df, target_sat, duration / 2, step_size_m=200.0)

    feat_df = compute_temporal_features(df)

    sat_df = feat_df[(feat_df["satellite_id"] == target_sat) & (feat_df["visible"])]
    print(f"Target satellite: {target_sat}")
    print(f"Rows with features: {sat_df[FEATURE_COLUMNS].notna().all(axis=1).sum()} / {len(sat_df)}")

    spoofed_mean_z = sat_df[sat_df["is_spoofed"]]["pr_zscore"].abs().mean()
    normal_mean_z = sat_df[~sat_df["is_spoofed"]]["pr_zscore"].abs().mean()
    print(f"\nMean |pr_zscore|, spoofed rows: {spoofed_mean_z:.2f}")
    print(f"Mean |pr_zscore|, normal rows:  {normal_mean_z:.2f}")
    print("(expect spoofed >> normal for a step attack -- the whole point "
          "of pr_zscore is to catch exactly this)")