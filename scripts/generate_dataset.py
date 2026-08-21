"""
generate_dataset.py

Day 1 deliverable: orchestrates constellation.py -> terrain.py ->
observation.py into one final CSV covering a full sidereal day.

Output columns: satellite_id, type, t, x, y, z, azimuth_deg, elevation_deg,
mask_angle_deg, visible, visibility_reason, true_range_m, pseudorange_m,
cn0_db_hz, doppler_hz, label

label = "clean" for everything at this stage; Day 2's spoofing attacks will
overwrite this for affected rows.
"""

import numpy as np
import pandas as pd
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from simulator.constellation import propagate_constellation
from simulator.terrain import satellite_visibility
from simulator.observation import generate_observations


def generate_full_dataset(duration_s: float = None, dt: float = None,
                           seed: int = 42) -> pd.DataFrame:
    """
    Run the full Day 1 pipeline and return the final dataframe.
    """
    if duration_s is None:
        duration_s = config.SIDEREAL_DAY_SECONDS
    if dt is None:
        dt = config.SIM_TIMESTEP_S

    print(f"Propagating constellation over {duration_s/3600:.2f} hours "
          f"at {dt}s steps...")
    df = propagate_constellation(0, duration_s, dt)

    print("Computing terrain visibility...")
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

    print("Generating observations (pseudorange, C/N0, Doppler)...")
    df = generate_observations(df, seed=seed)

    df["label"] = "clean"

    return df


def summarize(df: pd.DataFrame):
    print(f"\nTotal rows: {len(df)}")
    print(f"Unique satellites: {df['satellite_id'].nunique()}")

    print("\nVisibility reason breakdown:")
    print(df["visibility_reason"].value_counts())

    visible_df = df[df["visible"]]

    print("\nVisibility rate per satellite:")
    print(df.groupby("satellite_id")["visible"].mean().sort_values())

    print(f"\nC/N0 range (visible obs): "
          f"{visible_df['cn0_db_hz'].min():.1f} to {visible_df['cn0_db_hz'].max():.1f} dB-Hz")

    n_missing_obs = visible_df["pseudorange_m"].isna().sum()
    print(f"\nVisible rows missing pseudorange (should be 0): {n_missing_obs}")

    n_invisible_with_obs = df[~df["visible"]]["pseudorange_m"].notna().sum()
    print(f"Invisible rows WITH pseudorange (should be 0): {n_invisible_with_obs}")


if __name__ == "__main__":
    df = generate_full_dataset()
    summarize(df)

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    out_path = os.path.join(
        RESULTS_DIR,
        "dataset_day1_clean.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nSaved dataset to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")