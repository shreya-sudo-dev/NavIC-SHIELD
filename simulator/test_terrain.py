import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import config

from simulator.constellation import propagate_constellation
from simulator.receiver import compute_az_el
from simulator.terrain import mountain_mask_profile, is_visible


# Simulate one hour first
duration = config.SIDEREAL_DAY_SECONDS
dt = config.SIM_TIMESTEP_S

df = propagate_constellation(
    0,
    duration,
    dt
)
print("\nColumns returned by constellation:")
print(df.columns.tolist())

# Calculate terrain/visibility information
azimuths = []
elevations = []
mask_angles = []
visibility = []
reasons = []

for _, row in df.iterrows():

    sat_ecef = np.array([
        row["x"],
        row["y"],
        row["z"]
    ])

    az, el = compute_az_el(sat_ecef)

    mask = mountain_mask_profile(az)

    visible, reason = is_visible(
        elevation_deg=el,
        azimuth_deg=az
    )

    azimuths.append(az)
    elevations.append(el)
    mask_angles.append(mask)
    visibility.append(visible)
    reasons.append(reason)


df["azimuth_deg"] = azimuths
df["elevation_deg"] = elevations
df["mask_angle_deg"] = mask_angles
df["visible"] = visibility
df["visibility_reason"] = reasons


print("\n===== TERRAIN INTEGRATION TEST =====")

print("\nTotal observations:", len(df))

print("\nVisibility counts:")
print(df["visibility_reason"].value_counts())

print("\nVisible observations:", df["visible"].sum())
print("Blocked observations:", (~df["visible"]).sum())

print("\nSample:")
print(
    df[
        [
            "satellite_id",
            "t",
            "azimuth_deg",
            "elevation_deg",
            "mask_angle_deg",
            "visible",
            "visibility_reason",
        ]
    ].head(20).to_string(index=False)
)


print("\n===== TERRAIN GEOMETRY DIAGNOSTIC =====")

visible_df = df[df["elevation_deg"] >= config.MIN_ELEVATION_DEG].copy()

print("\nVisible elevation range:")
print(
    f"min = {visible_df['elevation_deg'].min():.2f}°"
    f", max = {visible_df['elevation_deg'].max():.2f}°"
)

print("\nVisible azimuth range:")
print(
    f"min = {visible_df['azimuth_deg'].min():.2f}°"
    f", max = {visible_df['azimuth_deg'].max():.2f}°"
)

# Find observations closest to the ridge direction
visible_df["ridge_distance_deg"] = (
    abs(
        (
            visible_df["azimuth_deg"]
            - 250.0
            + 180.0
        ) % 360.0 - 180.0
    )
)

closest = visible_df.sort_values(
    "ridge_distance_deg"
).head(20)

print("\nObservations closest to ridge direction (250°):")

print(
    closest[
        [
            "satellite_id",
            "t",
            "azimuth_deg",
            "elevation_deg",
            "mask_angle_deg",
            "ridge_distance_deg",
        ]
    ].to_string(index=False)
)

# Calculate how much margin each visible observation has
visible_df["terrain_margin_deg"] = (
    visible_df["elevation_deg"]
    - visible_df["mask_angle_deg"]
)

print("\nSmallest terrain margins:")

print(
    visible_df.sort_values("terrain_margin_deg")[
        [
            "satellite_id",
            "t",
            "azimuth_deg",
            "elevation_deg",
            "mask_angle_deg",
            "terrain_margin_deg",
        ]
    ].head(20).to_string(index=False)
)

print("Visible observations:",
      (df["visibility_reason"] == "visible").sum())

print("Terrain-blocked observations:",
      (df["visibility_reason"] == "terrain_blocked").sum())

print("Below-min-elevation observations:",
      (df["visibility_reason"] == "below_min_elevation").sum())