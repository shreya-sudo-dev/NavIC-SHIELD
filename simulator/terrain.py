"""
simulator/terrain.py

Simplified terrain masking for the NavIC receiver.

The terrain is represented by an azimuth-dependent masking angle.
A ridge blocks low-elevation satellites in a particular azimuth
direction, while satellites looking away from the ridge use the
minimum elevation mask.

This is a simplified terrain model, not a DEM/SRTM model.
"""

import numpy as np

import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from simulator.receiver import compute_az_el


# ---------------------------------------------------------------------------
# Terrain configuration
# ---------------------------------------------------------------------------

# Initial ridge direction.
# This is a starting value and should later be justified from the
# simulated NavIC satellite azimuth distribution.
RIDGE_AZIMUTH_DEG = 250.0

# Maximum elevation required to clear the ridge.
PEAK_MASK_DEG = 25.0

# Minimum elevation required when looking away from the ridge.
VALLEY_MASK_DEG = config.MIN_ELEVATION_DEG

# Controls how wide the ridge's influence is in azimuth.
RIDGE_WIDTH_DEG = 45.0


# ---------------------------------------------------------------------------
# Angular utilities
# ---------------------------------------------------------------------------

def angular_difference_deg(
    azimuth_deg: float,
    reference_deg: float
) -> float:
    """
    Return the smallest absolute angular difference between two azimuths.

    Examples:
        355° and 5°  -> 10°
        250° and 250° -> 0°
    """

    return abs(
        (azimuth_deg - reference_deg + 180.0) % 360.0 - 180.0
    )


# ---------------------------------------------------------------------------
# Terrain masking profile
# ---------------------------------------------------------------------------

def mountain_mask_profile(
    azimuth_deg: float,
    ridge_azimuth_deg: float = RIDGE_AZIMUTH_DEG,
    peak_mask_deg: float = PEAK_MASK_DEG,
    valley_mask_deg: float = VALLEY_MASK_DEG,
    ridge_width_deg: float = RIDGE_WIDTH_DEG,
) -> float:
    """
    Calculate the minimum elevation required to see a satellite
    at a given azimuth.

    The mask is highest directly toward the ridge and smoothly
    decreases away from it.
    """

    delta = angular_difference_deg(
        azimuth_deg,
        ridge_azimuth_deg
    )

    # Smooth Gaussian-shaped ridge influence.
    weight = np.exp(
        -0.5 * (delta / ridge_width_deg) ** 2
    )

    mask_angle = (
        valley_mask_deg
        + (peak_mask_deg - valley_mask_deg) * weight
    )

    return float(mask_angle)


# ---------------------------------------------------------------------------
# Visibility decision
# ---------------------------------------------------------------------------

def is_visible(
    elevation_deg: float,
    azimuth_deg: float,
    ridge_azimuth_deg: float = RIDGE_AZIMUTH_DEG,
    peak_mask_deg: float = PEAK_MASK_DEG,
    valley_mask_deg: float = VALLEY_MASK_DEG,
    ridge_width_deg: float = RIDGE_WIDTH_DEG,
):
    """
    Determine whether a satellite is visible.

    Returns
    -------
    visible : bool
    reason : str

    Possible reasons:
        "below_min_elevation"
        "terrain_blocked"
        "visible"
    """

    # First apply the basic horizon/elevation mask.
    if elevation_deg < config.MIN_ELEVATION_DEG:
        return False, "below_min_elevation"

    # Then apply terrain masking.
    mask_angle = mountain_mask_profile(
        azimuth_deg,
        ridge_azimuth_deg,
        peak_mask_deg,
        valley_mask_deg,
        ridge_width_deg,
    )

    if elevation_deg < mask_angle:
        return False, "terrain_blocked"

    return True, "visible"


# ---------------------------------------------------------------------------
# Satellite-level visibility
# ---------------------------------------------------------------------------

def satellite_visibility(
    sat_ecef,
    receiver_ecef=None,
    receiver_lat_deg=None,
    receiver_lon_deg=None,
):
    """
    Convert satellite ECEF position to azimuth/elevation and
    determine visibility after terrain masking.
    """

    azimuth_deg, elevation_deg = compute_az_el(
        sat_ecef=sat_ecef,
        receiver_ecef=receiver_ecef,
        receiver_lat_deg=receiver_lat_deg,
        receiver_lon_deg=receiver_lon_deg,
    )

    mask_angle_deg = mountain_mask_profile(azimuth_deg)

    visible, reason = is_visible(
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
    )

    return {
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "mask_angle_deg": mask_angle_deg,
        "visible": visible,
        "visibility_reason": reason,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # -------------------------------------------------------
    # Check 1: mask angle versus azimuth
    # -------------------------------------------------------

    azimuths = np.linspace(0.0, 360.0, 721)

    mask_angles = np.array([
        mountain_mask_profile(az)
        for az in azimuths
    ])

    plt.figure(figsize=(10, 5))

    plt.plot(
        azimuths,
        mask_angles,
        label="Terrain mask"
    )

    plt.axvline(
        RIDGE_AZIMUTH_DEG,
        linestyle="--",
        label="Ridge direction"
    )

    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Minimum visible elevation (degrees)")
    plt.title("Simplified NavIC Terrain Mask")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "terrain_mask.png",
        dpi=150
    )

    print("Saved terrain_mask.png")

    # -------------------------------------------------------
    # Check 2: mask values at selected azimuths
    # -------------------------------------------------------

    print("\nTerrain mask values:")

    for az in [0, 90, 180, 225, 250, 270, 315]:

        mask = mountain_mask_profile(az)

        print(
            f"Azimuth {az:6.1f}° -> "
            f"minimum elevation {mask:5.2f}°"
        )

    # -------------------------------------------------------
    # Check 3: visibility examples
    # -------------------------------------------------------

    print("\nVisibility examples:")

    examples = [
        (3.0, 250.0),
        (10.0, 250.0),
        (30.0, 250.0),
        (10.0, 90.0),
        (30.0, 90.0),
    ]

    for elevation, azimuth in examples:

        visible, reason = is_visible(
            elevation_deg=elevation,
            azimuth_deg=azimuth,
        )

        print(
            f"Az={azimuth:6.1f}°, "
            f"El={elevation:5.1f}° -> "
            f"{visible}, {reason}"
        )