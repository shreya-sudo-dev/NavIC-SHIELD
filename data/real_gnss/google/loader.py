"""
Google Smartphone Decimeter Challenge -> NavIC-SHIELD loader.

Converts real GNSS measurements into a common format that can later
be consumed by the existing NavIC-SHIELD feature, detection, and
navigation modules.

This loader does NOT perform spoofing detection.
It only prepares and validates the real GNSS observations.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WGS-84
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def geodetic_to_ecef(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_m: np.ndarray,
) -> np.ndarray:
    """
    Convert latitude/longitude/altitude to ECEF coordinates.

    Returns
    -------
    np.ndarray
        Shape (N, 3), columns = X, Y, Z in meters.
    """

    lat = np.radians(latitude_deg)
    lon = np.radians(longitude_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    prime_vertical_radius = WGS84_A / np.sqrt(
        1.0 - WGS84_E2 * sin_lat**2
    )

    x = (prime_vertical_radius + altitude_m) * cos_lat * np.cos(lon)

    y = (prime_vertical_radius + altitude_m) * cos_lat * np.sin(lon)

    z = (
        (prime_vertical_radius * (1.0 - WGS84_E2) + altitude_m)
        * sin_lat
    )

    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Satellite identifier
# ---------------------------------------------------------------------------

def make_satellite_id(df: pd.DataFrame) -> pd.Series:
    """
    Create a unique identifier for a constellation + satellite + signal.

    Example:
        GPS_2_GPS_L1
        GPS_2_GPS_L5
        GAL_12_GAL_E1
    """

    constellation_map = {
        1: "GPS",
        3: "GLO",
        6: "GAL",
    }

    constellation = (
        df["ConstellationType"]
        .map(constellation_map)
        .fillna("UNKNOWN")
    )

    return (
        constellation.astype(str)
        + "_"
        + df["Svid"].astype(int).astype(str)
        + "_"
        + df["SignalType"].astype(str)
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_google_sample(
    device_path: str | Path,
    ground_truth_path: str | Path,
) -> pd.DataFrame:
    """
    Load one Google Smartphone Decimeter Challenge sample.

    Parameters
    ----------
    device_path:
        Path to device_gnss.csv.

    ground_truth_path:
        Path to ground_truth.csv.

    Returns
    -------
    pd.DataFrame
        One row per real GNSS signal observation.
    """

    device_path = Path(device_path)
    ground_truth_path = Path(ground_truth_path)

    print(f"Loading device GNSS data: {device_path}")
    device = pd.read_csv(device_path)

    print(f"Loading ground truth: {ground_truth_path}")
    truth = pd.read_csv(ground_truth_path)

    # ---------------------------------------------------------------
    # Keep only raw GNSS measurement rows
    # ---------------------------------------------------------------

    if "MessageType" in device.columns:
        device = device[device["MessageType"] == "Raw"].copy()

    # ---------------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------------

    required_device = [
        "utcTimeMillis",
        "Svid",
        "ConstellationType",
        "SignalType",
        "RawPseudorangeMeters",
        "Cn0DbHz",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
    ]

    missing = [
        col for col in required_device
        if col not in device.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required device columns: {missing}"
        )

    required_truth = [
        "UnixTimeMillis",
        "LatitudeDegrees",
        "LongitudeDegrees",
    ]

    missing = [
        col for col in required_truth
        if col not in truth.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required ground-truth columns: {missing}"
        )

    # ---------------------------------------------------------------
    # Rename ground-truth timestamp so it can be merged
    # ---------------------------------------------------------------

    truth = truth.rename(
        columns={"UnixTimeMillis": "utcTimeMillis"}
    )

    # ---------------------------------------------------------------
    # Remove duplicate ground-truth timestamps if any
    # ---------------------------------------------------------------

    truth = truth.drop_duplicates(
        subset=["utcTimeMillis"]
    )

    # ---------------------------------------------------------------
    # Merge GNSS measurements with ground truth
    # ---------------------------------------------------------------

    df = device.merge(
        truth[
            [
                "utcTimeMillis",
                "LatitudeDegrees",
                "LongitudeDegrees",
                "AltitudeMeters",
                "SpeedMps",
                "AccuracyMeters",
                "BearingDegrees",
            ]
        ],
        on="utcTimeMillis",
        how="inner",
    )

    # ---------------------------------------------------------------
    # Remove observations without the required numerical data
    # ---------------------------------------------------------------

    required_numeric = [
        "RawPseudorangeMeters",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "Cn0DbHz",
        "LatitudeDegrees",
        "LongitudeDegrees",
    ]

    df = df.dropna(
        subset=required_numeric
    ).copy()

    # ---------------------------------------------------------------
    # Ground-truth altitude
    #
    # Some Google samples contain NaN altitude.
    # For the first integration we use zero where altitude is missing.
    # This is explicitly recorded in the output.
    # ---------------------------------------------------------------

    altitude = df["AltitudeMeters"].fillna(0.0).to_numpy()

    receiver_ecef = geodetic_to_ecef(
        df["LatitudeDegrees"].to_numpy(),
        df["LongitudeDegrees"].to_numpy(),
        altitude,
    )

    df["receiver_x_true_m"] = receiver_ecef[:, 0]
    df["receiver_y_true_m"] = receiver_ecef[:, 1]
    df["receiver_z_true_m"] = receiver_ecef[:, 2]

    # ---------------------------------------------------------------
    # Satellite ECEF coordinates
    # ---------------------------------------------------------------

    sat_x = df["SvPositionXEcefMeters"].to_numpy()
    sat_y = df["SvPositionYEcefMeters"].to_numpy()
    sat_z = df["SvPositionZEcefMeters"].to_numpy()

    # ---------------------------------------------------------------
    # True geometric satellite-receiver range
    # ---------------------------------------------------------------

    true_range = np.sqrt(
        (sat_x - receiver_ecef[:, 0]) ** 2
        + (sat_y - receiver_ecef[:, 1]) ** 2
        + (sat_z - receiver_ecef[:, 2]) ** 2
    )

    df["true_range_m"] = true_range

    # ---------------------------------------------------------------
    # Pseudorange residual
    #
    # NOTE:
    # Raw pseudorange includes receiver/satellite clock and propagation
    # effects, so this is NOT expected to be zero.
    #
    # It is a raw geometric residual useful for exploration.
    # ---------------------------------------------------------------

    # Preserve the original smartphone measurement.
    df["raw_pseudorange_m"] = df["RawPseudorangeMeters"]

    # Correct the dominant satellite-clock contribution.
    #
    # From the Google measurements:
    # raw_pseudorange - geometric_range ≈ -SvClockBiasMeters
    #
    # Therefore:
    # corrected_pseudorange ≈ raw_pseudorange + SvClockBiasMeters
    #
    # This makes the measurement closer to the simplified model used
    # by navigation/position_solver.py:
    #
    # pseudorange ≈ geometric_range + receiver_clock_bias + noise

    df["pseudorange_m"] = (
        df["raw_pseudorange_m"]
        + df["SvClockBiasMeters"]
    )

    # Keep this only as a diagnostic quantity.
    df["pseudorange_residual_m"] = (
        df["pseudorange_m"] - df["true_range_m"]
    )

    # ---------------------------------------------------------------
    # Time
    #
    # Convert absolute Unix milliseconds into seconds since the first
    # observation, matching the relative-time convention used by the
    # simulator.
    # ---------------------------------------------------------------

    first_time_ms = df["utcTimeMillis"].min()

    df["t"] = (
        (df["utcTimeMillis"] - first_time_ms) / 1000.0
    )

    # ---------------------------------------------------------------
    # Satellite identifier
    # ---------------------------------------------------------------

    df["satellite_id"] = make_satellite_id(df)

    # ---------------------------------------------------------------
    # Common NavIC-SHIELD names
    # ---------------------------------------------------------------

    df["x"] = df["SvPositionXEcefMeters"]
    df["y"] = df["SvPositionYEcefMeters"]
    df["z"] = df["SvPositionZEcefMeters"]

    df["elevation_deg"] = df["SvElevationDegrees"]
    df["azimuth_deg"] = df["SvAzimuthDegrees"]
    df["cn0_db_hz"] = df["Cn0DbHz"]

    # Real GNSS measurements are assumed visible because they were
    # recorded as satellite observations.
    df["visible"] = True

    # No spoofing labels exist in this real dataset.
    df["is_spoofed"] = False
    df["attack_type"] = "none"

    # ---------------------------------------------------------------
    # Select clean output columns
    # ---------------------------------------------------------------

    output_columns = [
        "t",
        "utcTimeMillis",
        "satellite_id",
        "ConstellationType",
        "SignalType",
        "Svid",

        "x",
        "y",
        "z",

        "elevation_deg",
        "azimuth_deg",
        "cn0_db_hz",

        "raw_pseudorange_m",
        "pseudorange_m",
        "SvClockBiasMeters",
        "true_range_m",
        "pseudorange_residual_m",

        "receiver_x_true_m",
        "receiver_y_true_m",
        "receiver_z_true_m",

        "LatitudeDegrees",
        "LongitudeDegrees",
        "AltitudeMeters",

        "SpeedMps",
        "AccuracyMeters",
        "BearingDegrees",

        "visible",
        "is_spoofed",
        "attack_type",
    ]

    output_columns = [
        col for col in output_columns
        if col in df.columns
    ]

    result = df[output_columns].copy()

    # Sort exactly as a GNSS time series should be ordered.
    result = result.sort_values(
        ["t", "satellite_id"]
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def summarize_real_data(df: pd.DataFrame) -> None:
    """Print basic validation statistics."""

    print("\n" + "=" * 60)
    print("REAL GNSS DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows:                 {len(df):,}")
    print(f"Epochs:               {df['t'].nunique():,}")
    print(f"Satellite/signals:    {df['satellite_id'].nunique():,}")

    print(
        f"Time span:            "
        f"{df['t'].min():.1f} -> {df['t'].max():.1f} s"
    )

    print(
        f"Pseudorange range:    "
        f"{df['pseudorange_m'].min():.2f} -> "
        f"{df['pseudorange_m'].max():.2f} m"
    )

    print(
        f"C/N0 range:           "
        f"{df['cn0_db_hz'].min():.2f} -> "
        f"{df['cn0_db_hz'].max():.2f} dB-Hz"
    )

    print(
        f"Elevation range:      "
        f"{df['elevation_deg'].min():.2f} -> "
        f"{df['elevation_deg'].max():.2f} deg"
    )

    print(
        f"Residual mean:        "
        f"{df['pseudorange_residual_m'].mean():.2f} m"
    )

    print(
        f"Residual median:      "
        f"{df['pseudorange_residual_m'].median():.2f} m"
    )

    print("\nConstellations:")

    print(
        df["ConstellationType"]
        .value_counts()
        .sort_index()
    )

    print("\nSignal types:")

    print(
        df["SignalType"]
        .value_counts()
    )

    print("\nObservations per epoch:")

    print(
        df.groupby("t")
        .size()
        .describe()
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[3]

    device_path = (
        project_root
        / "data"
        / "real_gnss"
        / "google"
        / "sample"
        / "device_gnss.csv"
    )

    truth_path = (
        project_root
        / "data"
        / "real_gnss"
        / "google"
        / "sample"
        / "ground_truth.csv"
    )

    output_path = (
        project_root
        / "data"
        / "real_gnss"
        / "google"
        / "sample"
        / "normalized_gnss.csv"
    )

    real_df = load_google_sample(
        device_path,
        truth_path,
    )

    summarize_real_data(real_df)

    real_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved normalized dataset to:\n"
        f"{output_path}"
    )