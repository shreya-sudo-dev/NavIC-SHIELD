"""
simulator/constellation.py

Propagates all NavIC satellites (GEO + GSO) as ECEF positions over a
simulation time window.

Two-step pipeline per satellite, per timestep:
  1. orbital_elements_to_eci  — Keplerian propagation -> ECI position
  2. eci_to_ecef              — rotate ECI -> ECEF using elapsed Earth
                                 rotation since epoch

The ECI->ECEF step is where the phase-offset/Earth-rotation trap lives:
the rotation angle MUST be a function of elapsed simulation time, not a
fixed/static offset, or satellites will appear to drift.
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def orbital_elements_to_eci(sat_params: dict, t: float) -> np.ndarray:
    """
    Propagate a single satellite's position in ECI coordinates at time t.

    Assumes a circular (or near-circular) orbit, which is a valid
    simplification for NavIC GEO/GSO satellites. Uses mean anomaly ==
    true anomaly (valid for e ~ 0).

    Parameters
    ----------
    sat_params : dict
        One entry from config.NAVIC_SATELLITES. Must contain 'sma',
        'inclination_deg', 'raan_deg', 'mean_anomaly_epoch_deg'.
    t : float
        Seconds elapsed since simulation epoch (t=0).

    Returns
    -------
    np.ndarray, shape (3,)
        Position in ECI frame, meters.
    """
    sma = sat_params["sma"]
    inc = np.radians(sat_params["inclination_deg"])
    raan = np.radians(sat_params["raan_deg"])
    m0 = np.radians(sat_params["mean_anomaly_epoch_deg"])

    # Mean motion for the simplified geosynchronous NavIC model.
    # Use Earth's rotation rate so that a GEO satellite remains
    # stationary in the ECEF frame.
    n = np.sqrt(config.EARTH_MU / sma ** 3)

    # Mean anomaly at time t (== true anomaly under circular-orbit assumption)
    theta = m0 + n * t

    # Position in the orbital plane (perifocal frame), circular orbit
    x_orb = sma * np.cos(theta)
    y_orb = sma * np.sin(theta)
    z_orb = 0.0

    # Rotate from orbital plane into ECI using inclination and RAAN.
    # Argument of perigee is irrelevant/undefined for a circular orbit,
    # so we go straight from perifocal (x_orb, y_orb, 0) through the
    # inclination rotation (about x-axis) then the RAAN rotation
    # (about z-axis).
    x1 = x_orb
    y1 = y_orb * np.cos(inc)
    z1 = y_orb * np.sin(inc)

    x_eci = x1 * np.cos(raan) - y1 * np.sin(raan)
    y_eci = x1 * np.sin(raan) + y1 * np.cos(raan)
    z_eci = z1

    return np.array([x_eci, y_eci, z_eci])


def eci_to_ecef(position_eci: np.ndarray, t: float) -> np.ndarray:
    """
    Rotate an ECI position into ECEF by the Earth's rotation accumulated
    since epoch (t=0). This is the trap: the rotation angle must scale
    with elapsed time, not be a fixed constant.

    Parameters
    ----------
    position_eci : np.ndarray, shape (3,)
        Position in ECI frame, meters.
    t : float
        Seconds elapsed since simulation epoch (t=0). At t=0, ECI and
        ECEF frames are assumed aligned (theta_earth(0) = 0).

    Returns
    -------
    np.ndarray, shape (3,)
        Position in ECEF frame, meters.
    """
    theta = config.EARTH_ROTATION_RATE * t   # accumulated rotation angle

    # ECEF = Rz(+theta) applied to ECI (rotating ECI *backwards* relative
    # to the rotating Earth frame is equivalent to rotating the vector by
    # +theta into ECEF). Standard convention: ECEF = R3(theta) * ECI.
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_eci, y_eci, z_eci = position_eci

    x_ecef = cos_t * x_eci + sin_t * y_eci
    y_ecef = -sin_t * x_eci + cos_t * y_eci
    z_ecef = z_eci

    return np.array([x_ecef, y_ecef, z_ecef])


def propagate_constellation(t_start: float, t_end: float, dt: float) -> pd.DataFrame:
    """
    Propagate all satellites in config.NAVIC_SATELLITES over
    [t_start, t_end] in steps of dt.

    Returns
    -------
    pd.DataFrame with columns: satellite_id, t, x, y, z (ECEF, meters)
    """
    timestamps = np.arange(t_start, t_end, dt)
    records = []

    for sat in config.NAVIC_SATELLITES:
        for t in timestamps:
            pos_eci = orbital_elements_to_eci(sat, t)
            pos_ecef = eci_to_ecef(pos_eci, t)
            records.append({
                "satellite_id": sat["id"],
                "type": sat["type"],
                "t": t,
                "x": pos_ecef[0],
                "y": pos_ecef[1],
                "z": pos_ecef[2],
            })

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Sanity checks (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    period = config.SIDEREAL_DAY_SECONDS
    dt = config.SIM_TIMESTEP_S

    df = propagate_constellation(0, period, dt)

    # --- Check 1: ECI orbit shape for one GSO satellite (should be a
    #     closed circle/ellipse) ---
    gso_sat = next(s for s in config.NAVIC_SATELLITES if s["type"] == "GSO")
    ts = np.arange(0, period, dt)
    eci_positions = np.array([orbital_elements_to_eci(gso_sat, t) for t in ts])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].plot(eci_positions[:, 0], eci_positions[:, 1])
    axes[0].set_title(f"ECI orbit (x-y plane): {gso_sat['id']}")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].axis("equal")

    # --- Check 2: ECEF ground track for the same GSO satellite (should
    #     trace a figure-8, NOT a drifting spiral) ---
    gso_ecef = df[df.satellite_id == gso_sat["id"]]
    axes[1].plot(gso_ecef["x"], gso_ecef["y"])
    axes[1].set_title(f"ECEF track (x-y plane): {gso_sat['id']}")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].axis("equal")

    # --- Check 3: GEO stationarity — ECEF position should barely move ---
    geo_sat = next(s for s in config.NAVIC_SATELLITES if s["type"] == "GEO")
    geo_ecef = df[df.satellite_id == geo_sat["id"]][["x", "y", "z"]].values
    geo_deviation = np.linalg.norm(geo_ecef - geo_ecef[0], axis=1)

    axes[2].plot(ts / 3600, geo_deviation)
    axes[2].set_title(f"GEO stationarity check: {geo_sat['id']}")
    axes[2].set_xlabel("time (hours)")
    axes[2].set_ylabel("deviation from t=0 position (m)")

    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=120)
    print(f"GEO satellite ({geo_sat['id']}) max ECEF deviation over "
          f"{period/3600:.2f} hours: {geo_deviation.max():.2f} m")
    print("Expect this to be near-zero (numerical-precision scale, "
          "not kilometers). If it's large or grows over time, the "
          "ECI->ECEF rotation direction or rate is wrong.")

    # --- Check 4: orbital period check via ascending-node crossing ---
    # For the GSO satellite, find when z crosses zero going positive (ECI)
    z_vals = eci_positions[:, 2]
    sign_changes = np.where((z_vals[:-1] < 0) & (z_vals[1:] >= 0))[0]
    if len(sign_changes) >= 2:
        crossing_times = ts[sign_changes]
        measured_period = np.mean(np.diff(crossing_times))
        print(f"Measured orbital period (ascending node crossings): "
              f"{measured_period:.2f} s")
        print(f"Expected sidereal period: {config.SIDEREAL_DAY_SECONDS:.2f} s "
              f"(NOT 86400 s solar day)")
    else:
        print("Not enough ascending-node crossings found in one period "
              "window to measure — check timestep/duration.")
