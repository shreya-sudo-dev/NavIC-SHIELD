"""
Geodetic <-> ECEF conversion, and topocentric azimuth/elevation
computation from a fixed ground receiver to arbitrary ECEF satellite
positions.
 
terrain.py depends directly on the elevation output of this module,
so the sign convention matters: positive elevation = above horizon.
"""
 
import numpy as np
 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
 
 
def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """
    WGS84 geodetic (lat, lon, alt) -> ECEF (x, y, z), meters.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
 
    a = config.WGS84_A
    e2 = config.WGS84_E2
 
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
 
    x = (n + alt_m) * np.cos(lat) * np.cos(lon)
    y = (n + alt_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - e2) + alt_m) * np.sin(lat)
 
    return np.array([x, y, z])
 
 
def ecef_to_geodetic(x: float, y: float, z: float, tol: float = 1e-12,
                      max_iter: int = 20) -> tuple:
    """
    ECEF -> WGS84 geodetic (lat_deg, lon_deg, alt_m).
 
    Iterative (Bowring-style) inverse. Only used for the round-trip
    sanity check below — production code only needs the forward
    direction (geodetic_to_ecef).
    """
    a = config.WGS84_A
    e2 = config.WGS84_E2
 
    lon = np.arctan2(y, x)
 
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - e2))  # initial guess
 
    for _ in range(max_iter):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat_new = np.arctan2(z, p * (1 - e2 * n / (n + alt)))
        if abs(lat_new - lat) < tol:
            lat = lat_new
            break
        lat = lat_new
 
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
 
    return np.degrees(lat), np.degrees(lon), alt
 
 
def _enu_rotation_matrix(receiver_lat_deg: float, receiver_lon_deg: float) -> np.ndarray:
    """
    Build the fixed 3x3 rotation matrix that transforms an ECEF
    difference vector into local East-North-Up (ENU) components, for a
    receiver at given lat/lon. Since the receiver is stationary, this
    matrix is constant and can be precomputed once per site.
    """
    lat = np.radians(receiver_lat_deg)
    lon = np.radians(receiver_lon_deg)
 
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
 
    # Rows: East, North, Up  (each a unit vector expressed in ECEF)
    r = np.array([
        [-sin_lon,            cos_lon,           0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat],
    ])
    return r
 
 
def ecef_to_enu(sat_ecef: np.ndarray, receiver_ecef: np.ndarray,
                 receiver_lat_deg: float, receiver_lon_deg: float) -> np.ndarray:
    """
    Convert a satellite's ECEF position into East-North-Up (ENU)
    coordinates relative to the receiver.
    """
    d = np.asarray(sat_ecef) - np.asarray(receiver_ecef)
    r = _enu_rotation_matrix(receiver_lat_deg, receiver_lon_deg)
    enu = r @ d
    return enu
 
 
def enu_to_az_el(enu_vector: np.ndarray) -> tuple:
    """
    Convert an ENU vector to (azimuth_deg, elevation_deg).
 
    azimuth: measured from North, clockwise, [0, 360)
    elevation: measured from the local horizontal plane, [-90, 90]
               positive = above horizon
    """
    e, n, u = enu_vector
    horizontal_range = np.sqrt(e ** 2 + n ** 2)
 
    elevation = np.degrees(np.arctan2(u, horizontal_range))
    azimuth = np.degrees(np.arctan2(e, n)) % 360.0
 
    return azimuth, elevation
 
 
# Precompute the default receiver's ECEF position once, since
# config.RECEIVER_SITE is fixed for this whole project.
_DEFAULT_RECEIVER_ECEF = geodetic_to_ecef(
    config.RECEIVER_SITE["lat_deg"],
    config.RECEIVER_SITE["lon_deg"],
    config.RECEIVER_SITE["alt_m"],
)
 
 
def compute_az_el(sat_ecef: np.ndarray, receiver_ecef: np.ndarray = None,
                   receiver_lat_deg: float = None,
                   receiver_lon_deg: float = None) -> tuple:
    """
    Convenience wrapper: ECEF satellite position -> (azimuth_deg, elevation_deg)
    as seen from the receiver. Defaults to config.RECEIVER_SITE if no
    receiver is given, so this can be called directly on
    constellation.py output.
    """
    if receiver_ecef is None:
        receiver_ecef = _DEFAULT_RECEIVER_ECEF
    if receiver_lat_deg is None:
        receiver_lat_deg = config.RECEIVER_SITE["lat_deg"]
    if receiver_lon_deg is None:
        receiver_lon_deg = config.RECEIVER_SITE["lon_deg"]
 
    enu = ecef_to_enu(sat_ecef, receiver_ecef, receiver_lat_deg, receiver_lon_deg)
    return enu_to_az_el(enu)
 
 
# ---------------------------------------------------------------------------
# Sanity checks (run this file directly)
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    from simulator.constellation import propagate_constellation
 
    site = config.RECEIVER_SITE
    receiver_ecef = _DEFAULT_RECEIVER_ECEF
 
    # --- Check 1: round-trip geodetic -> ECEF -> geodetic ---
    print("Round-trip check (geodetic -> ECEF -> geodetic):")
    test_points = [
        (34.1526, 77.5771, 3500.0),   # receiver site itself
        (0.0, 0.0, 0.0),              # equator/prime-meridian, sea level
        (-33.8688, 151.2093, 50.0),   # Sydney, low alt
        (89.9, 45.0, 1000.0),         # near-pole, tests numerical edge case
    ]
    max_err = 0.0
    for lat, lon, alt in test_points:
        xyz = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(*xyz)
        err_m = np.sqrt((lat - lat2) ** 2 + (lon - lon2) ** 2) * 111_000  # rough deg->m
        err_alt = abs(alt - alt2)
        max_err = max(max_err, err_m, err_alt)
        print(f"  in=({lat:.4f},{lon:.4f},{alt:.1f})  "
              f"out=({lat2:.6f},{lon2:.6f},{alt2:.4f})  "
              f"pos_err~{err_m:.2e} m  alt_err={err_alt:.2e} m")
    print(f"  Max error across test points: {max_err:.2e} m "
          f"(expect sub-millimeter, i.e. < 1e-3)")
 
    # --- Check 2: overhead / opposite-side elevation extremes ---
    print("\nElevation extremes check:")
    overhead_ecef = geodetic_to_ecef(site["lat_deg"], site["lon_deg"],
                                      config.GEOSYNCHRONOUS_SMA - config.WGS84_A)
    az, el = compute_az_el(overhead_ecef)
    print(f"  Satellite directly overhead: az={az:.2f} deg, el={el:.2f} deg "
          f"(expect el ~ 90)")
 
    opposite_lat = -site["lat_deg"]
    opposite_lon = site["lon_deg"] + 180.0
    opposite_ecef = geodetic_to_ecef(opposite_lat, opposite_lon,
                                      config.GEOSYNCHRONOUS_SMA - config.WGS84_A)
    az2, el2 = compute_az_el(opposite_ecef)
    print(f"  Satellite on opposite side of Earth: az={az2:.2f} deg, "
          f"el={el2:.2f} deg (expect el well below 0, i.e. deeply negative)")
 
    # --- Check 3: az/el over the full sidereal day for real constellation ---
    period = config.SIDEREAL_DAY_SECONDS
    dt = config.SIM_TIMESTEP_S
    df = propagate_constellation(0, period, dt)
 
    df["az"] = np.nan
    df["el"] = np.nan
    for idx, row in df.iterrows():
        az, el = compute_az_el(np.array([row.x, row.y, row.z]))
        df.at[idx, "az"] = az
        df.at[idx, "el"] = el
 
    # GEO elevation stability check
    geo_id = next(s["id"] for s in config.NAVIC_SATELLITES if s["type"] == "GEO")
    geo_el = df[df.satellite_id == geo_id]["el"]
    print(f"\nGEO satellite ({geo_id}) elevation over the day: "
          f"min={geo_el.min():.3f}, max={geo_el.max():.3f}, "
          f"range={geo_el.max() - geo_el.min():.4f} deg "
          f"(expect near-constant, i.e. small range)")
 
    gso_id = next(s["id"] for s in config.NAVIC_SATELLITES if s["type"] == "GSO")
    gso_el = df[df.satellite_id == gso_id]["el"]
    print(f"GSO satellite ({gso_id}) elevation over the day: "
          f"min={gso_el.min():.2f}, max={gso_el.max():.2f} "
          f"(expect a real sweep, not near-constant)")
 
    # --- Skyplot: az/el for all satellites over the day ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise, matches azimuth convention
    ax.set_rlim(90, 0)          # elevation 90 at center, 0 at edge
 
    for sat_id, group in df.groupby("satellite_id"):
        theta = np.radians(group["az"])
        r = 90 - group["el"]
        ax.plot(theta, r, label=sat_id, linewidth=1)
 
    ax.set_title("Skyplot: NavIC constellation over 1 sidereal day\n"
                  f"(from {site['name']})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig("skyplot.png", dpi=120)
    print("\nSaved skyplot to skyplot.png")
 
