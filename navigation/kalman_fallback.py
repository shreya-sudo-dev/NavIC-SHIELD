"""
navigation/kalman_fallback.py

Constant-POSITION Kalman filter (random-walk model, no velocity state)
over the solved receiver position (navigation/position_solver.py
output). Deliberately NOT constant-velocity: this project's receiver is
a known-stationary border site, so the correct fallback assumption
during untrusted GNSS is "position hasn't changed" -- not "extrapolate
whatever noise-driven velocity was last estimated." A constant-velocity
model was tried first and found to extrapolate spurious velocity
indefinitely once measurement updates were gated off, producing an
unbounded linear runaway during sustained attacks -- see project notes.

Two layers of defense against a spoofed measurement, both necessary:
  1. Confidence-weighted measurement noise (R) -- soft down-weighting,
     scales trust continuously with fusion confidence.
  2. Hard reject_threshold gate -- soft weighting ALONE still lets a
     small, nonzero Kalman gain accumulate toward a persistently biased
     measurement over many repeated updates (any nonzero R eventually
     converges given enough samples). The hard gate is what actually
     blocks a SUSTAINED attack from leaking into the state at all.

State: [x, y, z] in ECEF, meters.
"""

import numpy as np
import pandas as pd


class KalmanFallback:
    def __init__(self, initial_position, dt,
                 process_noise_std=0.5,       # meters, position random-walk per sqrt(second)
                 base_measurement_noise_std=8.5,  # from your solver's own clean-data result
                 min_confidence_noise_multiplier=1.0,
                 max_confidence_noise_multiplier=200.0,
                 reject_threshold=0.15):
        self.state = np.array(initial_position, dtype=float)  # just [x, y, z] now
        self.process_noise_std = process_noise_std
        self.base_measurement_noise_std = base_measurement_noise_std
        self.min_mult = min_confidence_noise_multiplier
        self.max_mult = max_confidence_noise_multiplier
        self.reject_threshold = reject_threshold

        self.P = np.eye(3) * (base_measurement_noise_std ** 2)
        self.history = []

    def predict(self, dt):
        # No motion model beyond "position may drift slightly" -- appropriate
        # for a KNOWN-stationary receiver, unlike constant-velocity which
        # extrapolates spurious noise-driven velocity indefinitely once
        # measurement updates stop.
        Q = np.eye(3) * (self.process_noise_std ** 2) * dt
        self.P = self.P + Q  # F = identity, so no state change here

    def confidence_to_measurement_noise(self, confidence):
        confidence = np.clip(confidence, 0.0, 1.0)
        multiplier = self.max_mult + confidence * (self.min_mult - self.max_mult)
        return self.base_measurement_noise_std * multiplier

    def update(self, measured_position, confidence):
        if measured_position is None or np.any(np.isnan(measured_position)) \
           or confidence < self.reject_threshold:
            self.history.append({"updated": False, "confidence": confidence})
            return

        meas_noise_std = self.confidence_to_measurement_noise(confidence)
        R = np.eye(3) * (meas_noise_std ** 2)

        y = np.array(measured_position) - self.state
        S = self.P + R
        K = self.P @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(3) - K) @ self.P

        self.history.append({"updated": True, "confidence": confidence,
                              "measurement_noise_std": meas_noise_std,
                              "innovation_norm": np.linalg.norm(y)})

    def position(self):
        return self.state.copy()

    
def run_kalman_over_timeseries(solved_position_df: pd.DataFrame,
                                confidence_series: pd.Series,
                                initial_position: np.ndarray,
                                **kalman_kwargs) -> pd.DataFrame:
    """
    Run the confidence-weighted Kalman filter over a full timeseries of
    solved positions + per-epoch confidence scores.

    Parameters
    ----------
    solved_position_df : pd.DataFrame
        Output of position_solver.solve_position_timeseries -- must
        contain t, solved_x, solved_y, solved_z (NaN rows are handled
        as missing measurements, see KalmanFallback.update).
    confidence_series : pd.Series
        Indexed the same way as solved_position_df (or same length,
        same order) -- confidence in [0, 1] per epoch. HIGH = trust
        GNSS, LOW = distrust (suspected spoofed).
    initial_position : np.ndarray, shape (3,)
        Starting position for the filter.
    **kalman_kwargs
        Passed through to KalmanFallback's constructor (process noise,
        measurement noise, etc).

    Returns
    -------
    pd.DataFrame with t, kalman_x, kalman_y, kalman_z, raw_x, raw_y,
    raw_z (the unfiltered solved position, for comparison), confidence.
    """
    df = solved_position_df.reset_index(drop=True).copy()
    df["confidence"] = confidence_series.reset_index(drop=True).values \
        if hasattr(confidence_series, "reset_index") else np.asarray(confidence_series)

    ts = df["t"].values
    dts = np.diff(ts, prepend=ts[0])
    dts[0] = dts[1] if len(dts) > 1 else 1.0

    kf = KalmanFallback(initial_position, dt=float(np.median(dts[1:])) if len(dts) > 1 else 30.0,
                         **kalman_kwargs)

    kalman_positions = []
    for i, row in df.iterrows():
        kf.predict(float(dts[i]))

        measured = None
        if not np.isnan(row["solved_x"]):
            measured = np.array([row["solved_x"], row["solved_y"], row["solved_z"]])

        kf.update(measured, row["confidence"])
        kalman_positions.append(kf.position())

    kalman_positions = np.array(kalman_positions)
    df["kalman_x"] = kalman_positions[:, 0]
    df["kalman_y"] = kalman_positions[:, 1]
    df["kalman_z"] = kalman_positions[:, 2]
    df["raw_x"] = df["solved_x"]
    df["raw_y"] = df["solved_y"]
    df["raw_z"] = df["solved_z"]

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
    from simulator.receiver import _DEFAULT_RECEIVER_ECEF
    from spoofing.step_attack import apply_step_attack
    from navigation.position_solver import solve_position_timeseries

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
    attack_start = duration / 2
    df = apply_step_attack(df, target_sat, attack_start, step_size_m=500.0)

    print("Solving position timeseries (with a step spoof injected on one satellite)...")
    solved = solve_position_timeseries(df, _DEFAULT_RECEIVER_ECEF)

    # Fake confidence series for this sanity check: 1.0 before the
    # attack, 0.05 during it (simulating a fusion model that correctly
    # flags the spoofed satellite -- real integration would come from
    # models/fusion.py's actual per-epoch confidence output)
    confidence = pd.Series(
        np.where(solved["t"] < attack_start, 1.0, 0.05), index=solved.index
    )

    result = run_kalman_over_timeseries(solved, confidence, _DEFAULT_RECEIVER_ECEF)

    true_pos = _DEFAULT_RECEIVER_ECEF
    result["raw_error_m"] = np.linalg.norm(
        result[["raw_x", "raw_y", "raw_z"]].values - true_pos, axis=1
    )
    result["kalman_error_m"] = np.linalg.norm(
        result[["kalman_x", "kalman_y", "kalman_z"]].values - true_pos, axis=1
    )

    print(f"\nMean position error DURING attack (raw/uncorrected): "
          f"{result[result['t'] >= attack_start]['raw_error_m'].mean():.1f} m")
    print(f"Mean position error DURING attack (Kalman-corrected):  "
          f"{result[result['t'] >= attack_start]['kalman_error_m'].mean():.1f} m")
    print("(expect Kalman error to stay much smaller -- it's leaning on motion "
          "continuity instead of the spoofed measurement)")

    plt.figure(figsize=(10, 5))
    plt.plot(result["t"] / 60, result["raw_error_m"], label="raw (uncorrected)", alpha=0.7)
    plt.plot(result["t"] / 60, result["kalman_error_m"], label="Kalman-corrected", alpha=0.9)
    plt.axvline(attack_start / 60, color="red", linestyle="--", label="attack start")
    plt.xlabel("time (minutes)")
    plt.ylabel("position error vs true receiver location (m)")
    plt.title("Kalman fallback: bounded error during a spoofing event")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("kalman_fallback_check.png", dpi=120)
    print("\nSaved kalman_fallback_check.png")