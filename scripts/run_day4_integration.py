"""
run_day4_integration.py

Full Day 4 integration: real fusion confidence -> Kalman fallback,
against the actual staggered step/drift/evasive attacks.

Emits exactly the four files dashboard/app.py reads, to results/csv/:
  - day4_position_results.csv  -- per-epoch solved/raw/Kalman position + error
  - day4_satellite_level.csv   -- per-satellite, per-epoch az/el/visibility/
                                   fusion spoof probability (drives the
                                   skyplot + attack monitor tabs)
  - day4_attack_info.csv       -- which satellite got which attack, when,
                                   for how long (NaN duration = unbounded)
  - day4_summary.csv           -- headline raw-vs-Kalman error by attack type

Run from the project root:
    python run_day4_integration.py
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
from simulator.receiver import _DEFAULT_RECEIVER_ECEF
from spoofing.step_attack import apply_step_attack
from spoofing.drift_attack import apply_drift_attack
from spoofing.evasive_attack import apply_evasive_attack
from features.temporal import compute_temporal_features
from features.spatial import compute_spatial_features
from models.baseline_detector import prepare_training_data, train_baseline_detector
from models.fusion import (
    add_temporal_scores, prepare_fusion_data, train_fusion_detector, FUSION_INPUT_COLUMNS,
)
from navigation.position_solver import solve_position_timeseries
from navigation.kalman_fallback import run_kalman_over_timeseries

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "csv")

STEP_SIZE_M = 200.0
STEP_DURATION_S = 7200
DRIFT_RATE_M_PER_S = 1.0
DRIFT_DURATION_S = 7200
EVASIVE_TARGET_OFFSET_M = 150.0
EVASIVE_RAMP_DURATION_S = 2 * 3600

TEST_WINDOW_S = 45 * 60
TRAIN_GAP_S = 300
KALMAN_REJECT_THRESHOLD = 0.15


def simulate_and_attack():
    duration = config.SIDEREAL_DAY_SECONDS
    dt = config.SIM_TIMESTEP_S

    print("Propagating constellation, computing terrain visibility, observations...")
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

    sat_ids = df[df["visible"]].groupby("satellite_id").size().sort_values(ascending=False).index.tolist()
    if len(sat_ids) < 5:
        raise RuntimeError(f"Only {len(sat_ids)} satellites visible -- need at least 5.")

    step_sat, drift_sat, evasive_sat = sat_ids[0], sat_ids[1], sat_ids[2]
    clean_sat_train, clean_sat_test = sat_ids[3], sat_ids[4]

    step_start = duration * 0.4
    drift_start = step_start + 4 * 3600
    evasive_start = drift_start + 4 * 3600

    print(f"Attacks: step->{step_sat}@{step_start/3600:.1f}h, "
          f"drift->{drift_sat}@{drift_start/3600:.1f}h, "
          f"evasive->{evasive_sat}@{evasive_start/3600:.1f}h")

    df = apply_step_attack(df, step_sat, step_start, step_size_m=STEP_SIZE_M,
                            duration_s=STEP_DURATION_S)
    df = apply_drift_attack(df, drift_sat, drift_start, drift_rate_m_per_s=DRIFT_RATE_M_PER_S,
                             duration_s=DRIFT_DURATION_S)
    df = apply_evasive_attack(df, evasive_sat, evasive_start,
                               target_offset_m=EVASIVE_TARGET_OFFSET_M,
                               ramp_duration_s=EVASIVE_RAMP_DURATION_S)

    attack_info_rows = [
        {"satellite_id": step_sat, "attack_type": "step",
         "start_s": step_start, "duration_s": STEP_DURATION_S},
        {"satellite_id": drift_sat, "attack_type": "drift",
         "start_s": drift_start, "duration_s": DRIFT_DURATION_S},
        {"satellite_id": evasive_sat, "attack_type": "evasive",
         "start_s": evasive_start, "duration_s": np.nan},  # unbounded -- holds to end of day
    ]
    attack_info_df = pd.DataFrame(attack_info_rows)

    attack_info = {
        "step_sat": step_sat, "drift_sat": drift_sat, "evasive_sat": evasive_sat,
        "clean_sat_train": clean_sat_train, "clean_sat_test": clean_sat_test,
        "step_start": step_start, "drift_start": drift_start, "evasive_start": evasive_start,
        "true_attack_starts": {step_sat: step_start, drift_sat: drift_start,
                                evasive_sat: evasive_start},
    }
    return df, attack_info, attack_info_df, duration


def train_detectors(df, attack_info):
    step_sat, drift_sat, evasive_sat = (attack_info["step_sat"], attack_info["drift_sat"],
                                         attack_info["evasive_sat"])
    clean_sat_train, clean_sat_test = attack_info["clean_sat_train"], attack_info["clean_sat_test"]
    true_attack_starts = attack_info["true_attack_starts"]

    def in_test_window(row):
        ts = true_attack_starts.get(row["satellite_id"])
        if ts is None:
            return False
        return ts <= row["t"] < ts + TEST_WINDOW_S

    def in_train_window(row):
        ts = true_attack_starts.get(row["satellite_id"])
        if ts is None:
            return True
        return row["t"] < ts or row["t"] >= ts + TEST_WINDOW_S + TRAIN_GAP_S

    is_attacked_sat = df["satellite_id"].isin([step_sat, drift_sat, evasive_sat])
    train_mask = (
        (is_attacked_sat & df.apply(in_train_window, axis=1))
        | (df["satellite_id"] == clean_sat_train)
    )

    temporal_clean, _, _, _ = prepare_training_data(df[train_mask])
    print(f"Training temporal-only baseline on {len(temporal_clean)} rows...")
    temporal_clf = train_baseline_detector(temporal_clean)

    df = add_temporal_scores(df, temporal_clf)

    fusion_train_clean, _, _ = prepare_fusion_data(df[train_mask])
    print(f"Training fusion model on {len(fusion_train_clean)} rows...")
    fusion_clf = train_fusion_detector(fusion_train_clean)

    return df, temporal_clf, fusion_clf


def score_every_row(df, fusion_clf):
    df = df.copy()
    df["fusion_spoof_prob"] = np.nan
    has_inputs = df[FUSION_INPUT_COLUMNS].notna().all(axis=1)
    X = df.loc[has_inputs, FUSION_INPUT_COLUMNS].values
    df.loc[has_inputs, "fusion_spoof_prob"] = fusion_clf.predict_proba(X)[:, 1]
    return df


def collapse_to_epoch_confidence(df):
    visible_df = df[df["visible"]]
    max_spoof_per_epoch = visible_df.groupby("t")["fusion_spoof_prob"].max()
    confidence_per_epoch = (1.0 - max_spoof_per_epoch).clip(0.0, 1.0)
    return confidence_per_epoch.fillna(1.0)


def main():
    df, attack_info, attack_info_df, duration = simulate_and_attack()

    print("Computing temporal + spatial features...")
    df = compute_temporal_features(df)
    df = compute_spatial_features(df)

    df, temporal_clf, fusion_clf = train_detectors(df, attack_info)

    print("Scoring every row with the fusion model...")
    df = score_every_row(df, fusion_clf)

    print("Collapsing per-satellite spoof probability into per-epoch confidence...")
    confidence_per_epoch = collapse_to_epoch_confidence(df)

    print("Solving receiver position for every epoch...")
    solved = solve_position_timeseries(df, _DEFAULT_RECEIVER_ECEF)

    solved = solved.merge(
        confidence_per_epoch.rename("confidence").reset_index(), on="t", how="left",
    )
    solved["confidence"] = solved["confidence"].fillna(1.0)

    print(f"Confidence stats: mean={solved['confidence'].mean():.3f}, "
          f"min={solved['confidence'].min():.3f}, "
          f"epochs below reject threshold: "
          f"{(solved['confidence'] < KALMAN_REJECT_THRESHOLD).sum()} / {len(solved)}")

    print("Running Kalman fallback with real fusion confidence...")
    result = run_kalman_over_timeseries(
        solved, solved["confidence"], _DEFAULT_RECEIVER_ECEF,
        reject_threshold=KALMAN_REJECT_THRESHOLD,
    )

    true_pos = _DEFAULT_RECEIVER_ECEF
    result["raw_error_m"] = np.linalg.norm(
        result[["raw_x", "raw_y", "raw_z"]].values - true_pos, axis=1
    )
    result["kalman_error_m"] = np.linalg.norm(
        result[["kalman_x", "kalman_y", "kalman_z"]].values - true_pos, axis=1
    )

    # --- Per-attack-window summary ---
    print("\n===== POSITION ERROR BY ATTACK WINDOW =====")
    windows = {
        "step": (attack_info["step_start"], attack_info["step_start"] + STEP_DURATION_S),
        "drift": (attack_info["drift_start"], attack_info["drift_start"] + DRIFT_DURATION_S),
        "evasive": (attack_info["evasive_start"], attack_info["evasive_start"] + TEST_WINDOW_S),
    }
    summary_rows = []
    for name, (start, end) in windows.items():
        window_df = result[(result["t"] >= start) & (result["t"] < end)]
        if len(window_df) == 0:
            continue
        summary_rows.append({
            "attack": name,
            "n_epochs": len(window_df),
            "raw_mean_error_m": window_df["raw_error_m"].mean(),
            "kalman_mean_error_m": window_df["kalman_error_m"].mean(),
            "raw_max_error_m": window_df["raw_error_m"].max(),
            "kalman_max_error_m": window_df["kalman_error_m"].max(),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # --- Save all four files the dashboard needs ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result.to_csv(os.path.join(RESULTS_DIR, "day4_position_results.csv"), index=False)
    print(f"\nSaved day4_position_results.csv ({len(result)} rows)")

    sat_level_cols = ["t", "satellite_id", "visible", "azimuth_deg", "elevation_deg",
                       "fusion_spoof_prob"]
    df[sat_level_cols].to_csv(os.path.join(RESULTS_DIR, "day4_satellite_level.csv"), index=False)
    print(f"Saved day4_satellite_level.csv ({len(df)} rows)")

    attack_info_df.to_csv(os.path.join(RESULTS_DIR, "day4_attack_info.csv"), index=False)
    print(f"Saved day4_attack_info.csv ({len(attack_info_df)} rows)")

    summary_df.to_csv(os.path.join(RESULTS_DIR, "day4_summary.csv"), index=False)
    print(f"Saved day4_summary.csv ({len(summary_df)} rows)")

    print(f"\nAll files in {RESULTS_DIR} -- ready for `streamlit run dashboard/app.py`")


if __name__ == "__main__":
    main()