"""
run_all.py

Consolidated end-to-end pipeline: Day 1 (simulator) -> Day 2 (spoofing +
temporal detector) -> Day 3 (spatial features + fusion), all in one run.

Does not reimplement anything -- imports and calls your existing modules
in the correct order. Purpose is to produce, in one run:
  1. results/full_dataset.csv       -- everything: geometry, observations,
                                        attack labels, temporal features,
                                        spatial features, model scores
  2. results/summary.csv            -- headline recall/F1 comparison table
                                        (temporal-only vs fusion, per attack type)
  3. Console output with the majority-spoofing diagnostic, so you know
     immediately if the results are trustworthy before using them in
     the report or moving to Day 4.

Run from the project root:
    python run_all.py
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
from spoofing.step_attack import apply_step_attack
from spoofing.drift_attack import apply_drift_attack
from spoofing.evasive_attack import apply_evasive_attack
from features.temporal import compute_temporal_features, FEATURE_COLUMNS as TEMPORAL_FEATURE_COLUMNS
from features.spatial import compute_spatial_features, SPATIAL_FEATURE_COLUMNS
from models.baseline_detector import (
    prepare_training_data, train_baseline_detector,
)
from models.fusion import (
    add_temporal_scores, prepare_fusion_data, train_fusion_detector,
    evaluate_by_attack_type, FUSION_INPUT_COLUMNS,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "csv")

# Attack timing/parameters -- same values used throughout Day 2/3 checkpoints.
STEP_SIZE_M = 200.0
STEP_DURATION_S = 7200          # 2h -- reverts before drift_start
DRIFT_RATE_M_PER_S = 1.0
DRIFT_DURATION_S = 7200         # 2h -- reverts before evasive_start
EVASIVE_TARGET_OFFSET_M = 150.0
EVASIVE_RAMP_DURATION_S = 2 * 3600

TEST_WINDOW_S = 45 * 60         # 45 min after each attack's own onset --
                                 # must stay short (well under the 2h evasive
                                 # ramp duration) or the test window starts
                                 # including the already-plateaued, easy-to-
                                 # detect steady-state portion of the attack,
                                 # which dilutes the temporal-vs-fusion gap
                                 # this whole comparison exists to measure
TRAIN_GAP_S = 300               # gap after test window before more train data resumes

def first_crossing_time(df, satellite_id, score_col, threshold, attack_start):
    sat_df = df[(df["satellite_id"] == satellite_id) & (df["t"] >= attack_start)].sort_values("t")
    crossed = sat_df[sat_df[score_col] > threshold]
    if len(crossed) == 0:
        return None
    return crossed["t"].iloc[0] - attack_start

def step1_simulate():
    """Day 1: constellation -> terrain -> observations."""
    print("=" * 70)
    print("STEP 1/5 -- SIMULATOR (constellation, terrain, observations)")
    print("=" * 70)

    duration = config.SIDEREAL_DAY_SECONDS
    dt = config.SIM_TIMESTEP_S

    print(f"Propagating constellation over {duration/3600:.2f} hours "
          f"at {dt}s steps...")
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
    if len(sat_ids) < 5:
        raise RuntimeError(
            f"Only {len(sat_ids)} satellites are ever visible from this "
            f"receiver site -- need at least 5 (3 for attacks, 2 clean). "
            f"Check RECEIVER_SITE / satellite orbital constants in config.py."
        )

    print(f"Visible satellites (most to least visible): {sat_ids}")
    return df, sat_ids, duration


def step2_apply_attacks(df, sat_ids, duration):
    """Day 2/3: apply staggered, bounded attacks."""
    print("\n" + "=" * 70)
    print("STEP 2/5 -- SPOOFING ATTACKS (staggered, bounded)")
    print("=" * 70)

    step_sat, drift_sat, evasive_sat = sat_ids[0], sat_ids[1], sat_ids[2]
    clean_sat_train, clean_sat_test = sat_ids[3], sat_ids[4]

    step_start = duration * 0.4
    drift_start = step_start + 4 * 3600
    evasive_start = drift_start + 4 * 3600

    print(f"step->{step_sat} @ {step_start/3600:.1f}h for {STEP_DURATION_S/3600:.1f}h")
    print(f"drift->{drift_sat} @ {drift_start/3600:.1f}h for {DRIFT_DURATION_S/3600:.1f}h")
    print(f"evasive->{evasive_sat} @ {evasive_start/3600:.1f}h (ramps {EVASIVE_RAMP_DURATION_S/3600:.1f}h, then holds)")
    print(f"clean (train)->{clean_sat_train}, clean (test)->{clean_sat_test}")

    df = apply_step_attack(df, step_sat, step_start, step_size_m=STEP_SIZE_M,
                            duration_s=STEP_DURATION_S)
    df = apply_drift_attack(df, drift_sat, drift_start, drift_rate_m_per_s=DRIFT_RATE_M_PER_S,
                             duration_s=DRIFT_DURATION_S)
    df = apply_evasive_attack(df, evasive_sat, evasive_start,
                               target_offset_m=EVASIVE_TARGET_OFFSET_M,
                               ramp_duration_s=EVASIVE_RAMP_DURATION_S)

    attack_info = {
        "step_sat": step_sat, "drift_sat": drift_sat, "evasive_sat": evasive_sat,
        "clean_sat_train": clean_sat_train, "clean_sat_test": clean_sat_test,
        "step_start": step_start, "drift_start": drift_start, "evasive_start": evasive_start,
        "true_attack_starts": {step_sat: step_start, drift_sat: drift_start,
                                evasive_sat: evasive_start},
    }

    # --- Majority-spoofing diagnostic ---
    print("\n--- Simultaneous-attack sanity check at evasive test window start ---")
    snapshot_t_candidates = df.loc[
        (df["t"] >= evasive_start) & (df["visible"]), "t"
    ].unique()
    snapshot_t = snapshot_t_candidates.min() if len(snapshot_t_candidates) > 0 else evasive_start
    snapshot = df[(df["t"] == snapshot_t) & (df["visible"])]
    spoofed_now = snapshot[snapshot["is_spoofed"]]["satellite_id"].tolist()
    print(f"Satellites visible at t={snapshot_t}: {snapshot['satellite_id'].tolist()}")
    print(f"Of those, currently spoofed: {spoofed_now}")
    if set(spoofed_now) - {evasive_sat}:
        print(f"*** WARNING: satellites other than {evasive_sat} are still spoofed -- "
              f"majority-spoofing risk, downstream results may be unreliable. ***")
    else:
        print("OK -- only the evasive satellite is spoofed at this point.")

    return df, attack_info


def step3_compute_features(df):
    """Day 2/3: temporal + spatial feature engineering."""
    print("\n" + "=" * 70)
    print("STEP 3/5 -- FEATURE ENGINEERING (temporal + spatial)")
    print("=" * 70)

    print("Computing temporal features...")
    df = compute_temporal_features(df)
    print("Computing spatial features...")
    df = compute_spatial_features(df)
    return df


def step4_train_and_evaluate(df, attack_info):
    """Day 2/3: train temporal-only baseline + fusion, evaluate both on
    the same test set."""
    print("\n" + "=" * 70)
    print("STEP 4/5 -- TRAINING + EVALUATION")
    print("=" * 70)

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
    test_mask = (
        (is_attacked_sat & df.apply(in_test_window, axis=1))
        | (df["satellite_id"] == clean_sat_test)
    )
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

    test_clean_fusion, _, _ = prepare_fusion_data(df[test_mask])
    common_test = test_clean_fusion.copy()
    X_temporal_test = common_test[TEMPORAL_FEATURE_COLUMNS].values
    X_fusion_test = common_test[FUSION_INPUT_COLUMNS].values
    y_test = common_test["is_spoofed"].astype(int).values
    attack_type_test = common_test["attack_type"].values

    evasive_sat = attack_info["evasive_sat"]
    ev_test_rows = common_test[common_test["satellite_id"] == evasive_sat]
    print(f"\nEvasive test window diagnostic ({len(ev_test_rows)} rows):")
    print(ev_test_rows[["t", "spatial_deviation", "temporal_prob", "is_spoofed"]]
        .sort_values("t").to_string(index=False))

    y_pred_temporal = temporal_clf.predict(X_temporal_test)
    y_pred_fusion = fusion_clf.predict(X_fusion_test)

    evasive_sat = attack_info["evasive_sat"]
    spatial_latency = first_crossing_time(df, evasive_sat, "spatial_deviation", threshold=10.0,
                                        attack_start=attack_info["true_attack_starts"][evasive_sat])
    temporal_latency = first_crossing_time(df, evasive_sat, "temporal_prob", threshold=0.5,
                                            attack_start=attack_info["true_attack_starts"][evasive_sat])
    print(f"\nEvasive attack detection latency: spatial={spatial_latency}s "
        f"({spatial_latency/60:.1f} min), temporal={temporal_latency}s "
        f"({temporal_latency/60 if temporal_latency else 'N/A'} min)")

    print(f"\nTest set: {len(common_test)} rows")

    print("\n===== TEMPORAL-ONLY BASELINE =====")
    temporal_results = evaluate_by_attack_type(y_test, y_pred_temporal, attack_type_test)
    print(temporal_results.to_string(index=False))

    print("\n===== TEMPORAL + SPATIAL FUSION =====")
    fusion_results = evaluate_by_attack_type(y_test, y_pred_fusion, attack_type_test)
    print(fusion_results.to_string(index=False))

    comparison = temporal_results[["attack_type", "recall", "f1"]].rename(
        columns={"recall": "temporal_only_recall", "f1": "temporal_only_f1"}
    ).merge(
        fusion_results[["attack_type", "recall", "f1"]].rename(
            columns={"recall": "fusion_recall", "f1": "fusion_f1"}
        ),
        on="attack_type",
    )
    comparison["recall_improvement"] = comparison["fusion_recall"] - comparison["temporal_only_recall"]

    print("\n===== HEADLINE COMPARISON =====")
    print(comparison.to_string(index=False))

    return df, comparison


def step5_save_outputs(df, comparison):
    """Save the full dataset and results summary for Day 4/5 use."""
    print("\n" + "=" * 70)
    print("STEP 5/5 -- SAVING OUTPUTS")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset_path = os.path.join(RESULTS_DIR, "full_dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"Saved full dataset: {dataset_path} ({os.path.getsize(dataset_path)/1e6:.1f} MB)")

    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    comparison.to_csv(summary_path, index=False)
    print(f"Saved results summary: {summary_path}")


if __name__ == "__main__":
    df, sat_ids, duration = step1_simulate()
    df, attack_info = step2_apply_attacks(df, sat_ids, duration)
    df = step3_compute_features(df)
    df, comparison = step4_train_and_evaluate(df, attack_info)
    step5_save_outputs(df, comparison)

    print("\n" + "=" * 70)
    print("DONE. Day 1-3 pipeline complete.")
    print("results/csv/full_dataset.csv -- everything, for Day 4 dashboard / Day 5 report")
    print("results/csv/summary.csv       -- headline recall/F1 comparison table")
    print("=" * 70)