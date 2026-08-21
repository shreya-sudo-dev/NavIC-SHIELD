"""
models/baseline_detector.py

RandomForest classifier trained on temporal features only (Day 2's
detector). This is the "before" half of your headline result -- Day 3's
spatial fusion is measured against how much it improves over this.

Split by satellite_id, not by random row, so the test set contains
satellites the model has genuinely never seen any timestep from --
random row splitting would leak adjacent-in-time rows of the same
attack between train and test and overstate performance.
"""

import numpy as np
import pandas as pd

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score

from features.temporal import FEATURE_COLUMNS


def prepare_training_data(feat_df: pd.DataFrame):
    clean = feat_df.dropna(subset=FEATURE_COLUMNS).copy()
    X = clean[FEATURE_COLUMNS].values
    y = clean["is_spoofed"].astype(int).values
    attack_type = clean["attack_type"].values
    return clean, X, y, attack_type


def train_test_split_by_satellite(clean_df: pd.DataFrame, test_satellites: list):
    test_mask = clean_df["satellite_id"].isin(test_satellites)
    train_df = clean_df[~test_mask]
    test_df = clean_df[test_mask]
    return train_df, test_df


def train_baseline_detector(train_df: pd.DataFrame, seed: int = 42) -> RandomForestClassifier:
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["is_spoofed"].astype(int).values

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_by_attack_type(clf: RandomForestClassifier, test_df: pd.DataFrame) -> pd.DataFrame:
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["is_spoofed"].astype(int).values
    y_pred = clf.predict(X_test)

    test_df = test_df.copy()
    test_df["y_true"] = y_test
    test_df["y_pred"] = y_pred

    rows = []
    for attack in ["step", "drift", "evasive"]:
        subset = test_df[test_df["attack_type"] == attack]
        if len(subset) == 0:
            continue
        recall = recall_score(subset["y_true"], subset["y_pred"], zero_division=0)
        f1 = f1_score(subset["y_true"], subset["y_pred"], zero_division=0)
        rows.append({"attack_type": attack, "n_rows": len(subset),
                      "recall": recall, "f1": f1})

    overall_f1 = f1_score(y_test, y_pred, zero_division=0)
    overall_recall = recall_score(y_test, y_pred, zero_division=0)
    rows.append({"attack_type": "OVERALL", "n_rows": len(test_df),
                  "recall": overall_recall, "f1": overall_f1})

    return pd.DataFrame(rows)


def evaluate_detection_latency(clf: RandomForestClassifier, test_df: pd.DataFrame,
                                sim_timestep_s: float,
                                true_attack_starts: dict) -> pd.DataFrame:
    """
    true_attack_starts : dict {satellite_id: true_injection_time_seconds},
    taken directly from the start_time values passed to apply_*_attack.
    Using the test dataframe's earliest row instead (as an earlier version
    of this function did) measures latency-since-test-window-opened, not
    latency-since-attack-began -- a different, misleading quantity when
    the test window doesn't start right at attack onset.
    """
    X_test = test_df[FEATURE_COLUMNS].values
    test_df = test_df.copy()
    test_df["y_pred"] = clf.predict(X_test)

    rows = []
    for (sat_id, attack), group in test_df[test_df["attack_type"] != "none"].groupby(
        ["satellite_id", "attack_type"]
    ):
        group = group.sort_values("t")

        true_attack_start = true_attack_starts.get(sat_id)
        if true_attack_start is None:
            print(f"[evaluate_detection_latency] WARNING: no true attack_start "
                  f"provided for {sat_id}, falling back to test-set minimum.")
            true_attack_start = group["t"].min()

        first_detect = group[group["y_pred"] == 1]["t"].min()

        detected = not pd.isna(first_detect)
        latency_s = (first_detect - true_attack_start) if detected else None

        rows.append({
            "satellite_id": sat_id,
            "attack_type": attack,
            "test_window_start_t": group["t"].min(),
            "true_attack_start_t": true_attack_start,
            "detected_in_test_window": detected,
            "latency_since_true_onset_s": latency_s,
            "latency_since_true_onset_min": (latency_s / 60) if latency_s is not None else None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sanity check / Day 2 checkpoint (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import config
    from simulator.constellation import propagate_constellation
    from simulator.terrain import satellite_visibility
    from simulator.observation import generate_observations
    from spoofing.step_attack import apply_step_attack
    from spoofing.drift_attack import apply_drift_attack
    from spoofing.evasive_attack import apply_evasive_attack
    from features.temporal import compute_temporal_features

    print("Propagating full-day constellation...")
    duration = config.SIDEREAL_DAY_SECONDS
    dt = config.SIM_TIMESTEP_S
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
            f"receiver site -- need at least 5. Check RECEIVER_SITE / "
            f"satellite RAAN values in config.py."
        )

    step_sat, drift_sat, evasive_sat = sat_ids[0], sat_ids[1], sat_ids[2]
    clean_sat_train, clean_sat_test = sat_ids[3], sat_ids[4]

    print(f"\nAttack assignment: step->{step_sat}, drift->{drift_sat}, "
          f"evasive->{evasive_sat}")
    print(f"Clean satellites: train->{clean_sat_train}, test->{clean_sat_test}")

    attack_start = duration * 0.4
    true_attack_starts = {step_sat: attack_start, drift_sat: attack_start,
                           evasive_sat: attack_start}

    df = apply_step_attack(df, step_sat, attack_start, step_size_m=200.0)
    df = apply_drift_attack(df, drift_sat, attack_start, drift_rate_m_per_s=1.0)
    df = apply_evasive_attack(df, evasive_sat, attack_start,
                               target_offset_m=150.0, ramp_duration_s=2 * 3600)

    # time_split now placed 45 MINUTES after attack_start (not 60% of the
    # remaining day, ~8.6 hours) so the test window actually captures the
    # transient/onset period -- the scientifically interesting part -- not
    # just steady-state, which both step and evasive settle into within
    # ~2 hours and become indistinguishable from each other after.
    time_split = attack_start + 45 * 60
    gap_s = 300  # 5 min gap so no rolling window spans the boundary

    print(f"\ntime_split: {time_split/60:.1f} min ({(time_split - attack_start)/60:.1f} "
          f"min after attack onset)")

    print("Computing temporal features...")
    feat_df = compute_temporal_features(df)

    clean_df, X, y, attack_type = prepare_training_data(feat_df)
    print(f"\nTotal usable rows: {len(clean_df)}, spoofed: {y.sum()}, normal: {(y == 0).sum()}")

    train_df = clean_df[
        (clean_df["satellite_id"].isin([step_sat, drift_sat, evasive_sat])
         & ((clean_df["t"] < attack_start) | (clean_df["t"] >= time_split + gap_s)))
        | (clean_df["satellite_id"] == clean_sat_train)
    ]
    test_df = clean_df[
        (clean_df["satellite_id"].isin([step_sat, drift_sat, evasive_sat])
         & (clean_df["t"] >= attack_start) & (clean_df["t"] < time_split))
        | (clean_df["satellite_id"] == clean_sat_test)
    ]
    print(f"Train rows: {len(train_df)}, test rows: {len(test_df)}")

    print("\nTraining RandomForest baseline...")
    clf = train_baseline_detector(train_df)

    print("\nFeature importances:")
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(importances)

    print("\n===== PER-ATTACK-TYPE PERFORMANCE (temporal-only baseline) =====")
    results = evaluate_by_attack_type(clf, test_df)
    print(results.to_string(index=False))

    print("\n===== DETECTION LATENCY (relative to TRUE attack onset) =====")
    latency = evaluate_detection_latency(clf, test_df, config.SIM_TIMESTEP_S, true_attack_starts)
    print(latency.to_string(index=False))