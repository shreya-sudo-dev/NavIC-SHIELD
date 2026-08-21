"""
models/fusion.py

Combines Day 2's temporal (RandomForest) score with Day 3's spatial
consensus score into one fused confidence value. This is the headline
comparison: temporal-only (Day 2 baseline) vs. temporal+spatial hybrid
(this file), specifically on the evasive attack, where temporal-only
recall is expected to collapse.

Fusion approach: a small second classifier (LogisticRegression) trained
on [temporal_prob, spatial_deviation] -> is_spoofed. Deliberately
simple -- the temporal RandomForest and the spatial median-deviation
feature are each already doing the hard work; fusion just needs to
combine two already-informative scores.

IMPORTANT: step and drift attacks are bounded with duration_s so they
revert to clean well before the evasive attack's own test window opens.
Without this, an open-ended step/drift attack still active hours later
would put multiple satellites in "spoofed" state simultaneously at the
evasive test window, corrupting the spatial median's assumption that
most of the visible constellation is trustworthy -- which would
invalidate the headline evasive-attack comparison specifically. A
diagnostic check is included in the checkpoint below to confirm this
didn't happen before trusting the results.
"""

import numpy as np
import pandas as pd

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

from features.temporal import FEATURE_COLUMNS as TEMPORAL_FEATURE_COLUMNS
from features.spatial import SPATIAL_FEATURE_COLUMNS

FUSION_INPUT_COLUMNS = ["temporal_prob"] + SPATIAL_FEATURE_COLUMNS


def add_temporal_scores(df: pd.DataFrame, temporal_clf: RandomForestClassifier) -> pd.DataFrame:
    """
    Add a temporal_prob column: the Day 2 RandomForest's predicted
    probability of is_spoofed=1, for every row that has temporal
    features available. Rows missing temporal features (not enough
    rolling-window history, or not visible) get NaN.
    """
    df = df.copy()
    df["temporal_prob"] = np.nan

    has_features = df[TEMPORAL_FEATURE_COLUMNS].notna().all(axis=1)
    X = df.loc[has_features, TEMPORAL_FEATURE_COLUMNS].values
    df.loc[has_features, "temporal_prob"] = temporal_clf.predict_proba(X)[:, 1]

    return df


def prepare_fusion_data(df: pd.DataFrame):
    """
    Drop rows missing either temporal_prob or spatial_deviation --
    fusion needs both inputs. Returns the clean dataframe plus X, y.
    """
    clean = df.dropna(subset=FUSION_INPUT_COLUMNS + ["is_spoofed"]).copy()
    X = clean[FUSION_INPUT_COLUMNS].values
    y = clean["is_spoofed"].astype(int).values
    return clean, X, y


def train_fusion_detector(train_df: pd.DataFrame, seed: int = 42) -> LogisticRegression:
    X_train = train_df[FUSION_INPUT_COLUMNS].values
    y_train = train_df["is_spoofed"].astype(int).values

    clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


def evaluate_by_attack_type(y_true: np.ndarray, y_pred: np.ndarray,
                             attack_type: np.ndarray) -> pd.DataFrame:
    """
    Same shape as baseline_detector.py's evaluate_by_attack_type, but
    takes raw arrays so it can be reused for both the temporal-only and
    fusion models on the exact same test set -- that's what makes the
    before/after comparison valid.
    """
    rows = []
    for attack in ["step", "drift", "evasive"]:
        mask = attack_type == attack
        if mask.sum() == 0:
            continue
        recall = recall_score(y_true[mask], y_pred[mask], zero_division=0)
        f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
        rows.append({"attack_type": attack, "n_rows": int(mask.sum()),
                      "recall": recall, "f1": f1})

    overall_recall = recall_score(y_true, y_pred, zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, zero_division=0)
    rows.append({"attack_type": "OVERALL", "n_rows": len(y_true),
                  "recall": overall_recall, "f1": overall_f1})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sanity check / Day 3 checkpoint (run this file directly)
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
    from features.spatial import compute_spatial_features
    from models.baseline_detector import (
        prepare_training_data, train_baseline_detector,
    )

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
        raise RuntimeError(f"Only {len(sat_ids)} satellites visible -- need at least 5.")

    step_sat, drift_sat, evasive_sat = sat_ids[0], sat_ids[1], sat_ids[2]
    clean_sat_train, clean_sat_test = sat_ids[3], sat_ids[4]

    # Staggered attack timing, BOUNDED (duration_s) so step/drift fully
    # revert to clean before the evasive attack's own test window opens.
    step_start = duration * 0.4
    drift_start = step_start + 4 * 3600
    evasive_start = drift_start + 4 * 3600

    STEP_DURATION_S = 7200    # 2h -- ends well before drift_start
    DRIFT_DURATION_S = 7200   # 2h -- ends well before evasive_start

    print(f"\nAttacks (staggered, bounded): "
          f"step->{step_sat} @ {step_start/3600:.1f}h for {STEP_DURATION_S/3600:.1f}h, "
          f"drift->{drift_sat} @ {drift_start/3600:.1f}h for {DRIFT_DURATION_S/3600:.1f}h, "
          f"evasive->{evasive_sat} @ {evasive_start/3600:.1f}h (unbounded, holds at target)")
    print(f"Clean satellites: train->{clean_sat_train}, test->{clean_sat_test}")

    STEP_DURATION_S = 7200
    DRIFT_DURATION_S = 7200
    df = apply_step_attack(df, step_sat, step_start, step_size_m=200.0,
                            duration_s=STEP_DURATION_S)
    df = apply_drift_attack(df, drift_sat, drift_start, drift_rate_m_per_s=1.0,
                            duration_s=DRIFT_DURATION_S)

    df = apply_evasive_attack(df, evasive_sat, evasive_start,
                               target_offset_m=150.0, ramp_duration_s=2 * 3600)

    true_attack_starts = {step_sat: step_start, drift_sat: drift_start,
                           evasive_sat: evasive_start}

    # --- Diagnostic: confirm no majority-spoofing at the evasive test window ---
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
        print(f"WARNING: satellites other than {evasive_sat} are still spoofed at "
              f"the evasive test window -- majority-spoofing risk, results below "
              f"may be unreliable. Increase the gap between attacks or shorten "
              f"STEP_DURATION_S/DRIFT_DURATION_S further.")
    else:
        print("OK -- only the evasive satellite is spoofed at this point. "
              "Results below are trustworthy with respect to this specific risk.")

    print("\nComputing temporal features...")
    df = compute_temporal_features(df)
    print("Computing spatial features...")
    df = compute_spatial_features(df)

    # Test window: 45 minutes after EACH attack's own onset (per-satellite,
    # since attacks are staggered).
    gap_s = 300
    window_s = 45 * 60

    def in_test_window(row):
        ts = true_attack_starts.get(row["satellite_id"])
        if ts is None:
            return False
        return ts <= row["t"] < ts + window_s

    def in_train_window(row):
        ts = true_attack_starts.get(row["satellite_id"])
        if ts is None:
            return True  # clean satellites: everything is "train-eligible"
        return row["t"] < ts or row["t"] >= ts + window_s + gap_s

    print("Splitting train/test windows...")
    is_attacked_sat = df["satellite_id"].isin([step_sat, drift_sat, evasive_sat])
    test_mask = (
        (is_attacked_sat & df.apply(in_test_window, axis=1))
        | (df["satellite_id"] == clean_sat_test)
    )
    train_mask = (
        (is_attacked_sat & df.apply(in_train_window, axis=1))
        | (df["satellite_id"] == clean_sat_train)
    )

    # --- Train temporal-only baseline ---
    temporal_clean, _, _, _ = prepare_training_data(df[train_mask])
    print(f"\nTraining temporal-only baseline on {len(temporal_clean)} rows...")
    temporal_clf = train_baseline_detector(temporal_clean)

    # --- Add temporal_prob to the FULL dataframe using the trained model ---
    df = add_temporal_scores(df, temporal_clf)

    # --- Train fusion model ---
    fusion_train_clean, _, _ = prepare_fusion_data(df[train_mask])
    print(f"Training fusion model on {len(fusion_train_clean)} rows...")
    fusion_clf = train_fusion_detector(fusion_train_clean)

    # --- Evaluate BOTH models on the exact same test set ---
    test_clean_fusion, _, _ = prepare_fusion_data(df[test_mask])

    common_test = test_clean_fusion.copy()
    X_temporal_test = common_test[TEMPORAL_FEATURE_COLUMNS].values
    X_fusion_test = common_test[FUSION_INPUT_COLUMNS].values
    y_test = common_test["is_spoofed"].astype(int).values
    attack_type_test = common_test["attack_type"].values

    y_pred_temporal = temporal_clf.predict(X_temporal_test)
    y_pred_fusion = fusion_clf.predict(X_fusion_test)

    print(f"\nTest set: {len(common_test)} rows")

    print("\n===== TEMPORAL-ONLY BASELINE =====")
    temporal_results = evaluate_by_attack_type(y_test, y_pred_temporal, attack_type_test)
    print(temporal_results.to_string(index=False))

    print("\n===== TEMPORAL + SPATIAL FUSION =====")
    fusion_results = evaluate_by_attack_type(y_test, y_pred_fusion, attack_type_test)
    print(fusion_results.to_string(index=False))

    print("\n===== HEADLINE COMPARISON: recall by attack type =====")
    comparison = temporal_results[["attack_type", "recall"]].rename(
        columns={"recall": "temporal_only_recall"}
    ).merge(
        fusion_results[["attack_type", "recall"]].rename(
            columns={"recall": "fusion_recall"}
        ),
        on="attack_type",
    )
    comparison["improvement"] = comparison["fusion_recall"] - comparison["temporal_only_recall"]
    print(comparison.to_string(index=False))

    print("\nExpect: evasive shows the largest recall jump from temporal-only "
          "to fusion -- that's the headline result. Check the sanity-check "
          "warning above first if these numbers look suspicious.")