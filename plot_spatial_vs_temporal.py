"""
plot_spatial_vs_temporal.py

Generates the headline Day 3 comparison plot: spatial_deviation vs
temporal_prob over the evasive attack's onset window, for the evasive
satellite. Reads directly from results/full_dataset.csv (produced by
run_all.py) -- no need to re-run the whole pipeline.

Usage:
    python plot_spatial_vs_temporal.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import config

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATASET_PATH = os.path.join(RESULTS_DIR, "full_dataset.csv")

# Same attack timing used in run_all.py -- if you changed step_start/
# drift_start/evasive_start there, update these to match.
STEP_START_FRACTION = 0.4
STEP_DURATION_S = 7200
DRIFT_DURATION_S = 7200
EVASIVE_RAMP_DURATION_S = 2 * 3600

# Detection thresholds used for the latency markers (match run_all.py)
SPATIAL_THRESHOLD = 10.0
TEMPORAL_THRESHOLD = 0.5

PLOT_WINDOW_PADDING_S = 30 * 60  # show 30 min before attack start too


def find_evasive_satellite_and_start(df: pd.DataFrame):
    """
    Reconstruct which satellite was the evasive one and when its attack
    started, directly from the labeled data -- avoids hardcoding
    satellite IDs that can shift between runs.
    """
    evasive_rows = df[df["attack_type"] == "evasive"]
    if len(evasive_rows) == 0:
        raise RuntimeError("No rows with attack_type == 'evasive' found in "
                            "the dataset -- did run_all.py finish successfully?")
    evasive_sat = evasive_rows["satellite_id"].iloc[0]
    evasive_start = evasive_rows["t"].min()
    return evasive_sat, evasive_start


def first_crossing_time(df, satellite_id, score_col, threshold, attack_start):
    sat_df = df[(df["satellite_id"] == satellite_id) & (df["t"] >= attack_start)].sort_values("t")
    crossed = sat_df[sat_df[score_col] > threshold]
    if len(crossed) == 0:
        return None
    return crossed["t"].iloc[0] - attack_start


def main():
    print(f"Loading {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    evasive_sat, evasive_start = find_evasive_satellite_and_start(df)
    print(f"Evasive satellite: {evasive_sat}, attack start: t={evasive_start}")

    window_end = evasive_start + EVASIVE_RAMP_DURATION_S + PLOT_WINDOW_PADDING_S
    window_start = evasive_start - PLOT_WINDOW_PADDING_S

    sat_df = df[
        (df["satellite_id"] == evasive_sat)
        & (df["t"] >= window_start)
        & (df["t"] <= window_end)
    ].sort_values("t").copy()

    if sat_df["spatial_deviation"].isna().all() or sat_df["temporal_prob"].isna().all():
        raise RuntimeError(
            "spatial_deviation or temporal_prob is entirely NaN in this window -- "
            "make sure full_dataset.csv was saved AFTER both compute_spatial_features "
            "and add_temporal_scores ran in run_all.py."
        )

    sat_df["minutes_since_attack_start"] = (sat_df["t"] - evasive_start) / 60.0

    spatial_latency = first_crossing_time(df, evasive_sat, "spatial_deviation",
                                           SPATIAL_THRESHOLD, evasive_start)
    temporal_latency = first_crossing_time(df, evasive_sat, "temporal_prob",
                                            TEMPORAL_THRESHOLD, evasive_start)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # --- Top: spatial_deviation ---
    axes[0].plot(sat_df["minutes_since_attack_start"], sat_df["spatial_deviation"],
                 color="tab:blue", linewidth=1.2)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1, label="attack start")
    axes[0].axhline(SPATIAL_THRESHOLD, color="gray", linestyle=":", linewidth=1,
                     label=f"threshold ({SPATIAL_THRESHOLD} m)")
    if spatial_latency is not None:
        axes[0].axvline(spatial_latency / 60, color="tab:blue", linestyle="-",
                         linewidth=1.5, alpha=0.6,
                         label=f"spatial detects @ {spatial_latency/60:.1f} min")
    axes[0].set_title(f"Spatial consensus vs temporal detection -- {evasive_sat} (evasive attack)")
    axes[0].set_ylabel("spatial_deviation (m)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Bottom: temporal_prob ---
    axes[1].plot(sat_df["minutes_since_attack_start"], sat_df["temporal_prob"],
                 color="tab:orange", linewidth=1.2)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(TEMPORAL_THRESHOLD, color="gray", linestyle=":", linewidth=1,
                     label=f"threshold ({TEMPORAL_THRESHOLD})")
    if temporal_latency is not None:
        axes[1].axvline(temporal_latency / 60, color="tab:orange", linestyle="-",
                         linewidth=1.5, alpha=0.6,
                         label=f"temporal detects @ {temporal_latency/60:.1f} min")
    axes[1].set_xlabel("Minutes since attack start")
    axes[1].set_ylabel("temporal_prob (RandomForest)")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    if spatial_latency is not None and temporal_latency is not None:
        gap_min = (temporal_latency - spatial_latency) / 60
        fig.suptitle(f"Spatial detects the evasive attack {gap_min:.1f} minutes "
                      f"earlier than temporal-only", fontsize=12, y=1.00)

    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "spatial_vs_temporal_evasive.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    if spatial_latency is not None and temporal_latency is not None:
        print(f"Spatial latency: {spatial_latency/60:.1f} min, "
              f"Temporal latency: {temporal_latency/60:.1f} min, "
              f"gap: {(temporal_latency-spatial_latency)/60:.1f} min")


if __name__ == "__main__":
    main()