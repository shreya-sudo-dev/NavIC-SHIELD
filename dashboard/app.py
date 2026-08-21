"""
dashboard/app.py

Streamlit dashboard, 3 views per the plan:
  1. Live constellation -- skyplot of visible satellites at a scrubbable
     time, colored by fusion spoof-probability.
  2. Attack monitor -- per-satellite fusion confidence over time, with
     attack windows shaded, showing which satellite is flagged.
  3. Navigation view -- true vs. raw vs. Kalman-corrected receiver
     position, in a local East/North frame (meters of drift from the
     true site) rather than raw ECEF, which is unreadable directly.
     This is the single most persuasive demo visual per the plan.

Reads from results/*.csv, produced by run_day4_integration.py. Run
run_day4_integration.py FIRST if these files don't exist yet.

Run with:
    streamlit run dashboard/app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from simulator.receiver import _enu_rotation_matrix, _DEFAULT_RECEIVER_ECEF

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

st.set_page_config(page_title="NavIC Spoof Detection Dashboard", layout="wide")


@st.cache_data
def load_data():
    """
    Cached so the (fairly large) CSVs only get read once per session,
    not on every widget interaction/rerun.
    """
    missing = []
    for fname in ["day4_position_results.csv", "day4_satellite_level.csv",
                  "day4_attack_info.csv", "day4_summary.csv"]:
        if not os.path.exists(os.path.join(RESULTS_DIR, fname)):
            missing.append(fname)
    if missing:
        return None, None, None, None, missing

    epoch_df = pd.read_csv(os.path.join(RESULTS_DIR, "day4_position_results.csv"))
    sat_df = pd.read_csv(os.path.join(RESULTS_DIR, "day4_satellite_level.csv"))
    attack_info = pd.read_csv(os.path.join(RESULTS_DIR, "day4_attack_info.csv"))
    summary_df = pd.read_csv(os.path.join(RESULTS_DIR, "day4_summary.csv"))

    # Convert ECEF positions -> local East/North offset from the true
    # receiver site (meters). Raw ECEF numbers (millions of meters) are
    # meaningless to look at directly; the offset from the known-true
    # stationary site is what actually tells the story.
    R = _enu_rotation_matrix(config.RECEIVER_SITE["lat_deg"], config.RECEIVER_SITE["lon_deg"])
    true_pos = _DEFAULT_RECEIVER_ECEF

    for prefix in ["raw", "kalman"]:
        diffs = epoch_df[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].values - true_pos
        enu = diffs @ R.T
        epoch_df[f"{prefix}_east_m"] = enu[:, 0]
        epoch_df[f"{prefix}_north_m"] = enu[:, 1]

    return epoch_df, sat_df, attack_info, summary_df, []


epoch_df, sat_df, attack_info, summary_df, missing_files = load_data()

st.title("NavIC Spoof Detection -- Live Dashboard")

if missing_files:
    st.error(
        f"Missing result files: {', '.join(missing_files)}. "
        f"Run `python run_day4_integration.py` from the project root first, "
        f"then reload this dashboard."
    )
    st.stop()

attack_windows = {
    row["attack_type"]: (row["start_s"], row["start_s"] + (row["duration_s"] if pd.notna(row["duration_s"]) else epoch_df["t"].max() - row["start_s"]))
    for _, row in attack_info.iterrows()
}
attack_colors = {"step": "tab:orange", "drift": "tab:red", "evasive": "tab:purple"}

tab1, tab2, tab3 = st.tabs(["Live Constellation", "Attack Monitor", "Navigation"])

# ---------------------------------------------------------------------------
# TAB 1: Live constellation skyplot, scrubbable through the day
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Satellite visibility, scrub through the day")

    all_times = sorted(sat_df["t"].unique())
    t_hours = st.slider(
        "Time (hours into simulation)",
        min_value=0.0, max_value=float(all_times[-1] / 3600),
        value=float(attack_info.iloc[0]["start_s"] / 3600),
        step=config.SIM_TIMESTEP_S / 3600,
    )
    t_selected = min(all_times, key=lambda t: abs(t - t_hours * 3600))

    snapshot = sat_df[(sat_df["t"] == t_selected) & (sat_df["visible"])]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(90, 0)

        for _, row in snapshot.iterrows():
            spoof_prob = row["fusion_spoof_prob"]
            color = "green" if pd.isna(spoof_prob) or spoof_prob < 0.5 else "red"
            size = 100 + (spoof_prob * 300 if pd.notna(spoof_prob) else 0)
            ax.scatter(np.radians(row["azimuth_deg"]), 90 - row["elevation_deg"],
                       s=size, color=color, alpha=0.8, edgecolors="black", linewidths=0.5)
            ax.annotate(row["satellite_id"].replace("IRNSS-", ""),
                       (np.radians(row["azimuth_deg"]), 90 - row["elevation_deg"]),
                       fontsize=8, ha="center", va="bottom")

        ax.set_title(f"t = {t_selected/3600:.2f}h  ({len(snapshot)} satellites visible)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("**Legend**")
        st.markdown("🟢 Green = trusted (spoof prob < 0.5)")
        st.markdown("🔴 Red = suspected spoofed (spoof prob >= 0.5)")
        st.markdown("Marker size scales with spoof probability")
        st.markdown("---")
        st.markdown("**Visible satellites at this time**")
        display_cols = snapshot[["satellite_id", "azimuth_deg", "elevation_deg", "fusion_spoof_prob"]].copy()
        display_cols.columns = ["Satellite", "Azimuth (°)", "Elevation (°)", "Spoof Prob"]
        st.dataframe(display_cols.round(2), hide_index=True, width='stretch')

# ---------------------------------------------------------------------------
# TAB 2: Attack monitor -- per-satellite confidence over time
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Per-satellite spoof probability over the full day")

    attacked_sats = attack_info["satellite_id"].tolist()
    fig, axes = plt.subplots(len(attacked_sats), 1, figsize=(12, 3 * len(attacked_sats)), sharex=True)
    if len(attacked_sats) == 1:
        axes = [axes]

    for ax, sat_id in zip(axes, attacked_sats):
        sat_row = attack_info[attack_info["satellite_id"] == sat_id].iloc[0]
        attack_type = sat_row["attack_type"]
        sub = sat_df[(sat_df["satellite_id"] == sat_id) & (sat_df["visible"])].sort_values("t")

        ax.plot(sub["t"] / 3600, sub["fusion_spoof_prob"], color=attack_colors.get(attack_type, "gray"))
        start, end = attack_windows[attack_type]
        ax.axvspan(start / 3600, end / 3600, alpha=0.15, color=attack_colors.get(attack_type, "gray"))
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("spoof prob")
        ax.set_title(f"{sat_id} ({attack_type} attack)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (hours)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Epoch-level confidence (used by the Kalman fallback)")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(epoch_df["t"] / 3600, epoch_df["confidence"], linewidth=0.8)
    ax2.axhline(0.15, color="red", linestyle="--", linewidth=0.8, label="reject threshold (0.15)")
    for attack_type, (start, end) in attack_windows.items():
        ax2.axvspan(start / 3600, end / 3600, alpha=0.1, color=attack_colors.get(attack_type, "gray"))
    ax2.set_xlabel("time (hours)")
    ax2.set_ylabel("confidence")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

# ---------------------------------------------------------------------------
# TAB 3: Navigation view -- true vs raw vs Kalman-corrected path
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Receiver position: true site vs. raw vs. Kalman-corrected")
    st.caption("Positions shown as meters of drift (East/North) from the true, "
               "known-stationary receiver site -- not raw ECEF coordinates.")

    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(7, 7))
        ax3.scatter(epoch_df["raw_east_m"], epoch_df["raw_north_m"],
                   s=3, alpha=0.3, color="tab:blue", label="raw (uncorrected)")
        ax3.scatter(epoch_df["kalman_east_m"], epoch_df["kalman_north_m"],
                   s=3, alpha=0.5, color="tab:green", label="Kalman-corrected")
        ax3.scatter([0], [0], marker="*", s=300, color="black", label="true site", zorder=5)
        ax3.set_xlabel("East (m)")
        ax3.set_ylabel("North (m)")
        ax3.set_title("Full-day position scatter")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis("equal")
        st.pyplot(fig3)
        plt.close(fig3)

    with col2:
        fig4, ax4 = plt.subplots(figsize=(7, 7))
        ax4.plot(epoch_df["t"] / 3600, epoch_df["raw_error_m"], label="raw error", alpha=0.6, linewidth=0.8)
        ax4.plot(epoch_df["t"] / 3600, epoch_df["kalman_error_m"], label="Kalman-corrected error", linewidth=1.2)
        for attack_type, (start, end) in attack_windows.items():
            ax4.axvspan(start / 3600, end / 3600, alpha=0.1, color=attack_colors.get(attack_type, "gray"),
                       label=f"{attack_type} window")
        ax4.set_xlabel("time (hours)")
        ax4.set_ylabel("position error (m)")
        ax4.set_title("Position error over the full day")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown("---")
    st.subheader("Per-attack summary")
    st.dataframe(summary_df.round(2), hide_index=True, width='stretch')