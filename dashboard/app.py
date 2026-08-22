"""
NavIC-SHIELD
Adversarial-Resilient GNSS Positioning

Professional aerospace/defence monitoring dashboard.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import textwrap


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

import config

from simulator.receiver import (
    _DEFAULT_RECEIVER_ECEF,
    _enu_rotation_matrix,
)


# ============================================================================
# PATHS
# ============================================================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "csv",
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NavIC-SHIELD | GNSS Security",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# DESIGN SYSTEM
# ============================================================================

# Backgrounds
BG = "#07111A"
BG_SECONDARY = "#0A1621"
PANEL = "#0E1C28"
PANEL_HOVER = "#122536"

# Borders
BORDER = "#1B3445"
BORDER_ACTIVE = "#2A5268"

# Primary accent
CYAN = "#22D3EE"
CYAN_DARK = "#0E7490"

# Status colors
GREEN = "#34D399"
AMBER = "#F59E0B"
RED = "#EF4444"
PURPLE = "#A78BFA"

# Text
TEXT = "#E6F1F5"
TEXT_SECONDARY = "#A9BBC5"
TEXT_MUTED = "#718894"

ACCENT = CYAN
ACCENT_OK = GREEN
ACCENT_WARNING = AMBER
ACCENT_DANGER = RED

TEXT_MAIN = TEXT
PANEL_BG = PANEL

# Attack colors
ATTACK_COLORS = {
    "step": AMBER,
    "drift": RED,
    "evasive": PURPLE,
}


# ============================================================================
# GLOBAL CSS
# ============================================================================

st.markdown(
    f"""
    <style>

    /* ------------------------------------------------------------
       GLOBAL
    ------------------------------------------------------------ */

    .stApp {{
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(34, 211, 238, 0.045),
                transparent 32%
            ),
            {BG};
        color: {TEXT};
    }}

    html, body, [class*="css"] {{
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }}

    /* Remove excessive Streamlit spacing */

    .block-container {{
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}


    /* ------------------------------------------------------------
       HEADINGS
    ------------------------------------------------------------ */

    h1, h2, h3 {{
        color: {TEXT} !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }}

    h1 {{
        font-size: 2rem !important;
    }}

    h2 {{
        font-size: 1.35rem !important;
    }}

    h3 {{
        font-size: 1.05rem !important;
    }}


    /* ------------------------------------------------------------
       HEADER
    ------------------------------------------------------------ */

    .navic-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;

        padding: 18px 22px;
        margin-bottom: 20px;

        background:
            linear-gradient(
                135deg,
                rgba(14, 28, 40, 0.98),
                rgba(8, 20, 30, 0.98)
            );

        border: 1px solid {BORDER};
        border-radius: 12px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.25);
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}

    .brand-icon {{
        width: 42px;
        height: 42px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 10px;

        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.35);

        color: {CYAN};
        font-size: 22px;
    }}

    .brand-title {{
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: {TEXT};
    }}

    .brand-subtitle {{
        margin-top: 2px;

        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;

        color: {TEXT_MUTED};
    }}

    .header-meta {{
        text-align: right;
        font-family: "Consolas", monospace;
    }}

    .header-meta-label {{
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {TEXT_MUTED};
    }}

    .header-meta-value {{
        margin-top: 3px;
        font-size: 0.82rem;
        color: {CYAN};
    }}

    /* =========================================================================
    OPERATIONAL THREAT ASSESSMENT
    ========================================================================= */

    .threat-panel {{
        background: linear-gradient(
            135deg,
            rgba(14, 28, 40, 0.98),
            rgba(9, 20, 30, 0.98)
        );
        border: 1px solid rgba(34, 211, 238, 0.20);
        border-radius: 14px;
        padding: 22px 24px;
        margin: 10px 0 24px 0;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.20);
    }}

    .threat-panel-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
    }}

    .section-kicker {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.8px;
        color: #22D3EE;
        margin-bottom: 5px;
    }}

    .threat-panel-title {{
        font-size: 21px;
        font-weight: 700;
        letter-spacing: 0.4px;
        color: #E6F1F5;
    }}

    .threat-badge {{
        display: flex;
        align-items: center;
        gap: 8px;

        padding: 7px 14px;
        border-radius: 999px;

        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.2px;

        border: 1px solid rgba(255,255,255,0.10);
    }}

    .threat-status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }}

    .threat-critical {{
        background: rgba(239, 68, 68, 0.14);
        color: #F87171;
        border-color: rgba(239, 68, 68, 0.35);
    }}

    .threat-high {{
        background: rgba(239, 68, 68, 0.10);
        color: #FB7185;
        border-color: rgba(239, 68, 68, 0.25);
    }}

    .threat-elevated {{
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border-color: rgba(245, 158, 11, 0.30);
    }}

    .threat-nominal {{
        background: rgba(52, 211, 153, 0.10);
        color: #34D399;
        border-color: rgba(52, 211, 153, 0.25);
    }}

    .threat-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 14px;
    }}

    .threat-metric {{
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 14px 16px;
    }}

    .evidence-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 16px;
    }}

    .evidence-item {{
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 16px;
        min-height: 92px;
    }}

    .evidence-label {{
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #7f95a8;
        margin-bottom: 8px;
    }}

    .evidence-value {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.1;
    }}

    .evidence-sub {{
        margin-top: 7px;
        font-size: 0.72rem;
        color: #718497;
    }}

    .evidence-divider {{
        height: 1px;
        background: rgba(255,255,255,0.07);
        margin: 18px 0 4px 0;
    }}

    @media (max-width: 900px) {{
        .evidence-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}

    @media (max-width: 600px) {{
        .evidence-grid {{
            grid-template-columns: 1fr;
        }}
    }}

    .metric-label {{
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #718894;
        margin-bottom: 7px;
    }}

    .metric-value {{
        font-family: monospace;
        font-size: 22px;
        font-weight: 700;
        color: #E6F1F5;
    }}

    .assessment-box {{
        background: rgba(0,0,0,0.14);
        border-left: 2px solid #22D3EE;
        border-radius: 0 8px 8px 0;
        padding: 12px 15px;
        margin-top: 10px;
    }}

    .assessment-label {{
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 1.4px;
        color: #718894;
        margin-bottom: 5px;
    }}

    .assessment-text {{
        font-size: 13px;
        line-height: 1.55;
        color: #A9BBC5;
    }}


    /* ------------------------------------------------------------
       STATUS BANNER
    ------------------------------------------------------------ */

    .mission-status {{
        display: flex;
        align-items: center;
        justify-content: space-between;

        padding: 16px 20px;
        margin: 10px 0 20px;

        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}

    .status-left {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .status-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }}

    .status-title {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {TEXT_MUTED};
    }}

    .status-value {{
        margin-top: 2px;
        font-size: 1rem;
        font-weight: 600;
    }}

    .status-detail {{
        text-align: right;
        font-family: "Consolas", monospace;
        font-size: 0.75rem;
        color: {TEXT_SECONDARY};
    }}

    .status-nominal {{
        border-color: rgba(52, 211, 153, 0.35);
    }}

    .status-nominal .status-dot {{
        background: {GREEN};
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.7);
    }}

    .status-nominal .status-value {{
        color: {GREEN};
    }}

    .status-degraded {{
        border-color: rgba(245, 158, 11, 0.4);
    }}

    .status-degraded .status-dot {{
        background: {AMBER};
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.7);
    }}

    .status-degraded .status-value {{
        color: {AMBER};
    }}

    .status-critical {{
        border-color: rgba(239, 68, 68, 0.5);
        background:
            linear-gradient(
                90deg,
                rgba(239, 68, 68, 0.08),
                {PANEL}
            );
    }}

    .status-critical .status-dot {{
        background: {RED};
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.9);
    }}

    .status-critical .status-value {{
        color: {RED};
    }}


    /* ------------------------------------------------------------
       KPI CARDS
    ------------------------------------------------------------ */

    div[data-testid="stMetric"] {{
        background:
            linear-gradient(
                145deg,
                {PANEL},
                {BG_SECONDARY}
            );

        border: 1px solid {BORDER};
        border-radius: 10px;

        padding: 15px 18px;

        min-height: 105px;

        box-shadow:
            0 8px 20px rgba(0, 0, 0, 0.18);
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: {BORDER_ACTIVE};
    }}

    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;

        font-size: 0.68rem !important;
        font-weight: 600 !important;

        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    div[data-testid="stMetricValue"] {{
        color: {TEXT} !important;

        font-family:
            "Consolas",
            "SFMono-Regular",
            monospace;

        font-size: 1.45rem !important;
        font-weight: 600;
    }}


    /* ------------------------------------------------------------
       TABS
    ------------------------------------------------------------ */

    .stTabs [data-baseweb="tab-list"] {{
        gap: 3px;

        background: {PANEL};

        border: 1px solid {BORDER};
        border-radius: 9px;

        padding: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 38px;

        padding: 0 18px;

        border-radius: 6px;

        color: {TEXT_MUTED};
        font-size: 0.78rem;
        font-weight: 600;

        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .stTabs [aria-selected="true"] {{
        background: rgba(34, 211, 238, 0.10) !important;

        color: {CYAN} !important;

        border-bottom: 2px solid {CYAN};
    }}


    /* ------------------------------------------------------------
       BUTTONS
    ------------------------------------------------------------ */

    .stButton button {{
        background: {PANEL};
        color: {TEXT_SECONDARY};

        border: 1px solid {BORDER};
        border-radius: 7px;

        font-size: 0.75rem;
        font-weight: 600;

        transition:
            border-color 0.15s ease,
            background 0.15s ease,
            color 0.15s ease;
    }}

    .stButton button:hover {{
        background: {PANEL_HOVER};

        border-color: {CYAN};
        color: {CYAN};
    }}


    /* ------------------------------------------------------------
       DATA TABLES
    ------------------------------------------------------------ */

    .stDataFrame {{
        border: 1px solid {BORDER};
        border-radius: 9px;
        overflow: hidden;
    }}


    /* ------------------------------------------------------------
       DIVIDERS
    ------------------------------------------------------------ */

    hr {{
        border-color: {BORDER};
        opacity: 0.7;
    }}


    /* ------------------------------------------------------------
       CAPTIONS
    ------------------------------------------------------------ */

    .stCaption {{
        color: {TEXT_MUTED} !important;
    }}


    /* ------------------------------------------------------------
       PLOTLY CONTAINERS
    ------------------------------------------------------------ */

    div[data-testid="stPlotlyChart"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 4px;

        background: {PANEL};
    }}





    /* ------------------------------------------------------------
       FOOTER
    ------------------------------------------------------------ */

    .system-footer {{
        margin-top: 35px;
        padding-top: 15px;

        border-top: 1px solid {BORDER};

        display: flex;
        justify-content: space-between;

        font-family: "Consolas", monospace;
        font-size: 0.65rem;

        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# PLOTLY DESIGN SYSTEM
# ============================================================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,

    font=dict(
        color=TEXT,
        family="Inter, Arial, sans-serif",
        size=12,
    ),

    xaxis=dict(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        linecolor=BORDER,
        title_font=dict(color=TEXT_SECONDARY),
        tickfont=dict(color=TEXT_MUTED),
    ),

    yaxis=dict(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        linecolor=BORDER,
        title_font=dict(color=TEXT_SECONDARY),
        tickfont=dict(color=TEXT_MUTED),
    ),

    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY),
    ),

    margin=dict(
        l=45,
        r=25,
        t=55,
        b=45,
    ),

    hoverlabel=dict(
        bgcolor=BG_SECONDARY,
        bordercolor=BORDER_ACTIVE,
        font=dict(
            color=TEXT,
            size=12,
        ),
    ),
)


# ============================================================================
# SYSTEM CONSTANTS
# ============================================================================

KALMAN_REJECT_THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    missing = []
    for fname in ["day4_position_results.csv", "day4_satellite_level.csv",
                  "day4_attack_info.csv", "day4_summary.csv"]:
        if not os.path.exists(os.path.join(RESULTS_DIR, fname)):
            missing.append(fname)
    if missing:
        return None, None, None, None, None, missing

    epoch_df = pd.read_csv(os.path.join(RESULTS_DIR, "day4_position_results.csv"))
    sat_df = pd.read_csv(os.path.join(RESULTS_DIR, "day4_satellite_level.csv"))
    attack_info = pd.read_csv(os.path.join(RESULTS_DIR, "day4_attack_info.csv"))
    summary_df = pd.read_csv(os.path.join(RESULTS_DIR, "day4_summary.csv"))

    comparison_df = None
    comparison_path = os.path.join(RESULTS_DIR, "summary.csv")
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)

    R = _enu_rotation_matrix(config.RECEIVER_SITE["lat_deg"], config.RECEIVER_SITE["lon_deg"])
    true_pos = _DEFAULT_RECEIVER_ECEF
    for prefix in ["raw", "kalman"]:
        diffs = epoch_df[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].values - true_pos
        enu = diffs @ R.T
        epoch_df[f"{prefix}_east_m"] = enu[:, 0]
        epoch_df[f"{prefix}_north_m"] = enu[:, 1]

    return epoch_df, sat_df, attack_info, summary_df, comparison_df, []


epoch_df, sat_df, attack_info, summary_df, comparison_df, missing_files = load_data()

# ---------------------------------------------------------------------------
# NAVIC-SHIELD COMMAND HEADER
# ---------------------------------------------------------------------------

st.markdown(
    textwrap.dedent(f"""
    <div class="navic-header">
        <div class="brand">
            <div class="brand-icon">🛰</div>
            <div>
                <div class="brand-title">NAVIC-SHIELD</div>
                <div class="brand-subtitle">
                    ADVERSARIAL-RESILIENT<br>
                    NAVIGATION INTELLIGENCE
                </div>
            </div>
        </div>
        <div class="header-meta">
            <div class="header-meta-label">REFERENCE SITE</div>
            <div class="header-meta-value">
                {config.RECEIVER_SITE['name']}
            </div>
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

all_times = sorted(sat_df["t"].unique())
if not attack_info.empty:
    default_t = float(attack_info.iloc[0]["start_s"] / 3600)
else:
    default_t = 0.0

if "selected_t_hours" not in st.session_state:
    st.session_state.selected_t_hours = default_t

if "selected_event" not in st.session_state:
    st.session_state.selected_event = None

t_hours = st.slider(
    "Time (hours into simulation)",
    min_value=0.0,
    max_value=float(all_times[-1] / 3600),
    value=float(st.session_state.selected_t_hours),
    step=config.SIM_TIMESTEP_S / 3600,
)

st.session_state.selected_t_hours = t_hours

t_selected = min(
    all_times,
    key=lambda t: abs(t - t_hours * 3600)
)

# ============================================================
# CURRENT NAVIGATION STATUS — FOLLOW SELECTED TIMELINE
# ============================================================

# Find the navigation epoch closest to the selected slider time
epoch_idx = np.abs(
    epoch_df["t"].to_numpy() - t_selected
).argmin()

current_epoch = epoch_df.iloc[epoch_idx]

latest_confidence = float(
    current_epoch["confidence"]
)

# Get satellite observations at the selected epoch
satellite_times = sat_df["t"].to_numpy()

sat_idx = np.abs(
    satellite_times - t_selected
).argmin()

current_satellite_time = satellite_times[sat_idx]

current_sat_df = sat_df[
    sat_df["t"] == current_satellite_time
]

if not current_sat_df.empty:
    latest_spoof_probability = float(
        current_sat_df["fusion_spoof_prob"]
        .fillna(0)
        .max()
    )
else:
    latest_spoof_probability = 0.0


# Determine navigation status
# ============================================================================
# OPERATIONAL THREAT ASSESSMENT
# ============================================================================

if latest_spoof_probability >= 0.80:
    threat_level = "CRITICAL"
    threat_class = "threat-critical"
    threat_color = RED

    threat_description = (
        "Strong evidence of anomalous GNSS measurements. "
        "Navigation integrity is currently considered degraded."
    )

elif latest_spoof_probability >= 0.50:
    threat_level = "HIGH"
    threat_class = "threat-high"
    threat_color = RED

    threat_description = (
        "Multiple measurements show elevated deviation "
        "from the established GNSS baseline."
    )

elif latest_spoof_probability >= 0.25:
    threat_level = "ELEVATED"
    threat_class = "threat-elevated"
    threat_color = AMBER

    threat_description = (
        "Anomalous measurements are present, but evidence "
        "is insufficient to establish a high-confidence threat."
    )

else:
    threat_level = "NOMINAL"
    threat_class = "threat-nominal"
    threat_color = GREEN

    threat_description = (
        "Current GNSS measurements remain within the "
        "expected operational baseline."
    )


# ============================================================================
# NAVIGATION IMPACT ASSESSMENT
# ============================================================================

# ============================================================================
# NAVIGATION IMPACT + ANALYST ASSESSMENT
# ============================================================================

if latest_confidence < KALMAN_REJECT_THRESHOLD:
    navigation_assessment = (
        "Navigation confidence is below the configured acceptance "
        "threshold. Resilient positioning assistance may be required."
    )
else:
    navigation_assessment = (
        "Navigation confidence remains within the configured "
        "acceptance range."
    )


# ---------------------------------------------------------------------------
# Analyst assessment
# ---------------------------------------------------------------------------

if threat_level == "CRITICAL":

    analyst_assessment = (
        "High-confidence GNSS integrity anomaly detected. "
        "The observed measurement pattern is consistent with a "
        "potential spoofing event. Navigation outputs should be "
        "treated as degraded while the anomaly persists."
    )

    operational_impact = (
        "Primary navigation solution may be unreliable. "
        "Maintain continuous integrity monitoring and prioritize "
        "resilient positioning if available."
    )

elif threat_level == "HIGH":

    analyst_assessment = (
        "Multiple GNSS measurements exhibit significant deviation "
        "from the established baseline. The available evidence "
        "indicates elevated risk of navigation manipulation."
    )

    operational_impact = (
        "Navigation integrity is potentially compromised. "
        "Continued monitoring is recommended before relying on the "
        "affected positioning solution."
    )

elif threat_level == "ELEVATED":

    analyst_assessment = (
        "GNSS measurements show anomalous behavior relative to the "
        "nominal baseline. Current evidence does not yet establish "
        "a high-confidence spoofing event."
    )

    operational_impact = (
        "Navigation remains usable with increased uncertainty. "
        "Continue monitoring for persistence or escalation of the "
        "observed anomaly."
    )

else:

    analyst_assessment = (
        "No significant GNSS integrity anomaly is currently "
        "indicated. Observed measurements remain consistent with "
        "the established operational baseline."
    )

    operational_impact = (
        "Navigation solution remains suitable for continued "
        "operation under the current simulated conditions."
    )

st.markdown(
    textwrap.dedent(
        f"""
        <div class="threat-panel">
            <div class="threat-panel-header">
                <div>
                    <div class="section-kicker">
                        OPERATIONAL INTELLIGENCE
                    </div>
                    <div class="threat-panel-title">
                        CURRENT THREAT ASSESSMENT
                    </div>
                </div>
                <div class="threat-badge {threat_class}">
                    <span
                        class="threat-status-dot"
                        style="background:{threat_color};">
                    </span>
                    {threat_level}
                </div>
            </div>
            <div class="threat-grid">
                <div class="threat-metric">
                    <div class="metric-label">
                        NAVIGATION CONFIDENCE
                    </div>
                    <div class="metric-value">
                        {latest_confidence:.3f}
                    </div>
                </div>
                <div class="threat-metric">
                    <div class="metric-label">
                        SPOOF PROBABILITY
                    </div>
                    <div class="metric-value">
                        {latest_spoof_probability:.3f}
                    </div>
                </div>
                <div class="threat-metric">
                    <div class="metric-label">
                        SELECTED EPOCH
                    </div>
                    <div class="metric-value">
                        {t_hours:.2f} h
                    </div>
                </div>
            </div>
            <div class="assessment-box">
                <div class="assessment-label">
                    THREAT ASSESSMENT
                </div>
                <div class="assessment-text">
                    {threat_description}
                </div>
            </div>
            <div class="assessment-box">
                <div class="assessment-label">
                    ANALYST ASSESSMENT
                </div>
                <div class="assessment-text">
                    {analyst_assessment}
                </div>
            </div>
            <div class="assessment-box">
                <div class="assessment-label">
                    OPERATIONAL IMPACT
                </div>
                <div class="assessment-text">
                    {operational_impact}
                </div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

attack_windows = {
    row["attack_type"]: (
        row["start_s"],
        row["start_s"] + (row["duration_s"] if pd.notna(row["duration_s"])
                           else epoch_df["t"].max() - row["start_s"])
    )
    for _, row in attack_info.iterrows()
}

# ---------------------------------------------------------------------------
# Headline metric cards -- the numbers that sell the project in 5 seconds
# ---------------------------------------------------------------------------
evasive_row = summary_df[summary_df["attack"] == "evasive"]
step_row = summary_df[summary_df["attack"] == "step"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    max_reduction = ((summary_df["raw_mean_error_m"] - summary_df["kalman_mean_error_m"])
                      / summary_df["raw_mean_error_m"]).max() * 100
    st.metric("Best-case error reduction", f"{max_reduction:.1f}%",
              help="Kalman-corrected vs raw position error, best attack window")
with col2:
    if not step_row.empty:
        st.metric("Step attack correction", f"{step_row['kalman_mean_error_m'].values[0]:.1f} m",
                   delta=f"-{step_row['raw_mean_error_m'].values[0] - step_row['kalman_mean_error_m'].values[0]:.0f} m",
                   delta_color="inverse",
                   help="Mean position error after Kalman correction, vs raw")
with col3:
    if not evasive_row.empty:
        st.metric("Evasive attack correction", f"{evasive_row['kalman_mean_error_m'].values[0]:.1f} m",
                   delta=f"-{evasive_row['raw_mean_error_m'].values[0] - evasive_row['kalman_mean_error_m'].values[0]:.0f} m",
                   delta_color="inverse")
with col4:
    n_attacks = len(attack_info)
    st.metric("Attack scenarios modeled", n_attacks,
               help="step / drift / evasive (drift-evasive spoofing)")

st.markdown("---")

# ---------------------------------------------------------------------------
# SHARED SATELLITE SELECTION
# ---------------------------------------------------------------------------

if "global_selected_satellite" not in st.session_state:
    st.session_state.global_selected_satellite = None


def sync_satellite_selection(source_key):
    """Keep satellite selection synchronized across dashboard sections."""
    selected = st.session_state.get(source_key)

    if selected is not None:
        st.session_state.global_selected_satellite = selected

        # Synchronize both satellite selector widgets.
        if source_key != "constellation_satellite_selector":
            st.session_state.constellation_satellite_selector = selected

        if source_key != "threat_intelligence_satellite":
            st.session_state.threat_intelligence_satellite = selected

tab1, tab2, tab3, tab4 = st.tabs([
    "Constellation Intelligence", "Signal Threat Intelligence", "Navigation Assurance", "Detector Performance",
])

# ---------------------------------------------------------------------------
# TAB 1: Live constellation skyplot, scrubbable, interactive
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Constellation Intelligence")
    st.caption(
        "Monitor satellite visibility, geometry, and signal integrity "
        "at the selected operational epoch."
    )

    # ============================================================================
    # CONSTELLATION INTELLIGENCE SUMMARY
    # ============================================================================

    snapshot_time = sat_df["t"].iloc[
            np.abs(sat_df["t"].to_numpy() - t_selected).argmin()
        ]

    snapshot = sat_df[
        (sat_df["t"] == snapshot_time) &
        (sat_df["visible"])
    ]

    visible_count = len(snapshot)

    suspicious_count = int(
        (snapshot["fusion_spoof_prob"].fillna(0) >= 0.5).sum()
    )

    trusted_count = visible_count - suspicious_count

    if visible_count >= 7:
        geometry_status = "FAVORABLE"
        geometry_color = GREEN
    elif visible_count >= 5:
        geometry_status = "ADEQUATE"
        geometry_color = AMBER
    else:
        geometry_status = "CONSTRAINED"
        geometry_color = RED


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "VISIBLE SATELLITES",
            visible_count,
            help="Satellites contributing observations at the selected epoch."
        )

    with col2:
        st.metric(
            "SUSPECTED",
            suspicious_count,
            help="Satellites with fusion spoof probability ≥ 0.50."
        )

    with col3:
        st.metric(
            "NOMINAL",
            trusted_count,
            help="Satellites currently below the spoof-probability threshold."
        )

    with col4:
        st.metric(
            "GEOMETRY STATUS",
            geometry_status,
            help="Operational assessment based on the number of visible satellites."
        )

    st.markdown(
        f"""
        <div class="assessment-box" style="margin-top:12px;">
            <div class="assessment-label">
                CONSTELLATION ASSESSMENT
            </div>
            <div class="assessment-text">
                {visible_count} satellites are visible at the selected epoch.
                {suspicious_count} are currently classified as potentially anomalous.
                Constellation geometry is assessed as
                <strong style="color:{geometry_color};">
                    {geometry_status}
                </strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()

        spoof = snapshot["fusion_spoof_prob"].fillna(0)

        colors = [
            ACCENT_DANGER if p >= 0.5 else ACCENT_OK
            for p in spoof
        ]

        sizes = 14 + spoof * 26

        # Keep satellite ID and elevation together.
        customdata = np.column_stack([
            snapshot["satellite_id"].astype(str).to_numpy(),
            snapshot["elevation_deg"].astype(float).to_numpy(),
        ])

        fig.add_trace(
            go.Scatterpolar(
                r=90 - snapshot["elevation_deg"],
                theta=snapshot["azimuth_deg"],

                mode="markers+text",

                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(
                        color=TEXT_MAIN,
                        width=1
                    ),
                ),

                text=snapshot["satellite_id"].str.replace(
                    "IRNSS-",
                    "",
                    regex=False
                ),

                textposition="top center",

                textfont=dict(
                    color=TEXT_MAIN,
                    size=11
                ),

                customdata=customdata,

                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Az %{theta:.2f}°<br>"
                    "El %{customdata[1]:.2f}°"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            **PLOTLY_LAYOUT,

            # IMPORTANT:
            # allows a normal mouse click to create a Plotly point selection
            clickmode="event+select",

            polar=dict(
                bgcolor=PANEL_BG,

                radialaxis=dict(
                    range=[0, 90],
                    tickvals=[0, 30, 60, 90],
                    ticktext=["90°", "60°", "30°", "0°"],
                    gridcolor=BORDER,
                ),

                angularaxis=dict(
                    direction="clockwise",
                    rotation=90,
                    gridcolor=BORDER,
                ),
            ),

            title=(
                f"t = {t_selected / 3600:.2f}h  |  "
                f"{len(snapshot)} satellites visible"
            ),

            height=500,
            showlegend=False,
        )

        plot_event = st.plotly_chart(
            fig,
            width="stretch",
            key="constellation_skyplot",
            on_select="rerun",
            selection_mode="points",
        )

        # ---------------------------------------------------------------------------
        # SATELLITE SELECTION
        # ---------------------------------------------------------------------------

        satellite_options = (
            snapshot["satellite_id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if satellite_options:

            # Preserve the shared selection if possible.
            if (
                st.session_state.global_selected_satellite
                not in satellite_options
            ):
                st.session_state.global_selected_satellite = satellite_options[0]

            selected_satellite_id = st.selectbox(
                "Inspect satellite",
                satellite_options,
                index=satellite_options.index(
                    st.session_state.global_selected_satellite
                ),
                key="constellation_satellite_selector",
                on_change=sync_satellite_selection,
                args=("constellation_satellite_selector",),
            )

        else:
            selected_satellite_id = None

        selected_rows = snapshot[
            snapshot["satellite_id"].astype(str) == selected_satellite_id
        ]

        selected_satellite = (
            selected_rows.iloc[0]
            if not selected_rows.empty
            else None
        )

        if selected_satellite is not None:

            satellite_id = str(selected_satellite["satellite_id"])

            azimuth = float(selected_satellite["azimuth_deg"])
            elevation = float(selected_satellite["elevation_deg"])

            spoof_probability = float(
                selected_satellite["fusion_spoof_prob"]
                if pd.notna(selected_satellite["fusion_spoof_prob"])
                else 0.0
            )

            if spoof_probability >= 0.80:
                signal_status = "CRITICAL"
                signal_color = RED
                signal_assessment = (
                    "Strong evidence of anomalous signal behavior. "
                    "This satellite is currently contributing significant "
                    "threat evidence to the navigation integrity assessment."
                )

            elif spoof_probability >= 0.50:
                signal_status = "SUSPECTED"
                signal_color = RED
                signal_assessment = (
                    "Satellite measurements show elevated anomaly probability. "
                    "Measurement integrity should be treated with caution."
                )

            elif spoof_probability >= 0.25:
                signal_status = "ELEVATED"
                signal_color = AMBER
                signal_assessment = (
                    "Moderate anomaly evidence is present, but the current "
                    "measurements do not establish a high-confidence threat."
                )

            else:
                signal_status = "NOMINAL"
                signal_color = GREEN
                signal_assessment = (
                    "Current satellite measurements remain within the "
                    "expected operational signal baseline."
                )

            st.markdown(
                f"""
                <div class="threat-panel" style="margin-top:18px;">
                    <div class="threat-panel-header">
                        <div>
                            <div class="section-kicker">
                                SATELLITE INTELLIGENCE
                            </div>
                            <div class="threat-panel-title">
                                {satellite_id}
                            </div>
                        </div>
                        <div class="threat-badge"
                            style="
                                color:{signal_color};
                                border-color:{signal_color};
                                background:rgba(255,255,255,0.03);
                            ">
                            <span
                                class="threat-status-dot"
                                style="background:{signal_color};">
                            </span>
                            {signal_status}
                        </div>
                    </div>
                    <div class="threat-grid">
                        <div class="threat-metric">
                            <div class="metric-label">
                                AZIMUTH
                            </div>
                            <div class="metric-value">
                                {azimuth:.2f}°
                            </div>
                        </div>
                        <div class="threat-metric">
                            <div class="metric-label">
                                ELEVATION
                            </div>
                            <div class="metric-value">
                                {elevation:.2f}°
                            </div>
                        </div>
                        <div class="threat-metric">
                            <div class="metric-label">
                                SPOOF PROBABILITY
                            </div>
                            <div class="metric-value">
                                {spoof_probability:.3f}
                            </div>
                        </div>
                    </div>
                    <div class="assessment-box">
                        <div class="assessment-label">
                            ANALYST ASSESSMENT
                        </div>
                        <div class="assessment-text">
                            {signal_assessment}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="assessment-box" style="margin-top:18px;">
                    <div class="assessment-label">
                        SATELLITE INTELLIGENCE
                    </div>
                    <div class="assessment-text">
                        Select a satellite from the constellation plot
                        to inspect its current operational intelligence record.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("**NOMINAL** — spoof probability < 0.50")
        st.markdown("**SUSPECTED ANOMALY** — spoof probability ≥ 0.50")
        st.caption(
            "Marker size represents estimated spoof probability. "
            "Select an epoch using the timeline above."
        )
        st.markdown("---")
        display_cols = snapshot[["satellite_id", "azimuth_deg", "elevation_deg",
                                  "fusion_spoof_prob"]].copy()
        display_cols.columns = ["Satellite", "Az (°)", "El (°)", "Spoof Prob"]
        st.dataframe(display_cols.round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# TAB 2: SIGNAL THREAT INTELLIGENCE
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Signal Threat Intelligence")
    st.caption(
        "Inspect satellite-level detection intelligence, signal geometry, "
        "attack classification, and temporal anomaly behavior."
    )

    # -----------------------------------------------------------------------
    # SATELLITE SELECTION
    # -----------------------------------------------------------------------

    satellite_options = sorted(
        sat_df["satellite_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not satellite_options:
        st.warning("No visible satellites are available.")
    else:

         # If nothing has been selected yet, use the first satellite.
        if (
            st.session_state.global_selected_satellite
            not in satellite_options
        ):
            st.session_state.global_selected_satellite = satellite_options[0]

        selected_satellite_id = st.selectbox(
            "SELECT SATELLITE",
            satellite_options,
            index=satellite_options.index(
                st.session_state.global_selected_satellite
            ),
            key="threat_intelligence_satellite",
            on_change=sync_satellite_selection,
            args=("threat_intelligence_satellite",),
            help="Select a satellite to inspect its complete signal-threat record.",
        )

        # Current-epoch record for selected satellite
        selected_current = snapshot[
            snapshot["satellite_id"].astype(str) == selected_satellite_id
        ].copy()

        # If the satellite is not visible at the selected epoch,
        # use its nearest available observation.
        if selected_current.empty:

            satellite_all_history = sat_df[
                sat_df["satellite_id"].astype(str) == selected_satellite_id
            ].copy()

            if not satellite_all_history.empty:

                nearest_idx = (
                    satellite_all_history["t"] - t_selected
                ).abs().idxmin()

                selected_current = satellite_all_history.loc[
                    [nearest_idx]
                ].copy()

        # Full temporal history for selected satellite
        selected_history = sat_df[
            (sat_df["satellite_id"].astype(str) == selected_satellite_id) &
            (sat_df["visible"])
        ].sort_values("t").copy()

        # -------------------------------------------------------------------
        # CURRENT SATELLITE RECORD
        # -------------------------------------------------------------------

        if selected_current.empty:
            st.warning(
                f"No current observation is available for {selected_satellite_id}."
            )
        else:

            current = selected_current.iloc[0]

            azimuth = float(current["azimuth_deg"])
            elevation = float(current["elevation_deg"])

            current_spoof = float(
                current["fusion_spoof_prob"]
                if pd.notna(current["fusion_spoof_prob"])
                else 0.0
            )

            # ---------------------------------------------------------------
            # STATUS CLASSIFICATION
            # ---------------------------------------------------------------

            if current_spoof >= 0.80:
                signal_status = "CRITICAL"
                signal_color = RED
                signal_assessment = (
                    "High-confidence anomalous behavior is currently "
                    "associated with this satellite. Its measurements "
                    "should be treated as compromised until the anomaly "
                    "subsides."
                )

            elif current_spoof >= 0.50:
                signal_status = "SUSPECTED"
                signal_color = RED
                signal_assessment = (
                    "The selected satellite currently exhibits elevated "
                    "spoofing probability. Measurement integrity should "
                    "be treated with caution."
                )

            elif current_spoof >= 0.25:
                signal_status = "ELEVATED"
                signal_color = AMBER
                signal_assessment = (
                    "Moderate anomalous evidence is present. Continued "
                    "monitoring is recommended to determine whether the "
                    "behavior persists."
                )

            else:
                signal_status = "NOMINAL"
                signal_color = GREEN
                signal_assessment = (
                    "The selected satellite is currently operating within "
                    "the expected signal-integrity baseline."
                )

            # ---------------------------------------------------------------
            # ATTACK INFORMATION
            # ---------------------------------------------------------------

            matching_attack = attack_info[
                attack_info["satellite_id"].astype(str)
                == selected_satellite_id
            ]

            if not matching_attack.empty:

                attack_row = matching_attack.iloc[0]

                attack_type = str(
                    attack_row["attack_type"]
                ).upper()

                attack_start = float(
                    attack_row["start_s"]
                )

                if pd.notna(attack_row["duration_s"]):

                    attack_duration = float(
                        attack_row["duration_s"]
                    )

                    attack_end = attack_start + attack_duration

                else:

                    attack_duration = None
                    attack_end = float(
                        epoch_df["t"].max()
                    )

            else:

                attack_type = "NONE"
                attack_start = None
                attack_end = None
                attack_duration = None

            # ---------------------------------------------------------------
            # TEMPORAL STATISTICS
            # ---------------------------------------------------------------

            if not selected_history.empty:

                history_prob = (
                    selected_history["fusion_spoof_prob"]
                    .fillna(0)
                    .astype(float)
                )

                max_spoof = float(history_prob.max())
                mean_spoof = float(history_prob.mean())

                suspicious_epochs = int(
                    (history_prob >= 0.50).sum()
                )

                total_epochs = len(history_prob)

                suspicious_fraction = (
                    suspicious_epochs / total_epochs
                    if total_epochs > 0
                    else 0.0
                )

            else:

                max_spoof = current_spoof
                mean_spoof = current_spoof
                suspicious_epochs = 0
                total_epochs = 0
                suspicious_fraction = 0.0

            # ===============================================================
            # HEADER
            # ===============================================================

            st.markdown(
                f"""
                <div class="threat-panel" style="margin-top:18px;">
                    <div class="threat-panel-header">
                        <div>
                            <div class="section-kicker">
                                SATELLITE INTELLIGENCE
                            </div>
                            <div class="threat-panel-title">
                                {selected_satellite_id}
                            </div>
                        </div>
                        <div class="threat-badge"
                            style="
                                color:{signal_color};
                                border-color:{signal_color};
                                background:rgba(255,255,255,0.03);
                            ">
                            <span
                                class="threat-status-dot"
                                style="background:{signal_color};">
                            </span>
                            {signal_status}
                        </div>
                    </div>
                    <div class="assessment-box">
                        <div class="assessment-label">
                            CURRENT SIGNAL ASSESSMENT
                        </div>
                        <div class="assessment-text">
                            {signal_assessment}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ===============================================================
            # CURRENT DETECTION METRICS
            # ===============================================================

            st.markdown("#### Current Detection State")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Fusion Spoof Probability",
                    f"{current_spoof:.3f}",
                )

            with col2:
                st.metric(
                    "Azimuth",
                    f"{azimuth:.2f}°",
                )

            with col3:
                st.metric(
                    "Elevation",
                    f"{elevation:.2f}°",
                )

            with col4:
                st.metric(
                    "Selected Epoch",
                    f"{t_hours:.2f} h",
                )

            # ===============================================================
            # TEMPORAL DETECTION METRICS
            # ===============================================================

            st.markdown("#### Historical Detection Profile")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Peak Spoof Probability",
                    f"{max_spoof:.3f}",
                )

            with col2:
                st.metric(
                    "Mean Spoof Probability",
                    f"{mean_spoof:.3f}",
                )

            with col3:
                st.metric(
                    "Suspicious Epochs",
                    suspicious_epochs,
                )

            with col4:
                st.metric(
                    "Suspicious Fraction",
                    f"{suspicious_fraction * 100:.1f}%",
                )

            # ===============================================================
            # ATTACK PROFILE
            # ===============================================================

            st.markdown("#### Threat Attribution")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Attack Scenario",
                    attack_type,
                )

            with col2:
                if attack_start is not None:
                    st.metric(
                        "Attack Start",
                        f"{attack_start / 3600:.2f} h",
                    )
                else:
                    st.metric(
                        "Attack Start",
                        "—",
                    )

            with col3:
                if attack_end is not None:
                    st.metric(
                        "Attack End",
                        f"{attack_end / 3600:.2f} h",
                    )
                else:
                    st.metric(
                        "Attack End",
                        "—",
                    )

            with col4:
                if attack_duration is not None:
                    st.metric(
                        "Attack Duration",
                        f"{attack_duration / 3600:.2f} h",
                    )
                else:
                    st.metric(
                        "Attack Duration",
                        "—",
                    )

            # ===============================================================
            # TEMPORAL SPOOF PROBABILITY
            # ===============================================================

            st.markdown("---")
            st.markdown("#### Satellite Threat Timeline")

            if not selected_history.empty:

                history = selected_history.copy()

                history["t_hours"] = history["t"] / 3600

                history_prob = (
                    history["fusion_spoof_prob"]
                    .fillna(0)
                    .astype(float)
                )

                satellite_fig = go.Figure()

                satellite_fig.add_trace(
                    go.Scatter(
                        x=history["t_hours"],
                        y=history_prob,
                        mode="lines+markers",
                        name=selected_satellite_id,
                        line=dict(
                            color=signal_color,
                            width=2.5,
                        ),
                        marker=dict(
                            size=5,
                            color=signal_color,
                        ),
                        hovertemplate=(
                            "<b>"
                            + selected_satellite_id
                            + "</b><br>"
                            "Time: %{x:.2f} h<br>"
                            "Fusion spoof probability: %{y:.3f}"
                            "<extra></extra>"
                        ),
                    )
                )

                # Current timeline position
                satellite_fig.add_vline(
                    x=t_hours,
                    line_width=2,
                    line_dash="dash",
                    line_color=CYAN,
                    annotation_text=f"SELECTED  {t_hours:.2f} h",
                    annotation_position="top",
                    annotation_font_color=CYAN,
                )

                # Detection threshold
                satellite_fig.add_hline(
                    y=0.50,
                    line_dash="dot",
                    line_color=TEXT_MUTED,
                    opacity=0.7,
                    annotation_text="spoof threshold",
                    annotation_position="right",
                )

                # Attack window
                if attack_start is not None:

                    satellite_fig.add_vrect(
                        x0=attack_start / 3600,
                        x1=attack_end / 3600,
                        fillcolor=ATTACK_COLORS.get(
                            attack_type.lower(),
                            TEXT_MUTED,
                        ),
                        opacity=0.10,
                        line_width=0,
                    )

                satellite_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    height=380,
                    xaxis_title="Time (hours)",
                    yaxis_title="Fusion spoof probability",
                    yaxis_range=[-0.05, 1.05],
                    title=(
                        f"{selected_satellite_id} — "
                        "Temporal Threat Profile"
                    ),
                    showlegend=False,
                )

                st.plotly_chart(
                    satellite_fig,
                    width="stretch",
                    key="selected_satellite_threat_timeline",
                )

            # ===============================================================
            # SATELLITE RECORD
            # ===============================================================

            # ===============================================================
            # DETECTION EVIDENCE
            # ===============================================================

            st.markdown("#### Detection Evidence")

            # Status description
            if signal_status == "CRITICAL":
                evidence_summary = (
                    "Strong evidence of signal compromise. "
                    "The satellite should be excluded from trusted positioning "
                    "while the anomaly remains active."
                )
            elif signal_status == "SUSPECTED":
                evidence_summary = (
                    "Elevated spoofing evidence detected. "
                    "Satellite measurements require continued integrity monitoring."
                )
            elif signal_status == "ELEVATED":
                evidence_summary = (
                    "Anomalous behavior is present but has not reached "
                    "the critical confidence level."
                )
            else:
                evidence_summary = (
                    "No significant spoofing evidence is currently detected."
                )

            st.markdown(
                f"""
                <div class="threat-panel" style="margin-top:12px;">
                    <div class="section-kicker">
                        DETECTION EVIDENCE
                    </div>
                    <div class="evidence-grid">
                        <div class="evidence-item">
                            <div class="evidence-label">
                                CURRENT PROBABILITY
                            </div>
                            <div class="evidence-value">
                                {current_spoof:.3f}
                            </div>
                            <div class="evidence-sub">
                                Fusion detector
                            </div>
                        </div>
                        <div class="evidence-item">
                            <div class="evidence-label">
                                PEAK PROBABILITY
                            </div>
                            <div class="evidence-value">
                                {max_spoof:.3f}
                            </div>
                            <div class="evidence-sub">
                                Historical maximum
                            </div>
                        </div>
                        <div class="evidence-item">
                            <div class="evidence-label">
                                TEMPORAL PERSISTENCE
                            </div>
                            <div class="evidence-value">
                                {suspicious_fraction * 100:.1f}%
                            </div>
                            <div class="evidence-sub">
                                {suspicious_epochs} suspicious epochs
                            </div>
                        </div>
                        <div class="evidence-item">
                            <div class="evidence-label">
                                THREAT CLASS
                            </div>
                            <div
                                class="evidence-value"
                                style="color:{signal_color};"
                            >
                                {attack_type}
                            </div>
                            <div class="evidence-sub">
                                Attack attribution
                            </div>
                        </div>
                    </div>
                    <div class="evidence-divider"></div>
                    <div class="evidence-grid">
                        <div class="evidence-item">
                            <div class="evidence-label">
                                AZIMUTH
                            </div>
                            <div class="evidence-value">
                                {azimuth:.2f}°
                            </div>
                            <div class="evidence-sub">
                                Signal geometry
                            </div>
                        </div>
                        <div class="evidence-item">
                            <div class="evidence-label">
                                ELEVATION
                            </div>
                            <div class="evidence-value">
                                {elevation:.2f}°
                            </div>
                            <div class="evidence-sub">
                                Signal geometry
                            </div>
                        </div>
                        <div class="evidence-item">
                            <div class="evidence-label">
                                MEAN PROBABILITY
                            </div>
                            <div class="evidence-value">
                                {mean_spoof:.3f}
                            </div>
                            <div class="evidence-sub">
                                Full observation history
                            </div>
                        </div>
                        <div class="evidence-item">
                            <div class="evidence-label">
                                INTEGRITY STATE
                            </div>
                            <div
                                class="evidence-value"
                                style="color:{signal_color};"
                            >
                                {signal_status}
                            </div>
                            <div class="evidence-sub">
                                Current assessment
                            </div>
                        </div>
                    </div>
                    <div class="assessment-box" style="margin-top:18px;">
                        <div class="assessment-label">
                            EVIDENCE SUMMARY
                        </div>
                        <div class="assessment-text">
                            {evidence_summary}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # ===============================================================
            # ANALYST ASSESSMENT
            # ===============================================================

            st.markdown(
                f"""
                <div class="assessment-box" style="margin-top:18px;">
                    <div class="assessment-label">
                        ANALYST ASSESSMENT
                    </div>
                    <div class="assessment-text">
                        <strong>{selected_satellite_id}</strong> is currently
                        classified as
                        <strong style="color:{signal_color};">
                            {signal_status}
                        </strong>
                        with a fusion spoof probability of
                        <strong>{current_spoof:.3f}</strong>.
                        Historical analysis shows a peak spoof probability
                        of <strong>{max_spoof:.3f}</strong> and
                        <strong>{suspicious_epochs}</strong> suspicious
                        epochs.
                        The associated threat scenario is
                        <strong>{attack_type}</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # NAVIGATION INTEGRITY CONFIDENCE
    # -----------------------------------------------------------------------

    st.markdown("---")
    st.subheader("Navigation Integrity Confidence")
    st.caption(
        "Navigation confidence reflects the current integrity assessment "
        "used to determine when resilient positioning assistance is required."
    )

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=epoch_df["t"] / 3600,
            y=epoch_df["confidence"],
            mode="lines+markers",
            name="confidence",
            line=dict(
                color=ACCENT,
                width=1.5,
            ),
            marker=dict(
                size=4,
                opacity=0.7,
            ),
            hovertemplate=(
                "<b>Navigation Epoch</b><br>"
                "Time: %{x:.2f} h<br>"
                "Confidence: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig2.add_vline(
        x=t_hours,
        line_width=2,
        line_dash="dash",
        line_color=CYAN,
        annotation_text=f"SELECTED  {t_hours:.2f} h",
        annotation_position="top",
        annotation_font_color=CYAN,
    )

    fig2.add_hline(
        y=0.15,
        line_dash="dash",
        line_color=ACCENT_DANGER,
        annotation_text="reject threshold",
        annotation_font_color=ACCENT_DANGER,
    )

    for attack_type, (start, end) in attack_windows.items():

        fig2.add_vrect(
            x0=start / 3600,
            x1=end / 3600,
            fillcolor=ATTACK_COLORS.get(
                attack_type,
                TEXT_MUTED,
            ),
            opacity=0.08,
            line_width=0,
        )

    fig2.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis_title="time (hours)",
        yaxis_title="confidence",
    )

    st.plotly_chart(
        fig2,
        width="stretch",
        key="confidence_chart",
    )

# ---------------------------------------------------------------------------
# TAB 3: Navigation view -- interactive scatter + error timeline
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Navigation Assurance")
    st.caption(
        "Assess positioning degradation and the effectiveness of "
        "resilient Kalman-based correction under simulated threats."
    )

    st.markdown(
        "**Interpretation:** Raw position represents the uncorrected "
        "navigation solution. Kalman-corrected position represents "
        "the resilient estimate after degraded measurements are filtered."
    )
    st.caption("Position displacement relative to the known stationary receiver site. "
                "Lower displacement indicates better navigation stability.")

    col1, col2 = st.columns(2)
    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scattergl(x=epoch_df["raw_east_m"], y=epoch_df["raw_north_m"],
                                     mode="markers", marker=dict(size=3, color=ACCENT_DANGER, opacity=0.35),
                                     name="raw (uncorrected)"))
        fig3.add_trace(go.Scattergl(x=epoch_df["kalman_east_m"], y=epoch_df["kalman_north_m"],
                                     mode="markers", marker=dict(size=3, color=ACCENT_OK, opacity=0.5),
                                     name="Kalman-corrected"))
        fig3.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                                   marker=dict(symbol="star", size=18, color=TEXT_MAIN),
                                   name="true site"))
        fig3.update_layout(**PLOTLY_LAYOUT, height=480, xaxis_title="East (m)",
                            yaxis_title="North (m)", title="Full-day position scatter")
        fig3.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig3, width="stretch")

    with col2:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=epoch_df["t"] / 3600, y=epoch_df["raw_error_m"],
                                   mode="lines", line=dict(color=ACCENT_DANGER, width=1),
                                   opacity=0.6, name="raw error"))
        fig4.add_trace(go.Scatter(x=epoch_df["t"] / 3600, y=epoch_df["kalman_error_m"],
                                   mode="lines", line=dict(color=ACCENT_OK, width=1.6),
                                   name="Kalman-corrected error"))
        for attack_type, (start, end) in attack_windows.items():
            fig4.add_vrect(x0=start / 3600, x1=end / 3600,
                            fillcolor=ATTACK_COLORS.get(attack_type, TEXT_MUTED), opacity=0.08,
                            line_width=0)
        fig4.update_layout(**PLOTLY_LAYOUT, height=480, xaxis_title="time (hours)",
                            yaxis_title="position error (m)", title="Position error over the full day")
        st.plotly_chart(fig4, width="stretch")

    st.markdown("---")
    st.subheader("Per-attack summary")
    st.dataframe(summary_df.round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# TAB 4: Temporal-only vs Fusion detector comparison
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Detection Performance")
    st.caption(
        "Compare temporal-only detection with spatio-temporal fusion "
        "across the simulated threat scenarios."
    )
    if comparison_df is None:
        st.warning("results/csv/summary.csv not found -- run `python run_all.py` "
                    "to generate the detector comparison table.")
    else:
        display = comparison_df[comparison_df["attack_type"] != "OVERALL"].copy()

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=display["attack_type"], y=display["temporal_only_recall"],
                               name="Temporal-only", marker_color=TEXT_MUTED))
        fig5.add_trace(go.Bar(x=display["attack_type"], y=display["fusion_recall"],
                               name="Temporal + Spatial Fusion", marker_color=ACCENT))
        fig5.update_layout(**PLOTLY_LAYOUT, height=420, barmode="group",
                            yaxis_title="recall", title="Detection recall by attack type")
        st.plotly_chart(fig5, width="stretch")

        st.info(
            "**Analyst note:** Recall alone understates the evasive-attack result. "
            "The drift-evasive attack is designed to remain below detection "
            "thresholds during an extended ramp. Detection latency is therefore "
            "a more informative operational metric than row-level recall alone. "
            "Spatio-temporal fusion detects the evasive condition meaningfully "
            "faster than the temporal-only baseline."
        )

        st.markdown("---")
        st.dataframe(display.round(3), hide_index=True, width="stretch")
