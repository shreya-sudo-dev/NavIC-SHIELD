from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "csv"
    / "real_gnss_selective_attack.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "csv"
FIG_DIR = BASE_DIR / "results" / "figures"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "real_gnss_temporal_selective_attack.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_CONFIGURATION = "3_satellites"


# Temporal attack schedule
#
# 0–15 min     +20 m
# 15–25 min    +50 m
# 25–35 min    +100 m
# 35–45 min    +500 m
# 45–56 min    +20 m
#
# These are all present in the CSV.

ATTACK_PROFILE = [
    (0, 900, 20.0),
    (900, 1500, 50.0),
    (1500, 2100, 100.0),
    (2100, 2700, 500.0),
    (2700, 999999, 20.0),
]


# ============================================================
# LOAD
# ============================================================

print("Loading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print()
print("Available columns:")
print(list(df.columns))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "t",
    "attack_configuration",
    "attack_bias_m",
    "attack_position_error_m",
    "attack_clock_bias_m",
    "attack_mean_abs_residual_m",
    "n_abnormal_20m",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise KeyError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# FILTER CONFIGURATION
# ============================================================

df = df[
    df["attack_configuration"]
    == TARGET_CONFIGURATION
].copy()

if df.empty:
    raise RuntimeError(
        f"No data found for configuration "
        f"'{TARGET_CONFIGURATION}'."
    )


print()
print("=" * 68)
print("TEMPORAL SELECTIVE GNSS ATTACK")
print("=" * 68)

print(
    f"Configuration:  {TARGET_CONFIGURATION}"
)

print(
    f"Rows available: {len(df)}"
)


# ============================================================
# AVAILABLE ATTACK LEVELS
# ============================================================

available_biases = sorted(
    df["attack_bias_m"]
    .dropna()
    .unique()
)

print()
print("Available attack levels:")
print(available_biases)


required_biases = sorted(
    set(
        bias
        for _, _, bias in ATTACK_PROFILE
    )
)

print()
print("Required temporal attack levels:")
print(required_biases)


missing_biases = [
    bias
    for bias in required_biases
    if bias not in available_biases
]

if missing_biases:
    raise RuntimeError(
        "\nTemporal experiment requires attack levels "
        f"{missing_biases}, but they are not present.\n\n"
        f"Available levels: {available_biases}"
    )


# ============================================================
# TIME
# ============================================================

df["t"] = pd.to_numeric(
    df["t"],
    errors="coerce"
)

df = df.dropna(
    subset=["t"]
)

df["time_minutes"] = (
    df["t"] / 60.0
)


# ============================================================
# TEMPORAL ATTACK FUNCTION
# ============================================================

def get_attack_bias(t):

    for start, end, bias in ATTACK_PROFILE:

        if start <= t < end:
            return bias

    return None


df["temporal_attack_bias_m"] = (
    df["t"].apply(get_attack_bias)
)


# Remove epochs outside our temporal experiment

df = df[
    df["temporal_attack_bias_m"].notna()
].copy()


# ============================================================
# SELECT MATCHING ATTACK RESULT
# ============================================================

result_rows = []

for t in sorted(df["t"].unique()):

    desired_bias = get_attack_bias(t)

    epoch_data = df[
        df["t"] == t
    ]

    matching = epoch_data[
        np.isclose(
            epoch_data["attack_bias_m"],
            desired_bias
        )
    ]

    if matching.empty:
        continue

    row = matching.iloc[0].copy()

    result_rows.append(row)


temporal = pd.DataFrame(
    result_rows
)


if temporal.empty:
    raise RuntimeError(
        "No temporal attack results were generated."
    )


# ============================================================
# ATTACK PHASE
# ============================================================

temporal["attack_active"] = True

temporal["attack_phase"] = np.select(
    [
        temporal["temporal_attack_bias_m"] == 20,
        temporal["temporal_attack_bias_m"] == 50,
        temporal["temporal_attack_bias_m"] == 100,
        temporal["temporal_attack_bias_m"] == 500,
    ],
    [
        "20m",
        "50m",
        "100m",
        "500m",
    ],
    default="unknown"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 68)
print("TEMPORAL ATTACK SUMMARY")
print("=" * 68)

print(
    f"Epochs analysed:        "
    f"{len(temporal)}"
)

print(
    f"Minimum attack bias:    "
    f"{temporal['temporal_attack_bias_m'].min():.2f} m"
)

print(
    f"Maximum attack bias:    "
    f"{temporal['temporal_attack_bias_m'].max():.2f} m"
)

print(
    f"Mean position error:    "
    f"{temporal['attack_position_error_m'].mean():.2f} m"
)

print(
    f"Maximum position error: "
    f"{temporal['attack_position_error_m'].max():.2f} m"
)

print(
    f"Mean clock bias:        "
    f"{temporal['attack_clock_bias_m'].mean():.2f} m"
)

print(
    f"Maximum residual:       "
    f"{temporal['attack_mean_abs_residual_m'].max():.2f} m"
)


# ============================================================
# PHASE SUMMARY
# ============================================================

print()
print("Attack phase statistics:")
print()

phase_summary = (
    temporal
    .groupby(
        [
            "attack_phase",
            "temporal_attack_bias_m"
        ]
    )
    .agg(
        epochs=("t", "count"),

        mean_position_error_m=(
            "attack_position_error_m",
            "mean"
        ),

        max_position_error_m=(
            "attack_position_error_m",
            "max"
        ),

        mean_clock_bias_m=(
            "attack_clock_bias_m",
            "mean"
        ),

        mean_residual_m=(
            "attack_mean_abs_residual_m",
            "mean"
        ),

        max_residual_m=(
            "attack_mean_abs_residual_m",
            "max"
        ),

        mean_abnormal_satellites=(
            "n_abnormal_20m",
            "mean"
        ),
    )
    .reset_index()
)

print(
    phase_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE DATA
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

temporal.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:")
print(OUTPUT_FILE)


# ============================================================
# FIGURE 1
# ATTACK PROFILE
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.step(
    temporal["time_minutes"],
    temporal["temporal_attack_bias_m"],
    where="post"
)

plt.xlabel("Time (minutes)")
plt.ylabel(
    "Injected selective attack bias (m)"
)

plt.title(
    "Temporal Selective GNSS Attack Profile"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

fig1 = (
    FIG_DIR
    / "real_gnss_temporal_selective_attack_profile.png"
)

plt.savefig(
    fig1,
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 2
# POSITION ERROR
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    temporal["time_minutes"],
    temporal["attack_position_error_m"]
)

plt.xlabel("Time (minutes)")
plt.ylabel("Position error (m)")

plt.title(
    "Temporal Selective Attack vs Position Error"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

fig2 = (
    FIG_DIR
    / "real_gnss_temporal_selective_position.png"
)

plt.savefig(
    fig2,
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 3
# CLOCK RESPONSE
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    temporal["time_minutes"],
    temporal["temporal_attack_bias_m"],
    label="Injected attack bias"
)

plt.plot(
    temporal["time_minutes"],
    temporal["attack_clock_bias_m"],
    label="Receiver clock bias"
)

plt.xlabel("Time (minutes)")
plt.ylabel("Bias (m)")

plt.title(
    "Temporal Selective Attack vs Receiver Clock"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

fig3 = (
    FIG_DIR
    / "real_gnss_temporal_selective_clock.png"
)

plt.savefig(
    fig3,
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 4
# RESIDUAL RESPONSE
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    temporal["time_minutes"],
    temporal["attack_mean_abs_residual_m"]
)

plt.xlabel("Time (minutes)")
plt.ylabel("Mean |residual| (m)")

plt.title(
    "Temporal Selective Attack vs Residual Response"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

fig4 = (
    FIG_DIR
    / "real_gnss_temporal_selective_residual.png"
)

plt.savefig(
    fig4,
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 5
# ABNORMAL SATELLITES
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    temporal["time_minutes"],
    temporal["n_abnormal_20m"]
)

plt.xlabel("Time (minutes)")

plt.ylabel(
    "Satellites with |residual| > 20 m"
)

plt.title(
    "Temporal Selective Attack vs Abnormal Satellites"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

fig5 = (
    FIG_DIR
    / "real_gnss_temporal_selective_satellites.png"
)

plt.savefig(
    fig5,
    dpi=200
)

plt.close()


# ============================================================
# DONE
# ============================================================

print()
print("Saved figures:")

print(fig1)
print(fig2)
print(fig3)
print(fig4)
print(fig5)

print()
print("=" * 68)
print("TEMPORAL SELECTIVE ATTACK COMPLETE")
print("=" * 68)