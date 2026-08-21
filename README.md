<div align="center">

# NavIC-SHIELD

**Adversarial-Resilient NavIC Spoofing Detection & Secure Positioning**

A terrain-aware GNSS simulation and AI-based spoofing detection framework designed to detect deceptive pseudorange manipulation and maintain reliable navigation through adaptive position estimation.

</div>

---

## What is NavIC-SHIELD?

NavIC-SHIELD is an experimental framework for studying GNSS spoofing against NavIC-like satellite navigation systems.

The system simulates a NavIC satellite constellation, models terrain-aware satellite visibility, generates realistic GNSS observations, injects different spoofing attacks, detects anomalous satellite behaviour using temporal and spatial evidence, estimates the receiver's position, and uses a Kalman-based fallback to reduce navigation error when GNSS confidence decreases.

**The Complete Pipeline:**
```text
Satellite Constellation
        ↓
Terrain-Aware Visibility
        ↓
GNSS Observation Generation
        ↓
Spoofing Attack Injection
        ↓
Temporal + Spatial Detection
        ↓
Confidence Fusion
        ↓
GNSS Position Solver
        ↓
Kalman Fallback
        ↓
Protected Navigation
```

## Why This Project?

GNSS receivers normally assume that received satellite measurements are trustworthy. A spoofing attack violates this assumption by deliberately modifying navigation measurements so that the receiver computes an incorrect position.

For defence, border navigation, autonomous systems, and other navigation-dependent applications, detecting spoofing is not enough. The system must also answer:

> *"What should the receiver do after detecting that GNSS cannot be trusted?"*

NavIC-SHIELD addresses both problems:
1. Detect suspicious satellite measurements.
2. Estimate the receiver position from pseudoranges.
3. Reduce dependence on corrupted GNSS measurements.
4. Maintain a bounded navigation estimate during the attack.
5. Smoothly reacquire GNSS after the attack ends.

## Core Features

* **Simulation:** NavIC-like satellite constellation simulation, ECEF satellite propagation, terrain-aware satellite visibility (elevation/masking constraints).
* **Observation:** Synthetic pseudorange, C/N0, and Doppler generation.
* **Attack Models:** Step spoofing, drift spoofing, and evasive spoofing attacks.
* **Detection:** Temporal anomaly detection, spatial satellite-consensus detection, and temporal + spatial fusion.
* **Navigation:** GNSS position estimation using iterative least squares, receiver clock-bias estimation, and confidence-aware Kalman fallback.
* **Evaluation & UI:** Raw vs. protected navigation comparison, interactive Streamlit dashboard, full-day simulation capability, attack-level and satellite-level evaluation.

## System Architecture

```text
                         NAVIC-SHIELD
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
       NAVIC SIMULATOR                    ATTACK ENGINE
             │                                 │
     ┌───────┼────────┐                ┌───────┼────────┐
     │       │        │                │       │        │
Constellation Terrain Observation     Step    Drift   Evasive
     │       │        │                │       │        │
     └───────┴────────┘                └───────┴────────┘
             │                                 │
             └──────────────┬──────────────────┘
                            ▼
                   GNSS OBSERVATIONS
                            │
                            ▼
                ┌──────────────────────┐
                │  SPOOFING DETECTION  │
                ├──────────────────────┤
                │ Temporal Features    │
                │ Spatial Features     │
                │ Fusion               │
                └──────────┬───────────┘
                           │
                           ▼
                  SPOOFING CONFIDENCE
                           │
                           ▼
                ┌──────────────────────┐
                │ GNSS POSITION SOLVER │
                │                      │
                │ Pseudorange → ECEF   │
                │ Position + Clock Bias│
                └──────────┬───────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ KALMAN FALLBACK    │
                 │                    │
                 │ Prediction         │
                 │ Confidence-weighted│
                 │ GNSS update        │
                 └──────────┬─────────┘
                            │
                            ▼
                  RESILIENT POSITION
```

## How It Works

### 1. Satellite Constellation
The simulator propagates the NavIC satellite constellation and produces satellite ECEF positions for every simulation timestep. Each observation contains information such as:
* Satellite ID & Type
* Time
* ECEF X/Y/Z
* Receiver-satellite geometry

### 2. Terrain-Aware Visibility
A satellite is not automatically considered usable. The simulator evaluates azimuth, elevation, terrain masking, and visibility constraints. Only satellites satisfying the visibility requirements contribute measurements to the receiver.

### 3. GNSS Observation Generation
For visible satellites, the simulator generates measurements including true geometric range, pseudorange, C/N0, and Doppler. The fundamental relationship is:
`pseudorange ≈ true range + receiver clock bias + measurement noise`

### 4. Spoofing Attacks
NavIC-SHIELD currently models three attack behaviours:

* **Step Attack:** A sudden pseudorange offset is introduced, representing an abrupt measurement manipulation.
  ```text
  normal ────────────────┐
                         │
                         └──────── spoofed
  ```
* **Drift Attack:** The pseudorange error increases gradually with time, designed to be less obvious than a sudden step.
  ```text
  normal
    │
    │        /
    │      /
    │    /
    │  /
    └────────────────── time
  ```
* **Evasive Attack:** The attack is designed to remain less obvious to individual detection signals, making it a harder case for the detector.

### 5. Temporal Detection
Temporal features examine how a satellite's measurements change over time (e.g., residual magnitude, first difference, temporal consistency). 
**Key idea:** *A satellite that suddenly behaves differently from its own recent history may be compromised.*

### 6. Spatial Detection
Spatial detection compares one satellite against the behaviour of the other visible satellites.
```text
             Satellite A
                  │
Satellite B ─────┼───── Satellite C
                  │
             Receiver
                  │
            Satellite D

        Consensus between satellites
                  ↓
          Outlier identification
```
If one satellite's residual behaviour strongly disagrees with the consensus of the others, it receives stronger spoofing evidence.

### 7. Fusion
Temporal and spatial evidence are combined into a final spoofing confidence.
```text
Temporal Evidence ────┐
                      ▼
                   Fusion ─────→ Spoof Confidence
                      ▲
Spatial Evidence ─────┘
```
This combines *"This satellite changed abnormally over time"* with *"This satellite also disagrees with the other satellites."*

### 8. GNSS Position Solver
Detection operates on individual satellite measurements, but navigation requires an actual receiver position. NavIC-SHIELD implements an iterative least-squares GNSS position solver. The solver estimates `[x, y, z, clock_bias]` using measurements from at least four visible satellites. The solution from the previous timestep is used as the initial estimate for the next timestep.

### 9. Kalman Fallback
When GNSS confidence decreases, blindly accepting the GNSS position can cause the navigation solution to follow the spoofed trajectory. NavIC-SHIELD uses a constant-velocity Kalman filter with the state `[x, y, vx, vy]`.

```text
Prediction ──→ Predicted position ──→ Check GNSS confidence
                                            │
        ┌───────────────────────────────────┴─────────────┐
        ▼                                                 ▼
High confidence                                     Low confidence
Trust GNSS more                                   Trust prediction more
```
Instead of completely switching between GNSS and prediction, the measurement noise covariance is adjusted according to confidence, producing a smoother recovery after the attack ends.

## Navigation Evaluation
The system compares the raw GNSS position with the Kalman-protected position to measure whether the fallback actually reduces navigation error during spoofing.

```text
           True Position
                 │
         ┌───────┴───────┐
         ▼               ▼
     Raw GNSS       Kalman-Protected
     Position          Position
         │               │
         └───────┬───────┘
                 ▼
           Position Error
```

## Attack Scenarios

| Attack | Behaviour | Detection Challenge |
| :--- | :--- | :--- |
| **Step** | Sudden offset | Easy / abrupt anomaly |
| **Drift** | Gradually increasing offset | Slow temporal change |
| **Evasive** | Less obvious manipulation | Harder combined case |

The framework is designed so that attack duration, start time, target satellite, and attack magnitude can be controlled during experiments.

## Project Results

The repository contains experimental results generated by the current implementation. *(Visualizations to be added)*
* Constellation
* Terrain-Aware Visibility
* Observation Model
* Spatial vs Temporal Detection
* Kalman Fallback

Additional validation figures are available in: `results/figures/`

**Experimental Outputs**
The generated CSV results are stored in `results/csv/`. Current outputs include:
* `dataset_day1_clean.csv`, `full_dataset.csv`, `summary.csv`
* `day4_attack_info.csv`, `day4_position_results.csv`, `day4_satellite_level.csv`, `day4_summary.csv`

These contain the generated observations, attack configuration, satellite-level detection information, position estimates, Kalman results, and summary metrics.

## Dashboard
NavIC-SHIELD includes an interactive dashboard for visualizing the simulation and detection pipeline. The dashboard provides views for:
* Satellite constellation & visibility
* Spoofing activity & detection results
* Navigation behaviour & position estimation
* Kalman fallback & experimental results

## Project Structure

```text
NavIC-SHIELD/
│
├── dashboard/
│   └── app.py
├── data/
│   └── README.md
├── features/
│   ├── __init__.py
│   ├── spatial.py
│   └── temporal.py
├── models/
│   ├── __init__.py
│   ├── baseline_detector.py
│   └── fusion.py
├── navigation/
│   ├── __init__.py
│   ├── position_solver.py
│   └── kalman_fallback.py
├── simulator/
│   ├── __init__.py
│   ├── constellation.py
│   ├── observation.py
│   ├── receiver.py
│   ├── terrain.py
│   └── test_terrain.py
├── spoofing/
│   ├── __init__.py
│   ├── step_attack.py
│   ├── drift_attack.py
│   └── evasive_attack.py
├── scripts/
│   ├── generate_dataset.py
│   ├── run_all.py
│   └── run_day4_integration.py
├── results/
│   ├── csv/
│   └── figures/
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started

### Requirements
* Python 3.10+
* Windows / Linux / macOS
* Virtual environment recommended
* Sufficient storage for generated datasets

### Installation
```bash
git clone https://github.com/<your-username>/NavIC-SHIELD.git
cd NavIC-SHIELD

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts ctivate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project
```bash
# 1. Generate the dataset
python scripts/generate_dataset.py

# 2. Run the main pipeline
python scripts/run_all.py

# 3. Run Day 4 navigation integration
python scripts/run_day4_integration.py

# 4. Start the dashboard
streamlit run dashboard/app.py
```

## Current Machine-Learning Pipeline
The current implementation uses a Random Forest-based baseline detector alongside the temporal/spatial feature and fusion pipeline. The ML architecture is intentionally modular so additional models can be evaluated without changing the simulator or navigation layers. 

**Planned model comparisons include:**
`Random Forest  →  XGBoost  →  GRU / 1D-CNN`

These will be evaluated using consistent attack scenarios and metrics.

## Evaluation Metrics

**Detection:**
* Precision, Recall, F1-score
* False alarm rate
* Detection delay
* Attack-specific performance

**Navigation:**
* Position error (Max, Mean)
* Raw GNSS vs Kalman-protected error
* Recovery behaviour after attack termination

## Design Philosophy
* **Simulation First:** A controlled NavIC simulation provides ground-truth data, making it possible to evaluate the detector under controlled attack conditions.
* **Detection + Navigation:** Spoofing detection alone is insufficient. The project explicitly separates: `Detection → Confidence → Navigation Trust → Position Estimation`.
* **Modular Architecture:** Each stage is implemented independently, allowing individual components to be tested and upgraded without rewriting the entire system.

## Limitations
The current framework is primarily a controlled simulation environment. Current limitations include:
* Synthetic NavIC observations (No raw NavIC RF/IQ processing).
* No real IMU integration.
* Simplified receiver dynamics and attack models.
* Current ML baseline is Random Forest.
* Real-world cross-system validation is still future work.

## Future Work
* XGBoost and GRU / 1D-CNN temporal detection.
* Real-world GNSS spoofing datasets & real NavIC receiver-data validation.
* Multi-satellite spoofing & more sophisticated attack scenarios.
* Explainable AI / SHAP analysis.
* Improved spatial consensus & real-time data integration.
* Advanced navigation filtering.

## Built With
* Python (NumPy, Pandas, Scikit-learn, Matplotlib)
* Streamlit
* NavIC/GNSS simulation models
* ECEF coordinate modelling & Kalman filtering

## Disclaimer
NavIC-SHIELD is a research and educational prototype intended for simulation, experimentation, and evaluation of GNSS spoofing detection and resilient positioning concepts. It is not intended to replace safety-critical navigation systems.

