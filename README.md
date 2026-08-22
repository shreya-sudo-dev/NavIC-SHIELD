# NavIC-SHIELD

### Adversarial-Resilient GNSS Positioning for Denied and Degraded Environments

**NavIC-SHIELD** is a spatio-temporal AI framework for detecting GNSS spoofing, assessing navigation integrity, and maintaining resilient receiver positioning under adversarial signal conditions.

The project combines NavIC constellation simulation, terrain-aware satellite visibility, synthetic GNSS observations, controlled spoofing attacks, temporal and spatial anomaly detection, spatio-temporal fusion, navigation integrity assessment, GNSS position solving, and confidence-aware Kalman fallback positioning.

> **Core idea:** spoofing detection should not end at identifying a suspicious signal. The navigation system should also determine whether the GNSS solution can still be trusted and, when confidence falls, transition toward a resilient position estimate.


---

## Overview

GNSS spoofing can manipulate navigation measurements without necessarily causing an obvious signal outage. A receiver may therefore continue producing apparently valid position estimates while its navigation solution is gradually or abruptly corrupted.

NavIC-SHIELD studies this problem through a controlled end-to-end simulation and evaluation pipeline.

\`\`\`text
NavIC Constellation
        |
        v
Terrain-Aware Visibility
        |
        v
GNSS Observation Generation
        |
        v
Controlled Spoofing Injection
        |
        +-----------------------+
        |                       |
        v                       v
Temporal Features        Spatial Features
        |                       |
        v                       v
Temporal Detector        Spatial Detector
        |                       |
        +-----------+-----------+
                    |
                    v
          Spatio-Temporal Fusion
                    |
                    v
       Navigation Integrity Confidence
                    |
             +------+------+
             |             |
             v             v
      High Confidence   Low Confidence
             |             |
             v             v
       GNSS Position   Kalman Fallback
          Solution       Positioning
             |             |
             +------+------+
                    |
                    v
          Resilient Position
\`\`\`

The objective is therefore broader than binary spoofing classification:

- Detect anomalous satellite behaviour.
- Quantify satellite-level spoofing probability.
- Aggregate satellite evidence into epoch-level navigation confidence.
- Estimate the receiver position from GNSS observations.
- Reduce dependence on potentially corrupted measurements.
- Use Kalman-based fallback positioning when GNSS confidence becomes unreliable.
- Evaluate raw versus protected navigation across controlled attack windows.

## Key Capabilities

- NavIC satellite constellation simulation
- Satellite ECEF propagation
- Terrain-aware satellite visibility
- Synthetic GNSS observation generation
- Pseudorange, C/N0, and Doppler modelling
- Controlled spoofing attack injection
  - Step spoofing
  - Drift spoofing
  - Evasive spoofing
- Temporal anomaly detection
- Spatial satellite-consensus detection
- Temporal + spatial fusion
- Satellite-level spoofing probability
- Epoch-level navigation integrity confidence
- Iterative GNSS position solving
- Receiver clock-bias estimation
- Confidence-aware Kalman fallback positioning
- Attack-window evaluation
- Interactive Streamlit dashboard
- Automated end-to-end experiment pipeline

## Attack Scenarios

NavIC-SHIELD evaluates three controlled spoofing behaviours.

### 1. Step Attack

A sudden pseudorange offset is introduced into the targeted satellite's measurements.

\`\`\`text
Normal --------------------+
                            |
                            v
                       Sudden Attack
                            |
                            v
Normal --------------------+
\`\`\`

The abrupt discontinuity makes this attack comparatively easier to detect.

### 2. Drift Attack

The spoofing error increases gradually during the attack window.

\`\`\`text
Error
  |
  |             /
  |           /
  |         /
  |       /
  |_____/
  |
  +-----------------------> Time
\`\`\`

The drift scenario tests whether the system can detect gradually developing navigation corruption before it produces a large positioning error.

### 3. Evasive Attack

The evasive attack is designed to remain difficult to distinguish from normal temporal behaviour.

Its importance is that it exposes the limitations of purely temporal detection and provides a challenging test case for spatial evidence and navigation resilience.

## System Architecture

The project is organized into several functional layers.

\`\`\`text
                    NAVIC-SHIELD
                         |
        +----------------+----------------+
        |                                 |
        v                                 v
   NAVIC SIMULATOR                  ATTACK ENGINE
        |                                 |
   +----+----+                       +----+----+
   |    |    |                       |    |    |
   v    v    v                       v    v    v
Const. Terrain Observation         Step Drift Evasive
        \       /                       |
         \     /                        |
          v   v                         |
       GNSS Observations <--------------+
                 |
                 v
        FEATURE ENGINEERING
          /           \
         v             v
    Temporal         Spatial
     Features        Features
         \             /
          v           v
       SPOOFING DETECTION
              |
              v
        FUSION MODEL
              |
              v
   NAVIGATION CONFIDENCE
              |
       +------+------+
       |             |
       v             v
   Position       Kalman
    Solver        Fallback
       |             |
       +------+------+
              |
              v
      RESILIENT POSITION
\`\`\`

The major software components are:

\`\`\`text
simulator/
    constellation.py
    terrain.py
    observation.py
    receiver.py

spoofing/
    step_attack.py
    drift_attack.py
    evasive_attack.py

features/
    temporal.py
    spatial.py

models/
    baseline_detector.py
    fusion.py

navigation/
    position_solver.py
    kalman_fallback.py

dashboard/
    app.py

scripts/
    generate_dataset.py
    run_all.py
    run_day4_integration.py
\`\`\`

## How It Works

### 1. Satellite Constellation Simulation

The simulator propagates a NavIC-like satellite constellation and generates satellite positions over the simulation period.

Each observation contains information such as:

- Satellite identifier
- Satellite type
- Simulation time
- Satellite ECEF position
- Receiver-satellite geometry
- Visibility state

### 2. Terrain-Aware Visibility

A satellite is not automatically considered usable simply because it exists in the constellation.

The simulator evaluates visibility using:

- Azimuth
- Elevation
- Terrain masking
- Minimum elevation constraints

Only satellites satisfying the visibility conditions contribute usable measurements.

This allows the experiment to separate natural geometric degradation from deliberate signal manipulation.

### 3. GNSS Observation Generation

For visible satellites, the simulator generates synthetic navigation observations.

The pseudorange model follows the basic relationship:

\`\`\`text
pseudorange = geometric range
            + receiver clock bias
            + measurement noise
            + attack contribution
\`\`\`

Additional signal information such as C/N0 and Doppler is also represented in the observation pipeline.

### 4. Spoofing Injection

Controlled attacks are injected into selected satellite observations.

The final integration experiment uses staggered attack windows so that each attack can be evaluated independently.

The experiment configuration is:

\`\`\`text
Step attack
    Target: IRNSS-1C
    Start:  9.6 h
    Duration: 2.0 h

Drift attack
    Target: IRNSS-1D
    Start:  13.6 h
    Duration: 2.0 h

Evasive attack
    Target: IRNSS-1E
    Start:  17.6 h
    Ramp:   2.0 h
    Hold:   subsequent period
\`\`\`

### 5. Temporal Detection

Temporal features describe how an individual satellite's measurements evolve over time.

They capture behaviour such as:

- Measurement persistence
- Temporal deviations
- Changes across consecutive epochs
- Gradual changes in measurement consistency

The underlying idea is:

A satellite that begins behaving differently from its own recent history may be compromised.

### 6. Spatial Detection

Spatial features compare a satellite against the other simultaneously visible satellites.

\`\`\`text
             Satellite A
                   |
                   |
Satellite B -------+------- Satellite C
                   |
                   |
                Receiver
                   |
             Satellite D
\`\`\`

If one satellite's residual behaviour strongly disagrees with the consensus of the other visible satellites, the satellite receives stronger anomaly evidence.

Spatial evidence is particularly useful when an attack does not create a strong temporal discontinuity.

### 7. Temporal + Spatial Fusion

The system combines temporal and spatial evidence into a fused spoofing probability.

\`\`\`text
Temporal Evidence --------\
                           \
                            --> Fusion --> Spoof Probability
                           /
Spatial Evidence --------/
\`\`\`

This creates a satellite-level estimate of how suspicious each observation is.

The fusion stage is intended to exploit complementary information:

\`\`\`text
Temporal:
"Did this satellite change abnormally?"

Spatial:
"Does this satellite disagree with the others?"

Fusion:
"How likely is this satellite to be compromised?"
\`\`\`

### 8. Navigation Integrity Confidence

Satellite-level spoofing probabilities are aggregated into an epoch-level navigation confidence measure.

This confidence represents the current assessment of whether the GNSS navigation solution should be trusted.

\`\`\`text
High confidence
      |
      v
Trust GNSS solution more

Low confidence
      |
      v
Trust prediction / fallback more
\`\`\`

### 9. GNSS Position Solver

The receiver position is estimated from pseudorange observations.

The solver estimates:

\`\`\`text
[x, y, z, clock_bias]
\`\`\`

using the available visible satellites.

The previous epoch solution can provide a useful initial estimate for the next epoch.

The resulting position estimates are evaluated using raw navigation error and protected/fallback navigation error.

### 10. Kalman Fallback

When navigation confidence decreases, blindly accepting the GNSS position can allow spoofed measurements to corrupt the navigation trajectory.

NavIC-SHIELD therefore uses a confidence-aware Kalman fallback.

Conceptually:

\`\`\`text
             Prediction
                 |
                 v
        Predicted Position
                 |
                 v
       Check GNSS Confidence
            /          \
           /            \
          v              v
 High Confidence     Low Confidence
          |              |
          v              v
   Trust GNSS more   Trust prediction
                         more
           \              /
            \            /
             v          v
          Protected Position
\`\`\`

The objective is not to eliminate GNSS usage, but to reduce dependence on unreliable GNSS measurements when the integrity assessment indicates elevated risk.

## Experimental Pipeline

The complete experiment can be executed through:

\`\`\`bash
python scripts/run_all.py
\`\`\`

The pipeline performs five stages:

\`\`\`text
STEP 1
Simulator
    |
    v
Constellation
Terrain Visibility
GNSS Observations

STEP 2
Spoofing Attacks
    |
    v
Step + Drift + Evasive

STEP 3
Feature Engineering
    |
    v
Temporal + Spatial Features

STEP 4
Training + Evaluation
    |
    v
Temporal Baseline
        +
Fusion Model

STEP 5
Output Generation
    |
    v
CSV Results
Dashboard Data
Evaluation Summary
\`\`\`

## Experimental Results

The final end-to-end experiment contains:

- 23.93 hours of simulated operation
- 30-second simulation steps
- 7 simulated NavIC satellites
- 11,184 training rows
- 3,141 test rows
- 2,873 navigation epochs
- 20,111 satellite-level observations

The attack evaluation contains 90 test rows for each attack type.

### Spoofing Detection

The temporal-only baseline and temporal + spatial fusion model produced the following attack-level results:

| Attack | Temporal Recall | Temporal F1 | Fusion Recall | Fusion F1 |
|---|---|---|---|---|
| Step | 1.000 | 1.000 | 1.000 | 1.000 |
| Drift | 0.967 | 0.983 | 0.989 | 0.994 |
| Evasive | 0.111 | 0.200 | 0.089 | 0.163 |
| Overall | 0.693 | 0.818 | 0.693 | 0.818 |

The fusion model improves drift detection recall by approximately 2.22 percentage points while maintaining perfect step detection.

However, the evasive attack remains difficult. Fusion recall decreases from 0.111 to 0.089 in the reported experiment.

This is an important result rather than something to hide: the experiment demonstrates that spatial evidence does not automatically solve evasive spoofing, and that the current detector has a clear limitation on this attack type.

### Evasive Detection Latency

The evasive test window also provides a useful latency measurement.

\`\`\`text
Spatial detection latency:
1474.36 s  (~24.6 min)

Temporal detection latency:
2404.36 s  (~40.1 min)
\`\`\`

Spatial evidence therefore responded earlier than the temporal detector in this experiment, although its overall classification recall did not improve.

### Navigation Protection

The navigation layer shows a much stronger resilience effect.

Measured position error during the attack windows:

| Attack | Raw Mean Error | Kalman Mean Error | Raw Maximum | Kalman Maximum |
|---|---|---|---|---|
| Step | 145.82 m | 1.90 m | 188.12 m | 1.90 m |
| Drift | 6289.44 m | 2.94 m | 11578.39 m | 2.94 m |
| Evasive | 29.10 m | 18.40 m | 96.41 m | 34.89 m |

The most significant result is the drift scenario:

\`\`\`text
Raw mean error:
6289.44 m

Kalman mean error:
2.94 m
\`\`\`

This demonstrates the central purpose of the project: detecting navigation degradation and using a fallback mechanism to prevent the navigation solution from following a severely corrupted GNSS estimate.

### Navigation Confidence

The final integration run produced:

\`\`\`text
Mean confidence:              0.596
Minimum confidence:           0.000
Epochs below reject threshold: 1153 / 2873
Reject threshold:             0.15
\`\`\`

The dashboard exposes this confidence evolution together with the attack windows and navigation error.

## Dashboard

The project includes an interactive Streamlit dashboard.

Launch it with:

\`\`\`bash
streamlit run dashboard/app.py
\`\`\`

The dashboard provides:

- Mission-level navigation overview
- Satellite visibility
- Satellite-level spoofing probability
- Temporal attack behaviour
- Navigation integrity confidence
- Raw versus protected position error
- Attack-window visualization
- Satellite-level investigation
- Interactive Plotly charts
- Mission timeline interaction

The dashboard reads its primary outputs from:

\`\`\`text
results/csv/
\`\`\`

The main dashboard files are:

- day4_position_results.csv
- day4_satellite_level.csv
- day4_attack_info.csv
- day4_summary.csv
- summary.csv

The dashboard is designed as an analysis interface rather than merely a visualization layer: the objective is to connect signal-level anomaly evidence to the resulting navigation integrity state.

## Repository Structure

\`\`\`text
NavIC-SHIELD/
|
+-- dashboard/
|   +-- app.py
|
+-- features/
|   +-- __init__.py
|   +-- temporal.py
|   +-- spatial.py
|
+-- models/
|   +-- __init__.py
|   +-- baseline_detector.py
|   +-- fusion.py
|
+-- navigation/
|   +-- __init__.py
|   +-- position_solver.py
|   +-- kalman_fallback.py
|
+-- simulator/
|   +-- __init__.py
|   +-- constellation.py
|   +-- terrain.py
|   +-- observation.py
|   +-- receiver.py
|
+-- spoofing/
|   +-- __init__.py
|   +-- step_attack.py
|   +-- drift_attack.py
|   +-- evasive_attack.py
|
+-- scripts/
|   +-- generate_dataset.py
|   +-- run_all.py
|   +-- run_day4_integration.py
|
+-- results/
|   +-- csv/
|   +-- figures/
|
+-- config.py
+-- requirements.txt
+-- README.md
+-- .gitignore
\`\`\`

## Installation

Clone the repository:

\`\`\`bash
git clone https://github.com/shreya-sudo-dev/NavIC-SHIELD.git
cd NavIC-SHIELD
\`\`\`

Create a virtual environment:

\`\`\`bash
python -m venv venv
\`\`\`

Activate it on Windows:

\`\`\`bash
venv\Scripts\activate
\`\`\`

Install dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Running the Project

### Run the complete experiment

\`\`\`bash
python scripts/run_all.py
\`\`\`

This runs the simulator, spoofing scenarios, feature engineering, model training, evaluation, and result generation.

### Generate dashboard outputs

\`\`\`bash
python scripts/run_day4_integration.py
\`\`\`

### Launch the dashboard

\`\`\`bash
streamlit run dashboard/app.py
\`\`\`

## Output Files

The experiment produces the following primary outputs:

\`\`\`text
results/csv/
|
+-- day4_position_results.csv
|   Per-epoch raw and Kalman-protected navigation results
|
+-- day4_satellite_level.csv
|   Satellite-level visibility, geometry, and spoofing information
|
+-- day4_attack_info.csv
|   Attack target and attack-window metadata
|
+-- day4_summary.csv
|   Navigation error summary by attack type
|
+-- full_dataset.csv
|   Full experiment dataset
|
+-- summary.csv
|   Headline temporal-versus-fusion evaluation
\`\`\`

Research-only experimental outputs are not required for the final dashboard and are intentionally excluded from the clean repository.

## Design Principles

### 1. Detection is not enough

A spoofing detector that identifies an attack but allows the navigation solution to follow the corrupted measurements does not provide complete navigation resilience.

NavIC-SHIELD therefore connects:

\`\`\`text
Detection
    |
    v
Integrity Assessment
    |
    v
Navigation Decision
    |
    v
Position Protection
\`\`\`

### 2. Temporal and spatial evidence are complementary

Temporal detection examines a satellite's own history.

Spatial detection examines disagreement with the surrounding satellite constellation.

The project evaluates both independently and in combination rather than assuming that one source of evidence is sufficient.

### 3. Controlled attacks enable measurable evaluation

The simulator provides explicit attack windows and targeted satellites, making it possible to measure:

- Recall
- F1 score
- Detection latency
- Raw navigation error
- Protected navigation error
- Maximum navigation error
- Confidence degradation

### 4. Limitations are part of the result

The evasive attack remains difficult for the current detector.

The reported experiment shows:

\`\`\`text
Fusion evasive recall = 0.0889
\`\`\`

This indicates that the current spatio-temporal detector should not be interpreted as a complete solution to sophisticated evasive spoofing.

Instead, the result identifies a concrete research direction: improving detection of attacks whose temporal and spatial signatures remain close to nominal behaviour.

## Current Research Direction

The current system establishes a complete experimental loop:

\`\`\`text
Satellite Simulation
        |
        v
Terrain-Aware Observations
        |
        v
Controlled Spoofing
        |
        v
Temporal + Spatial Detection
        |
        v
Navigation Integrity
        |
        v
Resilient Positioning
        |
        v
Interactive Evaluation
\`\`\`

Future work can extend this framework toward:

- More realistic NavIC signal and orbital modelling
- Larger and more diverse attack datasets
- Adaptive attack models
- Improved evasive-spoofing detection
- Real GNSS measurement validation
- Online/streaming inference
- More advanced state estimation
- Multi-receiver or cooperative detection
- Field-oriented evaluation in challenging terrain

## Limitations

NavIC-SHIELD is an experimental research framework.

The current evaluation primarily uses controlled synthetic observations and simulated attack scenarios. Therefore, the reported performance should not be interpreted as field performance of a deployed NavIC receiver.

In particular:

- The constellation and observation environment are simulated.
- Attack parameters are controlled.
- The evasive attack remains challenging for the current detector.
- Real-world multipath, atmospheric effects, receiver-specific errors, and signal-processing behaviour are not fully represented.
- The Kalman fallback is evaluated within the simulated navigation environment.

These limitations define the boundary of the current results and provide directions for subsequent validation.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Numerical Computing | NumPy, SciPy |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Version Control | Git / GitHub |
| Navigation | Least-Squares Position Solver + Kalman Filter |
| Simulation | Custom NavIC/GNSS simulation pipeline |

## Research Contribution

The primary contribution of NavIC-SHIELD is the integration of spoofing detection and navigation resilience into a single experimental framework.

Rather than treating spoofing detection as an isolated classification problem, the system evaluates the complete chain:

\`\`\`text
Satellite-Level Anomaly
        |
        v
Spoofing Probability
        |
        v
Navigation Integrity Confidence
        |
        v
Positioning Decision
        |
        v
Protected Navigation
\`\`\`

This makes it possible to evaluate not only whether an attack is detected, but also whether the navigation system remains usable when GNSS measurements become unreliable.

## Status

Project status: Experimental prototype complete

Completed:

- NavIC constellation simulation
- Terrain-aware visibility
- Synthetic GNSS observations
- Step, drift, and evasive attack models
- Temporal detector
- Spatial detector
- Spatio-temporal fusion
- GNSS position solver
- Confidence-aware Kalman fallback
- End-to-end evaluation
- Interactive Streamlit dashboard
- Clean project structure
- Experimental result generation

## Author

Shreya Sunil Sable

B.Tech Information Technology

India

## Disclaimer

NavIC-SHIELD is an academic and research prototype intended for experimentation with GNSS spoofing detection and resilient navigation concepts.

It is not a certified navigation, safety-critical, or operational defence system.