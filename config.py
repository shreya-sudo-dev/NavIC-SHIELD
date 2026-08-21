"""
config.py

Single source of truth for all constants used across the NavIC GNSS
simulator. Every number here should be traceable to the ISRO NavIC
SPS Signal-in-Space ICD (IRNSS-ICD-SPS, latest revision) or a standard
geodesy/astrodynamics reference. Downstream modules must import from
here rather than hardcoding values.

NOTE: A few numeric fields below are marked "VERIFY AGAINST ICD" —
fill these in against your copy of the ICD table/section before
treating them as final. Where I've given a value, it's the commonly
published NavIC constant, but you should confirm the exact figure
and section number and update the comment accordingly, since your
report will need to cite it.
"""

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0          

EARTH_MU = 3.986005e14                 

EARTH_ROTATION_RATE = 7.2921150e-5     
WGS84_A = 6_378_137.0                   
WGS84_F = 1 / 298.257223563             
WGS84_B = WGS84_A * (1 - WGS84_F)       
WGS84_E2 = 1 - (WGS84_B ** 2) / (WGS84_A ** 2)   

NAVIC_L5_FREQ = 1176.45e6               
NAVIC_S_FREQ = 2492.028e6               

NAVIC_SPS_CHIP_RATE = 1.023e6           

NAVIC_NOMINAL_RX_POWER_DBW = -159.0     

NAVIC_MAX_RX_POWER_DBW = -153.0         

NOISE_FLOOR_DBW_HZ = -204.0             

MIN_ELEVATION_DEG = 5.0                 

GEOSYNCHRONOUS_SMA = 42_164_000.0       
NAVIC_INCLINATION_GSO_DEG = 29.0        
NAVIC_INCLINATION_GEO_DEG = 0.0         

SIDEREAL_DAY_SECONDS = 23 * 3600 + 56 * 60 + 4.0906  # 86164.0906 s

NAVIC_SATELLITES = [
    {"id": "IRNSS-1C", "type": "GEO", "sma": GEOSYNCHRONOUS_SMA,
     "inclination_deg": NAVIC_INCLINATION_GEO_DEG, "longitude_deg": 83.0,
     "raan_deg": 83.0, "mean_anomaly_epoch_deg": 0.0},
    {"id": "IRNSS-1E", "type": "GEO", "sma": GEOSYNCHRONOUS_SMA,
     "inclination_deg": NAVIC_INCLINATION_GEO_DEG, "longitude_deg": 111.75,
     "raan_deg": 111.75, "mean_anomaly_epoch_deg": 0.0},
    {"id": "IRNSS-1F", "type": "GEO", "sma": GEOSYNCHRONOUS_SMA,
     "inclination_deg": NAVIC_INCLINATION_GEO_DEG, "longitude_deg": 32.5,
     "raan_deg": 32.5, "mean_anomaly_epoch_deg": 0.0},
    {"id": "IRNSS-1B", "type": "GSO", "sma": GEOSYNCHRONOUS_SMA,
     "inclination_deg": 29.0, "raan_deg": 55.0,
     "mean_anomaly_epoch_deg": 0.0},
    {"id": "IRNSS-1D", "type": "GSO", "sma": GEOSYNCHRONOUS_SMA,
     "inclination_deg": 29.0, "raan_deg": 111.75,
     "mean_anomaly_epoch_deg": 330.0},
    {"id": "IRNSS-1G", "type": "GSO", "sma": GEOSYNCHRONOUS_SMA,
     "inclination_deg": 29.0, "raan_deg": 55.0,
     "mean_anomaly_epoch_deg": 30.0},
    {"id": "IRNSS-1I", "type": "GSO", "sma": GEOSYNCHRONOUS_SMA,
     "inclination_deg": 29.0, "raan_deg": 111.75,
     "mean_anomaly_epoch_deg": 270.0},
]

RECEIVER_SITE = {
    "name": "Reference site (Ladakh-region, illustrative)",
    "lat_deg": 34.1526,
    "lon_deg": 77.5771,
    "alt_m": 3500.0,   
}

SIM_TIMESTEP_S = 30.0                  
SIM_DURATION_S = SIDEREAL_DAY_SECONDS   
                                         