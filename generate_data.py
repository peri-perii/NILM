"""
generate_data.py
----------------
Generates synthetic household power consumption data for one full day
(minute-by-minute resolution) with realistic appliance schedules.

Appliance schedules:
  - Refrigerator  : 150 W, cycles 30 min ON / 20 min OFF all day
  - AC            : 2000 W, 1 PM – 4 PM
  - Washing Machine: 500 W, 9 AM – 10 AM
  - Microwave     : 1200 W, 8:00–8:05 AM and 7:00–7:05 PM
  - TV            : 100 W, 8 PM – 11 PM
  - Water Heater  : 1500 W, 7:00–7:30 AM and 9:00–9:30 PM

Gaussian noise (μ=0, σ=20) is added to simulate real-world meter jitter.

Outputs:
  household_power.csv        — (timestamp, total_power_watts)
  appliance_ground_truth.csv — per-appliance wattage at each minute
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Time axis: one full day at 1-minute resolution ───────────────────────────
START = datetime(2024, 1, 15, 0, 0, 0)
MINUTES_IN_DAY = 1440
timestamps = [START + timedelta(minutes=i) for i in range(MINUTES_IN_DAY)]

# ── Helper: build a boolean mask for a time range ────────────────────────────
def time_mask(start_h: int, start_m: int, end_h: int, end_m: int) -> np.ndarray:
    """Return a boolean array (length 1440) that is True within the given window."""
    mask = np.zeros(MINUTES_IN_DAY, dtype=bool)
    start_min = start_h * 60 + start_m
    end_min   = end_h   * 60 + end_m
    mask[start_min:end_min] = True
    return mask


# ── Appliance profiles ────────────────────────────────────────────────────────

# 1. Refrigerator — duty-cycle 30 min ON / 20 min OFF throughout the day
fridge = np.zeros(MINUTES_IN_DAY)
cycle_len = 50          # 30 on + 20 off
for i in range(MINUTES_IN_DAY):
    pos = i % cycle_len
    if pos < 30:        # ON phase
        fridge[i] = 150.0

# 2. AC — solid block 1 PM to 4 PM
ac = np.zeros(MINUTES_IN_DAY)
ac_mask = time_mask(13, 0, 16, 0)
ac[ac_mask] = 2000.0

# 3. Washing Machine — 9 AM to 10 AM
washing = np.zeros(MINUTES_IN_DAY)
wash_mask = time_mask(9, 0, 10, 0)
washing[wash_mask] = 500.0

# 4. Microwave — two short bursts
microwave = np.zeros(MINUTES_IN_DAY)
microwave[time_mask(8, 0,  8, 5)] = 1200.0
microwave[time_mask(19, 0, 19, 5)] = 1200.0

# 5. TV — 8 PM to 11 PM
tv = np.zeros(MINUTES_IN_DAY)
tv[time_mask(20, 0, 23, 0)] = 100.0

# 6. Water Heater — two sessions
water_heater = np.zeros(MINUTES_IN_DAY)
water_heater[time_mask(7,  0, 7,  30)] = 1500.0
water_heater[time_mask(21, 0, 21, 30)] = 1500.0


# ── Aggregate signal + noise ──────────────────────────────────────────────────
noise = np.random.normal(loc=0, scale=20, size=MINUTES_IN_DAY)
total_power = fridge + ac + washing + microwave + tv + water_heater + noise

# Clip negatives (noise could push low values below 0)
total_power = np.clip(total_power, 0, None)


# ── Build DataFrames ──────────────────────────────────────────────────────────
df_power = pd.DataFrame({
    "timestamp":         [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
    "total_power_watts": np.round(total_power, 2),
})

df_ground_truth = pd.DataFrame({
    "timestamp":       [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
    "Refrigerator":    np.round(fridge,        2),
    "AC":              np.round(ac,            2),
    "WashingMachine":  np.round(washing,       2),
    "Microwave":       np.round(microwave,     2),
    "TV":              np.round(tv,            2),
    "WaterHeater":     np.round(water_heater,  2),
})


# ── Save ──────────────────────────────────────────────────────────────────────
df_power.to_csv("household_power.csv", index=False)
df_ground_truth.to_csv("appliance_ground_truth.csv", index=False)


# ── Summary ───────────────────────────────────────────────────────────────────
total_kwh = df_power["total_power_watts"].sum() / 60 / 1000   # W·min → kWh

print("=" * 55)
print("  Data Generation Complete")
print("=" * 55)
print(f"  Rows generated  : {len(df_power):,} (one per minute)")
print(f"  Total energy    : {total_kwh:.3f} kWh")
print(f"  Files saved:")
print(f"    • household_power.csv")
print(f"    • appliance_ground_truth.csv")
print("=" * 55)

# Per-appliance summary
appliances = {
    "Refrigerator":    fridge,
    "AC":              ac,
    "Washing Machine": washing,
    "Microwave":       microwave,
    "TV":              tv,
    "Water Heater":    water_heater,
}
print("\n  Per-appliance energy:")
for name, arr in appliances.items():
    kwh = arr.sum() / 60 / 1000
    print(f"    {name:<18} {kwh:.3f} kWh")
print("=" * 55)
