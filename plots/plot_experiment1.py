#!/usr/bin/env python3
"""
Figure 1: Thermal Trajectory Under Sustained Load (Experiment 1)

Plots temperature, CPU frequency, and throttle state over time for
a Raspberry Pi 4B (overclocked to 2.4 GHz) running sustained
convolution workloads across all 4 Cortex-A72 cores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_DIR = "../data"
DATA_FILE = f"{DATA_DIR}/experiment1_thermal.csv"
OUTPUT_FILE = f"{DATA_DIR}/figure1_thermal_trajectory.png"
OUTPUT_PDF = f"{DATA_DIR}/figure1_thermal_trajectory.pdf"

# --- Load and clean data ---

df = pd.read_csv(DATA_FILE)

# Drop rows with NaN in any column
df = df.dropna()

# Convert columns to numeric, coercing errors
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
df["freq_mhz"] = pd.to_numeric(df["freq_mhz"], errors="coerce")
df = df.dropna()

# Filter out obviously bad timestamps (must be reasonable epoch values)
df = df[df["timestamp"] > 1_000_000_000]
df["timestamp"] = df["timestamp"].astype(int)

# Convert throttle hex string to int
df["throttle_int"] = df["throttled"].apply(lambda x: int(str(x).strip(), 16) if isinstance(x, str) else int(x))

# Deduplicate by timestamp (monitor may have been restarted)
df = df.drop_duplicates(subset="timestamp", keep="first")
df = df.sort_values("timestamp").reset_index(drop=True)

# Convert to relative time (seconds from start)
t0 = df["timestamp"].iloc[0]
df["time_s"] = df["timestamp"] - t0

# Identify the stress test window:
# Stress starts when frequency jumps above 1500 (to 2400 MHz overclock)
# Stress ends when frequency drops back to 1500 after sustained 2400
freq_above = df[df["freq_mhz"] > 1600]
if len(freq_above) > 0:
    stress_start_s = freq_above["time_s"].iloc[0]
    # Find where freq drops back to 1500 after the sustained high period
    after_peak = df[(df["time_s"] > stress_start_s + 60) & (df["freq_mhz"] <= 1600)]
    if len(after_peak) > 0:
        stress_end_s = after_peak["time_s"].iloc[0]
    else:
        stress_end_s = df["time_s"].iloc[-1]
else:
    stress_start_s = 0
    stress_end_s = df["time_s"].iloc[-1]

# Decode throttle flags
df["soft_temp_active"] = (df["throttle_int"] & 0x8).astype(bool)        # bit 3
df["throttled_active"] = (df["throttle_int"] & 0x4).astype(bool)        # bit 2
df["freq_capped_active"] = (df["throttle_int"] & 0x2).astype(bool)     # bit 1
df["soft_temp_occurred"] = (df["throttle_int"] & 0x80000).astype(bool)  # bit 19

# Trim to relevant window (stress start - 30s to stress end + 120s for cooldown)
plot_start = max(0, stress_start_s - 30)
plot_end = min(df["time_s"].iloc[-1], stress_end_s + 300)
mask = (df["time_s"] >= plot_start) & (df["time_s"] <= plot_end)
df_plot = df[mask].copy()
df_plot["time_s"] = df_plot["time_s"] - plot_start  # rebase to 0

stress_start_plot = stress_start_s - plot_start
stress_end_plot = stress_end_s - plot_start

# --- Compute key stats ---

peak_temp = df_plot["temp_c"].max()
peak_time = df_plot.loc[df_plot["temp_c"].idxmax(), "time_s"]
idle_temp = df_plot["temp_c"].iloc[:10].mean()

first_throttle = df_plot[df_plot["soft_temp_occurred"]]
if len(first_throttle) > 0:
    throttle_onset_time = first_throttle["time_s"].iloc[0]
    throttle_onset_temp = first_throttle["temp_c"].iloc[0]
else:
    throttle_onset_time = None
    throttle_onset_temp = None

print(f"Idle temperature: {idle_temp:.1f} C")
print(f"Peak temperature: {peak_temp:.1f} C at t={peak_time:.0f}s")
print(f"Temperature rise: {peak_temp - idle_temp:.1f} C")
if throttle_onset_time is not None:
    print(f"Throttle onset: {throttle_onset_temp:.1f} C at t={throttle_onset_time:.0f}s")
print(f"Stress window: {stress_start_plot:.0f}s to {stress_end_plot:.0f}s ({stress_end_plot - stress_start_plot:.0f}s)")

# --- Plot ---

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

# Top panel: Temperature
ax1.plot(df_plot["time_s"], df_plot["temp_c"], color="#2563eb", linewidth=1.2, label="Junction Temperature")

# Smoothed trend line (rolling average)
window = min(15, len(df_plot) // 10)
if window > 1:
    smoothed = df_plot["temp_c"].rolling(window=window, center=True).mean()
    ax1.plot(df_plot["time_s"], smoothed, color="#1e40af", linewidth=2.5, alpha=0.8, label=f"Smoothed ({window}s window)")

# Throttle threshold
ax1.axhline(y=80, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.7, label="Soft Throttle Threshold (80 C)")

# Stress window shading
ax1.axvspan(stress_start_plot, stress_end_plot, alpha=0.08, color="#f59e0b", label="Stress Period")

# Throttle onset marker
if throttle_onset_time is not None:
    ax1.annotate(
        f"Throttle onset\n{throttle_onset_temp:.1f} C",
        xy=(throttle_onset_time, throttle_onset_temp),
        xytext=(throttle_onset_time + 30, throttle_onset_temp - 8),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
        color="#dc2626",
        fontweight="bold",
    )

# Peak marker
ax1.annotate(
    f"Peak: {peak_temp:.1f} C",
    xy=(peak_time, peak_temp),
    xytext=(peak_time + 30, peak_temp + 2),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#1e40af", lw=1.5),
    color="#1e40af",
    fontweight="bold",
)

ax1.set_ylabel("Temperature (C)", fontsize=12)
ax1.set_ylim(40, peak_temp + 8)
ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_title("Raspberry Pi 4B Thermal Trajectory Under Sustained Inference Load\n"
              "(4x Cortex-A72 @ 2.4 GHz OC, no heatsink, 500 samples/batch x 4 threads)",
              fontsize=12, fontweight="bold", pad=12)

# Bottom panel: CPU Frequency
ax2.plot(df_plot["time_s"], df_plot["freq_mhz"], color="#059669", linewidth=1.5, label="CPU Frequency")
ax2.axvspan(stress_start_plot, stress_end_plot, alpha=0.08, color="#f59e0b")
ax2.set_ylabel("Frequency (MHz)", fontsize=12)
ax2.set_xlabel("Time (seconds)", fontsize=12)
ax2.set_ylim(1000, 2800)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_FILE}, {OUTPUT_PDF}")
