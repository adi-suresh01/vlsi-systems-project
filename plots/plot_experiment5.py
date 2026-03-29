#!/usr/bin/env python3
"""
Figure 5: Sequential vs Parallel Thermal Comparison (Experiment 5)

Shows how multi-core parallel workloads generate more heat and reach
thermal throttling faster than single-core sequential workloads.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../data"
OUTPUT_FILE = f"{DATA_DIR}/figure5_seq_vs_parallel.png"
OUTPUT_PDF = f"{DATA_DIR}/figure5_seq_vs_parallel.pdf"


def load_and_clean(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    df["freq_mhz"] = pd.to_numeric(df["freq_mhz"], errors="coerce")
    df = df.dropna()
    df = df[df["timestamp"] > 1_000_000_000]
    df["timestamp"] = df["timestamp"].astype(int)
    df = df.drop_duplicates(subset="timestamp", keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_s"] = df["timestamp"] - df["timestamp"].iloc[0]
    df["throttle_int"] = df["throttled"].apply(lambda x: int(str(x).strip(), 16))
    df["soft_temp_occurred"] = (df["throttle_int"] & 0x80000).astype(bool)
    return df


df_seq = load_and_clean(f"{DATA_DIR}/experiment5_sequential.csv")
df_par = load_and_clean(f"{DATA_DIR}/experiment5_parallel.csv")

# Find stress windows (frequency jumps above 1600)
def find_stress_window(df):
    high = df[df["freq_mhz"] > 1600]
    if len(high) > 0:
        start = high["time_s"].iloc[0]
        after = df[(df["time_s"] > start + 60) & (df["freq_mhz"] <= 1600)]
        end = after["time_s"].iloc[0] if len(after) > 0 else df["time_s"].iloc[-1]
        return start, end
    return 0, df["time_s"].iloc[-1]

seq_start, seq_end = find_stress_window(df_seq)
par_start, par_end = find_stress_window(df_par)

# Find throttle onset times (first occurrence of soft temp flag, ignoring sticky flags from before)
# For sequential: flags start at 0x0, so first 0x80000 is real
seq_throttle = df_seq[df_seq["time_s"] >= seq_start]
seq_first_throttle = seq_throttle[seq_throttle["soft_temp_occurred"] & (seq_throttle["throttle_int"] & 0x8 > 0)]
# For parallel: flags start at 0xe0000 (sticky from seq run), so look for active bit 3
par_throttle = df_par[df_par["time_s"] >= par_start]
par_first_throttle = par_throttle[par_throttle["throttle_int"] & 0x8 > 0]

seq_throttle_time = seq_first_throttle["time_s"].iloc[0] - seq_start if len(seq_first_throttle) > 0 else None
par_throttle_time = par_first_throttle["time_s"].iloc[0] - par_start if len(par_first_throttle) > 0 else None

# Align both to stress start = 0
df_seq_plot = df_seq.copy()
df_seq_plot["time_s"] = df_seq_plot["time_s"] - seq_start
df_par_plot = df_par.copy()
df_par_plot["time_s"] = df_par_plot["time_s"] - par_start

# Trim to stress window + cooldown
seq_mask = (df_seq_plot["time_s"] >= -20) & (df_seq_plot["time_s"] <= (seq_end - seq_start) + 120)
par_mask = (df_par_plot["time_s"] >= -20) & (df_par_plot["time_s"] <= (par_end - par_start) + 120)
df_seq_plot = df_seq_plot[seq_mask]
df_par_plot = df_par_plot[par_mask]

# Smoothing
window = 10

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False,
                                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.25})

# Top panel: Temperature comparison
ax1.plot(df_seq_plot["time_s"], df_seq_plot["temp_c"],
         color="#3b82f6", linewidth=0.8, alpha=0.4)
ax1.plot(df_seq_plot["time_s"],
         df_seq_plot["temp_c"].rolling(window=window, center=True).mean(),
         color="#2563eb", linewidth=2.5, label="Sequential (1 core)")

ax1.plot(df_par_plot["time_s"], df_par_plot["temp_c"],
         color="#f87171", linewidth=0.8, alpha=0.4)
ax1.plot(df_par_plot["time_s"],
         df_par_plot["temp_c"].rolling(window=window, center=True).mean(),
         color="#dc2626", linewidth=2.5, label="Parallel (4 cores)")

ax1.axhline(y=80, color="#6b7280", linestyle="--", linewidth=1.5, alpha=0.6,
            label="Soft Throttle Threshold (80 C)")

# Annotate throttle onset
if seq_throttle_time is not None:
    ax1.axvline(x=seq_throttle_time, color="#2563eb", linestyle=":", linewidth=1, alpha=0.6)
    ax1.annotate(f"Seq throttle\n{seq_throttle_time:.0f}s",
                 xy=(seq_throttle_time, 80), xytext=(seq_throttle_time + 20, 74),
                 fontsize=8, color="#2563eb", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1))

if par_throttle_time is not None:
    ax1.axvline(x=par_throttle_time, color="#dc2626", linestyle=":", linewidth=1, alpha=0.6)
    ax1.annotate(f"Par throttle\n{par_throttle_time:.0f}s",
                 xy=(par_throttle_time, 80), xytext=(par_throttle_time + 20, 87),
                 fontsize=8, color="#dc2626", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))

ax1.set_xlabel("Time Since Stress Start (seconds)", fontsize=11)
ax1.set_ylabel("Temperature (C)", fontsize=11)
ax1.set_title("Sequential vs Parallel Thermal Trajectory\n"
              "(Raspberry Pi 4B, 2.4 GHz OC, no heatsink, 500 samples/batch)",
              fontsize=12, fontweight="bold")
ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Bottom panel: Rate of temperature rise (first 120s of stress)
def temp_rise_rate(df, start_time, duration=120):
    mask = (df["time_s"] >= 0) & (df["time_s"] <= duration)
    subset = df[mask].copy()
    if len(subset) < 10:
        return [], []
    smoothed = subset["temp_c"].rolling(window=5, center=True).mean()
    rate = smoothed.diff()  # degrees per second
    return subset["time_s"].values, rate.values

seq_t, seq_rate = temp_rise_rate(df_seq_plot, 0)
par_t, par_rate = temp_rise_rate(df_par_plot, 0)

if len(seq_t) > 0:
    ax2.plot(seq_t, seq_rate, color="#2563eb", linewidth=1.5, alpha=0.7, label="Sequential")
if len(par_t) > 0:
    ax2.plot(par_t, par_rate, color="#dc2626", linewidth=1.5, alpha=0.7, label="Parallel")

ax2.set_xlabel("Time Since Stress Start (seconds)", fontsize=11)
ax2.set_ylabel("Temp Rise Rate (C/s)", fontsize=11)
ax2.set_title("Heating Rate (first 120s)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 120)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")

# Print summary
print("=== Experiment 5 Summary ===")
print(f"Sequential: start {df_seq['temp_c'].iloc[:5].mean():.1f}C, peak {df_seq['temp_c'].max():.1f}C, "
      f"rise {df_seq['temp_c'].max() - df_seq['temp_c'].iloc[:5].mean():.1f}C")
print(f"Parallel:   start {df_par['temp_c'].iloc[:5].mean():.1f}C, peak {df_par['temp_c'].max():.1f}C, "
      f"rise {df_par['temp_c'].max() - df_par['temp_c'].iloc[:5].mean():.1f}C")
if seq_throttle_time is not None:
    print(f"Sequential throttle onset: {seq_throttle_time:.0f}s after stress start")
if par_throttle_time is not None:
    print(f"Parallel throttle onset: {par_throttle_time:.0f}s after stress start")

# Compute average heating rate in first 60s
seq_60 = df_seq_plot[(df_seq_plot["time_s"] >= 0) & (df_seq_plot["time_s"] <= 60)]
par_60 = df_par_plot[(df_par_plot["time_s"] >= 0) & (df_par_plot["time_s"] <= 60)]
if len(seq_60) > 1:
    seq_rate_avg = (seq_60["temp_c"].iloc[-1] - seq_60["temp_c"].iloc[0]) / 60
    print(f"Sequential avg heating rate (first 60s): {seq_rate_avg:.2f} C/s")
if len(par_60) > 1:
    par_rate_avg = (par_60["temp_c"].iloc[-1] - par_60["temp_c"].iloc[0]) / 60
    print(f"Parallel avg heating rate (first 60s): {par_rate_avg:.2f} C/s")

print(f"\nSaved: {OUTPUT_FILE}, {OUTPUT_PDF}")
