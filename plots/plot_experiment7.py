#!/usr/bin/env python3
"""
Figure 7: Real-World Model Inference Power-Thermal Profile (Experiment 7)

Compares MobileNetV2 ONNX inference power/thermal profile against the synthetic
convolution workload (Experiment 6). Shows that real model inference produces
different thermal dynamics than synthetic benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../data"
OUTPUT_FILE = f"{DATA_DIR}/figure7_mobilenetv2_inference.png"
OUTPUT_PDF = f"{DATA_DIR}/figure7_mobilenetv2_inference.pdf"


def load_and_clean(path):
    df = pd.read_csv(path)
    for col in ["timestamp", "temp_c", "freq_mhz", "voltage_v", "current_ma", "power_mw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "temp_c"])
    df = df[df["timestamp"] > 1_000_000_000]
    df["timestamp"] = df["timestamp"].astype(int)
    df = df.drop_duplicates(subset="timestamp", keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_s"] = df["timestamp"] - df["timestamp"].iloc[0]
    if "power_mw" in df.columns:
        df["power_w"] = df["power_mw"] / 1000.0
    if "throttled" in df.columns:
        df["throttle_int"] = df["throttled"].apply(lambda x: int(str(x).strip(), 16))
    return df


# --- Load data ---
df_mv2 = load_and_clean(f"{DATA_DIR}/experiment7_mobilenetv2.csv")
df_synth = load_and_clean(f"{DATA_DIR}/experiment6_power.csv")

# --- Find stress windows ---
def find_stress_window(df):
    """Find the period of sustained high-frequency operation."""
    high = df[df["freq_mhz"] >= 2400]
    if len(high) == 0:
        return 0, df["time_s"].iloc[-1]
    start = high["time_s"].iloc[0]
    # Look for sustained freq drop after at least 200s of stress
    late = df[(df["time_s"] > start + 200)]
    # Find where power drops significantly (end of workload)
    if "power_w" in df.columns:
        late_power = late[late["power_w"] < late["power_w"].median() * 0.8]
        if len(late_power) > 3:
            end = late_power["time_s"].iloc[0]
        else:
            end = df["time_s"].iloc[-1]
    else:
        end = df["time_s"].iloc[-1]
    return start, end

mv2_start, mv2_end = find_stress_window(df_mv2)
synth_start, synth_end = find_stress_window(df_synth)

# Align to stress start
df_mv2_plot = df_mv2.copy()
df_mv2_plot["time_s"] = df_mv2_plot["time_s"] - mv2_start
df_synth_plot = df_synth.copy()
df_synth_plot["time_s"] = df_synth_plot["time_s"] - synth_start

# --- Compute statistics ---
def compute_stats(df, start, end, label):
    stress = df[(df["time_s"] >= start + 10) & (df["time_s"] <= end)]
    idle = df[df["time_s"] < start]
    stats = {
        "label": label,
        "start_temp": df["temp_c"].iloc[:5].mean(),
        "peak_temp": df["temp_c"].max(),
        "idle_power": idle["power_w"].mean() if len(idle) > 0 and "power_w" in df.columns else None,
        "stress_power_mean": stress["power_w"].mean() if "power_w" in stress.columns else None,
        "stress_power_median": stress["power_w"].median() if "power_w" in stress.columns else None,
        "stress_power_p95": stress["power_w"].quantile(0.95) if "power_w" in stress.columns else None,
        "stress_temp_mean": stress["temp_c"].mean(),
        "peak_power": df["power_w"].max() if "power_w" in df.columns else None,
    }
    # Find throttle onset
    throttle = df[(df["time_s"] >= start) & (df["throttle_int"] & 0x8 > 0)]
    stats["throttle_onset"] = throttle["time_s"].iloc[0] - start if len(throttle) > 0 else None
    return stats

mv2_stats = compute_stats(df_mv2, mv2_start, mv2_end, "MobileNetV2")
synth_stats = compute_stats(df_synth, synth_start, synth_end, "Synthetic Conv")

# Smoothing
window = 7

# =============================================================================
# PLOTTING
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                          gridspec_kw={"height_ratios": [3, 2, 1.2], "hspace": 0.30})
ax_temp, ax_power, ax_freq = axes

# --- Panel 1: Temperature comparison ---
# Synthetic (from Experiment 6)
ax_temp.plot(df_synth_plot["time_s"], df_synth_plot["temp_c"],
             color="#93c5fd", linewidth=0.6, alpha=0.4)
ax_temp.plot(df_synth_plot["time_s"],
             df_synth_plot["temp_c"].rolling(window=window, center=True).mean(),
             color="#2563eb", linewidth=2.2, label="Synthetic Convolution (Exp 6)")

# MobileNetV2
ax_temp.plot(df_mv2_plot["time_s"], df_mv2_plot["temp_c"],
             color="#fca5a5", linewidth=0.6, alpha=0.4)
ax_temp.plot(df_mv2_plot["time_s"],
             df_mv2_plot["temp_c"].rolling(window=window, center=True).mean(),
             color="#dc2626", linewidth=2.2, label="MobileNetV2 ONNX Inference (Exp 7)")

ax_temp.axhline(y=80, color="#f59e0b", linestyle="--", linewidth=1.2, alpha=0.7,
                label="Soft Throttle (80C)")
ax_temp.axhline(y=85, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7,
                label="Throttle Limit (85C)")

ax_temp.set_ylabel("Temperature (C)", fontsize=11)
ax_temp.set_title("Real Model vs Synthetic Workload: Thermal Profile\n"
                  "(Raspberry Pi 4B, 2.4 GHz OC, no heatsink, INA219 sensor)",
                  fontsize=12, fontweight="bold")
ax_temp.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax_temp.grid(True, alpha=0.3)

# --- Panel 2: Power comparison ---
ax_power.plot(df_synth_plot["time_s"],
              df_synth_plot["power_w"].rolling(window=window, center=True).mean(),
              color="#2563eb", linewidth=2.0, label="Synthetic Conv", alpha=0.8)
ax_power.plot(df_mv2_plot["time_s"],
              df_mv2_plot["power_w"].rolling(window=window, center=True).mean(),
              color="#dc2626", linewidth=2.0, label="MobileNetV2", alpha=0.8)

# Add raw data faintly
ax_power.plot(df_synth_plot["time_s"], df_synth_plot["power_w"],
              color="#93c5fd", linewidth=0.5, alpha=0.25)
ax_power.plot(df_mv2_plot["time_s"], df_mv2_plot["power_w"],
              color="#fca5a5", linewidth=0.5, alpha=0.25)

# Stats box
stats_text = (
    f"{'Metric':<22} {'Synth':>8} {'MNv2':>8}\n"
    f"{'Start temp (C)':<22} {synth_stats['start_temp']:>7.1f}  {mv2_stats['start_temp']:>7.1f}\n"
    f"{'Peak temp (C)':<22} {synth_stats['peak_temp']:>7.1f}  {mv2_stats['peak_temp']:>7.1f}\n"
    f"{'Stress power mean (W)':<22} {synth_stats['stress_power_mean']:>7.2f}  {mv2_stats['stress_power_mean']:>7.2f}\n"
    f"{'Stress power P95 (W)':<22} {synth_stats['stress_power_p95']:>7.2f}  {mv2_stats['stress_power_p95']:>7.2f}\n"
    f"{'Peak power (W)':<22} {synth_stats['peak_power']:>7.2f}  {mv2_stats['peak_power']:>7.2f}"
)
props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d1d5db", alpha=0.9)
ax_power.text(0.98, 0.95, stats_text, transform=ax_power.transAxes, fontsize=7.5,
              verticalalignment="top", horizontalalignment="right", bbox=props,
              fontfamily="monospace")

ax_power.set_ylabel("System Power (W)", fontsize=11)
ax_power.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax_power.grid(True, alpha=0.3)

# --- Panel 3: Frequency behavior ---
ax_freq.plot(df_synth_plot["time_s"], df_synth_plot["freq_mhz"],
             color="#2563eb", linewidth=1.0, alpha=0.6, label="Synthetic Conv")
ax_freq.plot(df_mv2_plot["time_s"], df_mv2_plot["freq_mhz"],
             color="#dc2626", linewidth=1.0, alpha=0.6, label="MobileNetV2")

ax_freq.set_xlabel("Time Since Workload Start (seconds)", fontsize=11)
ax_freq.set_ylabel("CPU Freq (MHz)", fontsize=11)
ax_freq.set_title("CPU Frequency (DVFS behavior under load)", fontsize=10, fontweight="bold")
ax_freq.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax_freq.grid(True, alpha=0.3)
ax_freq.set_ylim(1000, 2600)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("EXPERIMENT 7: MobileNetV2 Real-World Inference")
print("=" * 60)
print()

for stats in [synth_stats, mv2_stats]:
    print(f"--- {stats['label']} ---")
    print(f"  Start temp:         {stats['start_temp']:.1f}C")
    print(f"  Peak temp:          {stats['peak_temp']:.1f}C")
    print(f"  Temp rise:          {stats['peak_temp'] - stats['start_temp']:.1f}C")
    if stats['idle_power']:
        print(f"  Idle power:         {stats['idle_power']:.2f}W")
    print(f"  Stress power mean:  {stats['stress_power_mean']:.2f}W")
    print(f"  Stress power med:   {stats['stress_power_median']:.2f}W")
    print(f"  Stress power P95:   {stats['stress_power_p95']:.2f}W")
    print(f"  Peak power:         {stats['peak_power']:.2f}W")
    if stats['throttle_onset']:
        print(f"  Throttle onset:     {stats['throttle_onset']:.0f}s")
    print()

print(f"Saved: {OUTPUT_FILE}, {OUTPUT_PDF}")
