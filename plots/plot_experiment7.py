#!/usr/bin/env python3
"""
Figure 7: Real-World Model Inference Power-Thermal Profiles (Experiment 7)

Compares three workloads on Raspberry Pi 5:
  1. Synthetic convolution (Experiment 6) -- multi-threaded, sustained
  2. MobileNetV2 ONNX inference -- real vision model, single-threaded
  3. SqueezeNet ONNX inference -- lightweight vision model, single-threaded

Shows that all three hit the same thermal ceiling regardless of model size,
because system power is determined by CPU utilization, not model complexity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../data"
OUTPUT_FILE = f"{DATA_DIR}/figure7_real_model_inference.png"
OUTPUT_PDF = f"{DATA_DIR}/figure7_real_model_inference.pdf"

# Colors
C_SYNTH = "#2563eb"   # blue
C_MV2 = "#dc2626"     # red
C_SQZ = "#059669"     # green


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


def find_stress_window(df):
    high = df[df["freq_mhz"] >= 2400]
    if len(high) == 0:
        return 0, df["time_s"].iloc[-1]
    start = high["time_s"].iloc[0]
    late = df[(df["time_s"] > start + 200)]
    if "power_w" in df.columns and len(late) > 0:
        late_power = late[late["power_w"] < late["power_w"].median() * 0.8]
        if len(late_power) > 3:
            return start, late_power["time_s"].iloc[0]
    return start, df["time_s"].iloc[-1]


def compute_stats(df, start, end, label):
    stress = df[(df["time_s"] >= start + 10) & (df["time_s"] <= end)]
    stats = {
        "label": label,
        "start_temp": df["temp_c"].iloc[:5].mean(),
        "peak_temp": df["temp_c"].max(),
        "stress_power_mean": stress["power_w"].mean() if "power_w" in stress.columns else None,
        "stress_power_median": stress["power_w"].median() if "power_w" in stress.columns else None,
        "stress_power_p95": stress["power_w"].quantile(0.95) if "power_w" in stress.columns else None,
        "peak_power": df["power_w"].max() if "power_w" in df.columns else None,
    }
    throttle = df[(df["time_s"] >= start) & (df["throttle_int"] & 0x8 > 0)]
    stats["throttle_onset"] = throttle["time_s"].iloc[0] - start if len(throttle) > 0 else None
    return stats


# --- Load all three datasets ---
df_synth = load_and_clean(f"{DATA_DIR}/experiment6_power.csv")
df_mv2 = load_and_clean(f"{DATA_DIR}/experiment7_mobilenetv2.csv")
df_sqz = load_and_clean(f"{DATA_DIR}/experiment7_squeezenet.csv")

# --- Find stress windows and align ---
synth_start, synth_end = find_stress_window(df_synth)
mv2_start, mv2_end = find_stress_window(df_mv2)
sqz_start, sqz_end = find_stress_window(df_sqz)

df_synth_plot = df_synth.copy()
df_synth_plot["time_s"] = df_synth_plot["time_s"] - synth_start
df_mv2_plot = df_mv2.copy()
df_mv2_plot["time_s"] = df_mv2_plot["time_s"] - mv2_start
df_sqz_plot = df_sqz.copy()
df_sqz_plot["time_s"] = df_sqz_plot["time_s"] - sqz_start

synth_stats = compute_stats(df_synth, synth_start, synth_end, "Synthetic Conv")
mv2_stats = compute_stats(df_mv2, mv2_start, mv2_end, "MobileNetV2")
sqz_stats = compute_stats(df_sqz, sqz_start, sqz_end, "SqueezeNet")

window = 7

# =============================================================================
# PLOTTING
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(11, 8),
                          gridspec_kw={"height_ratios": [3, 2], "hspace": 0.28})
ax_temp, ax_power = axes

# --- Panel 1: Temperature comparison (3 workloads) ---
for df_plot, color, light, label in [
    (df_synth_plot, C_SYNTH, "#93c5fd", "Synthetic Conv (4-thread, Exp 6)"),
    (df_mv2_plot, C_MV2, "#fca5a5", "MobileNetV2 (3.8 inf/s, 266ms)"),
    (df_sqz_plot, C_SQZ, "#6ee7b7", "SqueezeNet (25.2 inf/s, 40ms)"),
]:
    ax_temp.plot(df_plot["time_s"], df_plot["temp_c"],
                 color=light, linewidth=0.5, alpha=0.35)
    ax_temp.plot(df_plot["time_s"],
                 df_plot["temp_c"].rolling(window=window, center=True).mean(),
                 color=color, linewidth=2.2, label=label)

ax_temp.axhline(y=80, color="#f59e0b", linestyle="--", linewidth=1.2, alpha=0.7,
                label="Soft Throttle (80C)")
ax_temp.axhline(y=85, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7,
                label="Throttle Limit (85C)")

ax_temp.set_ylabel("Temperature (C)", fontsize=11)
ax_temp.set_title("Thermal Profiles: Synthetic vs Real Model Inference\n"
                  "(Raspberry Pi 5, 2.4 GHz, no heatsink, INA219 sensor)",
                  fontsize=12, fontweight="bold")
ax_temp.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax_temp.grid(True, alpha=0.3)

# --- Panel 2: Power comparison (3 workloads) ---
for df_plot, color, light, label in [
    (df_synth_plot, C_SYNTH, "#93c5fd", "Synthetic Conv"),
    (df_mv2_plot, C_MV2, "#fca5a5", "MobileNetV2"),
    (df_sqz_plot, C_SQZ, "#6ee7b7", "SqueezeNet"),
]:
    ax_power.plot(df_plot["time_s"], df_plot["power_w"],
                  color=light, linewidth=0.4, alpha=0.2)
    ax_power.plot(df_plot["time_s"],
                  df_plot["power_w"].rolling(window=window, center=True).mean(),
                  color=color, linewidth=2.0, label=label, alpha=0.85)

# Stats table
header = f"{'Metric':<20} {'Synth':>7} {'MNv2':>7} {'SqNet':>7}"
rows = [
    f"{'Start temp (C)':<20} {synth_stats['start_temp']:>6.1f}  {mv2_stats['start_temp']:>6.1f}  {sqz_stats['start_temp']:>6.1f}",
    f"{'Peak temp (C)':<20} {synth_stats['peak_temp']:>6.1f}  {mv2_stats['peak_temp']:>6.1f}  {sqz_stats['peak_temp']:>6.1f}",
    f"{'Power median (W)':<20} {synth_stats['stress_power_median']:>6.2f}  {mv2_stats['stress_power_median']:>6.2f}  {sqz_stats['stress_power_median']:>6.2f}",
    f"{'Power P95 (W)':<20} {synth_stats['stress_power_p95']:>6.2f}  {mv2_stats['stress_power_p95']:>6.2f}  {sqz_stats['stress_power_p95']:>6.2f}",
]
stats_text = header + "\n" + "\n".join(rows)
props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d1d5db", alpha=0.9)
ax_power.text(0.98, 0.95, stats_text, transform=ax_power.transAxes, fontsize=7.5,
              verticalalignment="top", horizontalalignment="right", bbox=props,
              fontfamily="monospace")

ax_power.set_xlabel("Time Since Workload Start (seconds)", fontsize=11)
ax_power.set_ylabel("System Power (W)", fontsize=11)
ax_power.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax_power.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("EXPERIMENT 7: Real-World Model Inference Comparison")
print("=" * 60)
print()

# Inference performance
print("--- Inference Performance ---")
print(f"  {'Workload':<20} {'Throughput':>12} {'Latency':>10}")
print(f"  {'Synthetic Conv':<20} {'continuous':>12} {'N/A':>10}")
print(f"  {'MobileNetV2':<20} {'3.8 inf/s':>12} {'266.2ms':>10}")
print(f"  {'SqueezeNet':<20} {'25.2 inf/s':>12} {'39.7ms':>10}")
print()

for stats in [synth_stats, mv2_stats, sqz_stats]:
    print(f"--- {stats['label']} ---")
    print(f"  Start temp:         {stats['start_temp']:.1f}C")
    print(f"  Peak temp:          {stats['peak_temp']:.1f}C")
    print(f"  Temp rise:          {stats['peak_temp'] - stats['start_temp']:.1f}C")
    print(f"  Stress power mean:  {stats['stress_power_mean']:.2f}W")
    print(f"  Stress power med:   {stats['stress_power_median']:.2f}W")
    print(f"  Stress power P95:   {stats['stress_power_p95']:.2f}W")
    print(f"  Peak power:         {stats['peak_power']:.2f}W")
    if stats['throttle_onset'] is not None:
        print(f"  Throttle onset:     {stats['throttle_onset']:.0f}s")
    else:
        print(f"  Throttle onset:     N/A")
    print()

print(f"Saved: {OUTPUT_FILE}, {OUTPUT_PDF}")
