#!/usr/bin/env python3
"""
Figure 8: Latency Degradation Under Sustained Thermal Stress (Experiment 8c)

Compares MobileNetV2 and SqueezeNet latency degradation over 300 seconds
of thermally stressed inference on Raspberry Pi 5.

Key finding: both models degrade ~15-16% but through different mechanisms.
MobileNetV2 triggers DVFS frequency reduction (2400 -> 1500 MHz).
SqueezeNet stays at full frequency but accumulates OS thermal interrupts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../data"
OUTPUT_FILE = f"{DATA_DIR}/figure8_latency_degradation.png"
OUTPUT_PDF = f"{DATA_DIR}/figure8_latency_degradation.pdf"

# Colors consistent with experiment 7
C_MV2 = "#dc2626"     # red
C_SQZ = "#059669"     # green
C_MV2_LIGHT = "#fca5a5"
C_SQZ_LIGHT = "#6ee7b7"
C_THROTTLE = "#f59e0b"
C_LIMIT = "#ef4444"


def load_latency(path):
    df = pd.read_csv(path)
    df["timestamp_epoch_ms"] = pd.to_numeric(df["timestamp_epoch_ms"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df.dropna()
    df["time_s"] = (df["timestamp_epoch_ms"] - df["timestamp_epoch_ms"].iloc[0]) / 1000.0
    return df


def load_power(path):
    df = pd.read_csv(path)
    for col in ["timestamp", "temp_c", "freq_mhz", "voltage_v", "current_ma", "power_mw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "temp_c"])
    df["timestamp"] = df["timestamp"].astype(int)
    df = df.drop_duplicates(subset="timestamp", keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "power_mw" in df.columns:
        df["power_w"] = df["power_mw"] / 1000.0
    return df


def window_stats(df_lat, window_sec=30, duration=300):
    """Compute mean latency in fixed-size time windows."""
    windows = []
    for w in range(int(duration / window_sec)):
        t0 = w * window_sec
        t1 = t0 + window_sec
        mask = (df_lat["time_s"] >= t0) & (df_lat["time_s"] < t1)
        subset = df_lat.loc[mask, "latency_ms"]
        if len(subset) > 0:
            windows.append({
                "window_start": t0,
                "window_mid": t0 + window_sec / 2,
                "mean": subset.mean(),
                "median": subset.median(),
                "p95": subset.quantile(0.95),
                "count": len(subset),
            })
    return pd.DataFrame(windows)


def align_power_to_inference(df_power, infer_start_epoch_s, duration=300):
    """Extract power data during the inference window."""
    mask = (df_power["timestamp"] >= infer_start_epoch_s) & \
           (df_power["timestamp"] <= infer_start_epoch_s + duration)
    df = df_power.loc[mask].copy()
    df["time_s"] = df["timestamp"] - infer_start_epoch_s
    return df


# --- Load data ---
lat_mv2 = load_latency(f"{DATA_DIR}/experiment8c_mobilenetv2_latency.csv")
lat_sqz = load_latency(f"{DATA_DIR}/experiment8c_squeezenet_latency.csv")
pwr_mv2 = load_power(f"{DATA_DIR}/experiment8c_mobilenetv2_power.csv")
pwr_sqz = load_power(f"{DATA_DIR}/experiment8c_squeezenet_power.csv")

# Inference start times (epoch seconds, from latency timestamps)
mv2_infer_start = int(lat_mv2["timestamp_epoch_ms"].iloc[0] / 1000)
sqz_infer_start = int(lat_sqz["timestamp_epoch_ms"].iloc[0] / 1000)

# Align power data to inference window
pwr_mv2_inf = align_power_to_inference(pwr_mv2, mv2_infer_start)
pwr_sqz_inf = align_power_to_inference(pwr_sqz, sqz_infer_start)

# Window statistics
win_mv2 = window_stats(lat_mv2)
win_sqz = window_stats(lat_sqz)

# Normalize to first window
mv2_base = win_mv2["mean"].iloc[0]
sqz_base = win_sqz["mean"].iloc[0]
win_mv2["normalized"] = win_mv2["mean"] / mv2_base * 100
win_sqz["normalized"] = win_sqz["mean"] / sqz_base * 100

# =============================================================================
# PLOTTING: 3 panels
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                          gridspec_kw={"height_ratios": [3, 3, 2], "hspace": 0.32})
ax_norm, ax_raw, ax_temp = axes

# --- Panel 1: Normalized latency (% of baseline) ---
ax_norm.plot(win_mv2["window_mid"], win_mv2["normalized"],
             color=C_MV2, linewidth=2.5, marker="o", markersize=6,
             label=f"MobileNetV2 (baseline {mv2_base:.1f} ms)")
ax_norm.plot(win_sqz["window_mid"], win_sqz["normalized"],
             color=C_SQZ, linewidth=2.5, marker="s", markersize=6,
             label=f"SqueezeNet (baseline {sqz_base:.1f} ms)")

ax_norm.axhline(y=100, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax_norm.set_ylabel("Latency (% of baseline)", fontsize=11)
ax_norm.set_title("Latency Degradation Under Sustained Thermal Stress\n"
                  "(Raspberry Pi 5, 2.4 GHz, no heatsink, pre-heated start)",
                  fontsize=12, fontweight="bold")
ax_norm.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax_norm.grid(True, alpha=0.3)
ax_norm.set_ylim(95, 120)

# Annotate final degradation
mv2_final = win_mv2["normalized"].iloc[-1]
sqz_final = win_sqz["normalized"].iloc[-1]
ax_norm.annotate(f"+{mv2_final - 100:.0f}%",
                 xy=(win_mv2["window_mid"].iloc[-1], mv2_final),
                 xytext=(10, 5), textcoords="offset points",
                 fontsize=10, fontweight="bold", color=C_MV2)
ax_norm.annotate(f"+{sqz_final - 100:.0f}%",
                 xy=(win_sqz["window_mid"].iloc[-1], sqz_final),
                 xytext=(10, -12), textcoords="offset points",
                 fontsize=10, fontweight="bold", color=C_SQZ)

# --- Panel 2: Raw latency (30s window means) ---
ax_mv2 = ax_raw
ax_sqz = ax_raw.twinx()

bars_mv2 = ax_mv2.bar(win_mv2["window_mid"] - 5, win_mv2["mean"],
                        width=9, color=C_MV2_LIGHT, edgecolor=C_MV2,
                        linewidth=0.8, alpha=0.8, label="MobileNetV2")
bars_sqz = ax_sqz.bar(win_sqz["window_mid"] + 5, win_sqz["mean"],
                        width=9, color=C_SQZ_LIGHT, edgecolor=C_SQZ,
                        linewidth=0.8, alpha=0.8, label="SqueezeNet")

ax_mv2.set_ylabel("MobileNetV2 Latency (ms)", fontsize=10, color=C_MV2)
ax_sqz.set_ylabel("SqueezeNet Latency (ms)", fontsize=10, color=C_SQZ)
ax_mv2.tick_params(axis="y", labelcolor=C_MV2)
ax_sqz.tick_params(axis="y", labelcolor=C_SQZ)

ax_mv2.set_ylim(200, 280)
ax_sqz.set_ylim(34, 46)

lines_mv2, labels_mv2 = ax_mv2.get_legend_handles_labels()
lines_sqz, labels_sqz = ax_sqz.get_legend_handles_labels()
ax_mv2.legend(lines_mv2 + lines_sqz, labels_mv2 + labels_sqz,
              loc="upper left", fontsize=9, framealpha=0.9)
ax_mv2.grid(True, alpha=0.3)

# --- Panel 3: Temperature during inference ---
window = 5
ax_temp.plot(pwr_mv2_inf["time_s"],
             pwr_mv2_inf["temp_c"].rolling(window=window, center=True).mean(),
             color=C_MV2, linewidth=2.2, label="MobileNetV2")
ax_temp.plot(pwr_sqz_inf["time_s"],
             pwr_sqz_inf["temp_c"].rolling(window=window, center=True).mean(),
             color=C_SQZ, linewidth=2.2, label="SqueezeNet")

ax_temp.axhline(y=80, color=C_THROTTLE, linestyle="--", linewidth=1.2, alpha=0.7,
                label="Soft Throttle (80C)")
ax_temp.axhline(y=85, color=C_LIMIT, linestyle="--", linewidth=1.2, alpha=0.7,
                label="Throttle Limit (85C)")

ax_temp.set_xlabel("Time Since Inference Start (seconds)", fontsize=11)
ax_temp.set_ylabel("Temperature (C)", fontsize=11)
ax_temp.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax_temp.grid(True, alpha=0.3)
ax_temp.set_ylim(70, 90)

# Summary stats box
stats_text = (
    f"MobileNetV2: {mv2_base:.1f} ms -> {win_mv2['mean'].iloc[-1]:.1f} ms "
    f"(+{(win_mv2['mean'].iloc[-1]/mv2_base - 1)*100:.0f}%), freq 2400->1500 MHz\n"
    f"SqueezeNet:  {sqz_base:.1f} ms -> {win_sqz['mean'].iloc[-1]:.1f} ms "
    f"(+{(win_sqz['mean'].iloc[-1]/sqz_base - 1)*100:.0f}%), freq stayed at 2400 MHz"
)
props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d1d5db", alpha=0.9)
ax_temp.text(0.02, 0.15, stats_text, transform=ax_temp.transAxes, fontsize=8,
             verticalalignment="bottom", bbox=props, fontfamily="monospace")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("EXPERIMENT 8c: Latency Degradation Under Thermal Stress")
print("=" * 60)
print()
print("--- MobileNetV2 ---")
print(f"  Inferences:    {len(lat_mv2)}")
print(f"  Duration:      {lat_mv2['time_s'].iloc[-1]:.1f}s")
print(f"  Baseline:      {mv2_base:.1f} ms (first 30s mean)")
print(f"  Final window:  {win_mv2['mean'].iloc[-1]:.1f} ms")
print(f"  Degradation:   {(win_mv2['mean'].iloc[-1]/mv2_base - 1)*100:.1f}%")
print(f"  Temp range:    {pwr_mv2_inf['temp_c'].min():.1f} - {pwr_mv2_inf['temp_c'].max():.1f} C")
print()
print("--- SqueezeNet ---")
print(f"  Inferences:    {len(lat_sqz)}")
print(f"  Duration:      {lat_sqz['time_s'].iloc[-1]:.1f}s")
print(f"  Baseline:      {sqz_base:.1f} ms (first 30s mean)")
print(f"  Final window:  {win_sqz['mean'].iloc[-1]:.1f} ms")
print(f"  Degradation:   {(win_sqz['mean'].iloc[-1]/sqz_base - 1)*100:.1f}%")
print(f"  Temp range:    {pwr_sqz_inf['temp_c'].min():.1f} - {pwr_sqz_inf['temp_c'].max():.1f} C")
print()
print(f"Saved: {OUTPUT_FILE}, {OUTPUT_PDF}")
