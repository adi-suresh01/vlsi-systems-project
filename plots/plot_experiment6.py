#!/usr/bin/env python3
"""
Figure 6: Predicted vs Measured Power and Thermal Validation (Experiment 6)

Compares the simulation framework's RC thermal model against real measurements
from a Raspberry Pi 4B with INA219 power sensor. Fits R_th and C_th to the
measured data and reports calibration parameters and error metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DATA_DIR = "../data"
OUTPUT_FILE = f"{DATA_DIR}/figure6_predicted_vs_measured.png"
OUTPUT_PDF = f"{DATA_DIR}/figure6_predicted_vs_measured.pdf"

# --- Load and clean data ---
df = pd.read_csv(f"{DATA_DIR}/experiment6_power.csv")
for col in ["timestamp", "temp_c", "freq_mhz", "voltage_v", "current_ma", "power_mw"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["timestamp", "temp_c", "power_mw"])
df = df[df["timestamp"] > 1_000_000_000]
df["timestamp"] = df["timestamp"].astype(int)
df = df.drop_duplicates(subset="timestamp", keep="first")
df = df.sort_values("timestamp").reset_index(drop=True)
df["time_s"] = df["timestamp"] - df["timestamp"].iloc[0]
df["power_w"] = df["power_mw"] / 1000.0
df["throttle_int"] = df["throttled"].apply(lambda x: int(str(x).strip(), 16))

# --- Identify stress window ---
# Stress starts when frequency sustains at 2400 MHz
high_freq = df[df["freq_mhz"] >= 2400]
stress_start_idx = high_freq.index[0] if len(high_freq) > 0 else 0
stress_start_time = df.loc[stress_start_idx, "time_s"]

# Stress ends when power drops and frequency drops after sustained high period
# Look for the transition: sustained 2400 MHz -> freq drops with power drop
late = df[df["time_s"] > stress_start_time + 250]
freq_drops = late[late["freq_mhz"] < 2000]
if len(freq_drops) > 0:
    # Find the cluster of freq drops (stress ending)
    stress_end_time = freq_drops["time_s"].iloc[0]
else:
    stress_end_time = df["time_s"].iloc[-1]

# --- Smooth power for thermal model input ---
# INA219 has noise/spikes; smooth with rolling median then mean
df["power_w_smooth"] = df["power_w"].rolling(window=5, center=True, min_periods=1).median()
df["power_w_smooth"] = df["power_w_smooth"].rolling(window=5, center=True, min_periods=1).mean()

# --- RC thermal model simulation ---
def simulate_rc_thermal(times, power_w, R_th, C_th, T_amb, T_init):
    """Forward Euler integration of dT/dt = (P*R_th - (T-T_amb)) / (R_th*C_th)"""
    temps = np.zeros(len(times))
    temps[0] = T_init
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            temps[i] = temps[i - 1]
            continue
        T = temps[i - 1]
        dTdt = (power_w[i - 1] * R_th - (T - T_amb)) / (R_th * C_th)
        T_new = T + dTdt * dt
        temps[i] = max(T_new, T_amb)
    return temps

# --- Default simulation parameters ---
R_TH_DEFAULT = 10.0   # C/W
C_TH_DEFAULT = 0.5    # J/C
T_AMB = 25.0          # Room ambient

times = df["time_s"].values.astype(float)
power = df["power_w_smooth"].values
temp_measured = df["temp_c"].values
T_init = temp_measured[0]

# Run with default parameters
temp_default = simulate_rc_thermal(times, power, R_TH_DEFAULT, C_TH_DEFAULT, T_AMB, T_init)

# --- Calibrate R_th, C_th to fit measured data ---
def objective(params):
    R_th, C_th = params
    if R_th <= 0 or C_th <= 0:
        return 1e10
    temp_pred = simulate_rc_thermal(times, power, R_th, C_th, T_AMB, T_init)
    # Use stress window only for fitting (ignore initial transient and cool-down)
    mask = (times >= stress_start_time) & (times <= stress_end_time + 60)
    residuals = temp_pred[mask] - temp_measured[mask]
    return np.mean(residuals ** 2)

result = minimize(objective, x0=[10.0, 5.0], method="Nelder-Mead",
                  options={"maxiter": 10000, "xatol": 0.01, "fatol": 0.01})
R_TH_CAL, C_TH_CAL = result.x

# Run with calibrated parameters
temp_calibrated = simulate_rc_thermal(times, power, R_TH_CAL, C_TH_CAL, T_AMB, T_init)

# --- Error metrics ---
stress_mask = (times >= stress_start_time + 10) & (times <= stress_end_time)
mae_default = np.mean(np.abs(temp_default[stress_mask] - temp_measured[stress_mask]))
mae_calibrated = np.mean(np.abs(temp_calibrated[stress_mask] - temp_measured[stress_mask]))
rmse_default = np.sqrt(np.mean((temp_default[stress_mask] - temp_measured[stress_mask]) ** 2))
rmse_calibrated = np.sqrt(np.mean((temp_calibrated[stress_mask] - temp_measured[stress_mask]) ** 2))
max_err_default = np.max(np.abs(temp_default[stress_mask] - temp_measured[stress_mask]))
max_err_calibrated = np.max(np.abs(temp_calibrated[stress_mask] - temp_measured[stress_mask]))

# --- Throttle onset times ---
first_soft = df[df["throttle_int"] & 0x80000 > 0]
first_active = df[df["throttle_int"] & 0x8 > 0]
throttle_soft_time = first_soft["time_s"].iloc[0] - stress_start_time if len(first_soft) > 0 else None
throttle_active_time = first_active["time_s"].iloc[0] - stress_start_time if len(first_active) > 0 else None

# --- Power statistics ---
stress_data = df[(df["time_s"] >= stress_start_time + 10) & (df["time_s"] <= stress_end_time)]
idle_data = df[df["time_s"] < stress_start_time]

idle_power_avg = idle_data["power_w"].mean() if len(idle_data) > 0 else 0
stress_power_avg = stress_data["power_w"].mean()
stress_power_median = stress_data["power_w"].median()
stress_power_p95 = stress_data["power_w"].quantile(0.95)
peak_power = df["power_w"].max()
peak_temp = df["temp_c"].max()

# --- Voltage statistics ---
stress_voltage_avg = stress_data["voltage_v"].mean()
stress_current_avg = stress_data["current_ma"].mean()

# =============================================================================
# PLOTTING
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                          gridspec_kw={"height_ratios": [3, 2, 1], "hspace": 0.30})
ax_temp, ax_power, ax_err = axes

# Align to stress start
t_plot = times - stress_start_time

# --- Panel 1: Temperature comparison ---
ax_temp.plot(t_plot, temp_measured, color="#2563eb", linewidth=1.8,
             label="Measured (SoC sensor)", zorder=3)
ax_temp.plot(t_plot, temp_default, color="#9ca3af", linewidth=1.5, linestyle="--",
             label=f"RC Model, default ($R_{{th}}$={R_TH_DEFAULT}, $C_{{th}}$={C_TH_DEFAULT})",
             alpha=0.8, zorder=2)
ax_temp.plot(t_plot, temp_calibrated, color="#dc2626", linewidth=1.8, linestyle="-",
             label=f"RC Model, calibrated ($R_{{th}}$={R_TH_CAL:.1f}, $C_{{th}}$={C_TH_CAL:.1f})",
             zorder=2)

ax_temp.axhline(y=80, color="#f59e0b", linestyle="--", linewidth=1.2, alpha=0.7,
                label="Soft Throttle (80C)")
ax_temp.axhline(y=85, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7,
                label="Throttle Limit (85C)")
ax_temp.axvline(x=0, color="#6b7280", linestyle=":", linewidth=1, alpha=0.5)
ax_temp.axvline(x=stress_end_time - stress_start_time, color="#6b7280",
                linestyle=":", linewidth=1, alpha=0.5)

# Annotate throttle onset
if throttle_soft_time is not None:
    ax_temp.annotate(f"Soft throttle\n{throttle_soft_time:.0f}s",
                     xy=(throttle_soft_time, 80),
                     xytext=(throttle_soft_time - 40, 60),
                     fontsize=8, color="#f59e0b", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1))

# Add error box
textstr = (f"Stress Phase Error (calibrated):\n"
           f"  MAE = {mae_calibrated:.1f}C\n"
           f"  RMSE = {rmse_calibrated:.1f}C\n"
           f"  Max = {max_err_calibrated:.1f}C")
props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d1d5db", alpha=0.9)
ax_temp.text(0.98, 0.35, textstr, transform=ax_temp.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right", bbox=props,
             fontfamily="monospace")

ax_temp.set_ylabel("Temperature (C)", fontsize=11)
ax_temp.set_title("Predicted vs Measured Thermal Trajectory\n"
                  "(Raspberry Pi 4B, 2.4 GHz OC, no heatsink, INA219 power sensor)",
                  fontsize=12, fontweight="bold")
ax_temp.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax_temp.grid(True, alpha=0.3)

# --- Panel 2: Measured power ---
ax_power.plot(t_plot, df["power_w"].values, color="#8b5cf6", linewidth=0.6, alpha=0.4)
ax_power.plot(t_plot, df["power_w_smooth"].values, color="#7c3aed", linewidth=2.0,
              label=f"Measured power (smoothed)")
ax_power.axhline(y=idle_power_avg, color="#6b7280", linestyle=":", linewidth=1,
                 label=f"Idle baseline ({idle_power_avg:.1f}W)")
ax_power.axhline(y=stress_power_median, color="#059669", linestyle="--", linewidth=1.2,
                 label=f"Stress median ({stress_power_median:.1f}W)")
ax_power.axvline(x=0, color="#6b7280", linestyle=":", linewidth=1, alpha=0.5)
ax_power.axvline(x=stress_end_time - stress_start_time, color="#6b7280",
                 linestyle=":", linewidth=1, alpha=0.5)

# Phase labels
ax_power.text(-5, peak_power * 0.95, "Idle", fontsize=9, color="#6b7280",
              ha="right", fontweight="bold")
ax_power.text((stress_end_time - stress_start_time) / 2, peak_power * 0.95,
              "Stress (4 cores, 300s)", fontsize=9, color="#7c3aed",
              ha="center", fontweight="bold")

# Voltage annotation
ax_power.text(0.98, 0.95,
              f"Bus voltage: {stress_voltage_avg:.2f}V avg\n"
              f"Current: {stress_current_avg:.0f} mA avg",
              transform=ax_power.transAxes, fontsize=8,
              verticalalignment="top", horizontalalignment="right",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="#d1d5db", alpha=0.9),
              fontfamily="monospace")

ax_power.set_ylabel("System Power (W)", fontsize=11)
ax_power.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax_power.grid(True, alpha=0.3)

# --- Panel 3: Prediction error ---
err_default = temp_default - temp_measured
err_calibrated = temp_calibrated - temp_measured

ax_err.plot(t_plot, err_default, color="#9ca3af", linewidth=1.2, linestyle="--",
            label=f"Default (MAE={mae_default:.1f}C)", alpha=0.8)
ax_err.plot(t_plot, err_calibrated, color="#dc2626", linewidth=1.5,
            label=f"Calibrated (MAE={mae_calibrated:.1f}C)")
ax_err.axhline(y=0, color="black", linewidth=0.8)
ax_err.axvline(x=0, color="#6b7280", linestyle=":", linewidth=1, alpha=0.5)
ax_err.axvline(x=stress_end_time - stress_start_time, color="#6b7280",
                linestyle=":", linewidth=1, alpha=0.5)
ax_err.fill_between(t_plot, -2, 2, color="#dcfce7", alpha=0.5, label="+/- 2C band")

ax_err.set_xlabel("Time Since Stress Start (seconds)", fontsize=11)
ax_err.set_ylabel("Error (C)", fontsize=11)
ax_err.set_title("Prediction Error (Predicted - Measured)", fontsize=10, fontweight="bold")
ax_err.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax_err.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("EXPERIMENT 6: Predicted vs Measured Validation")
print("=" * 60)
print()
print("--- Measured Data Summary ---")
print(f"Data points: {len(df)}")
print(f"Duration: {df['time_s'].iloc[-1]:.0f}s total")
print(f"Stress window: {stress_start_time:.0f}s to {stress_end_time - stress_start_time:.0f}s")
print()
print(f"Idle temperature: {idle_data['temp_c'].mean():.1f}C" if len(idle_data) > 0 else "")
print(f"Peak temperature: {peak_temp:.1f}C")
print(f"Idle power: {idle_power_avg:.2f}W")
print(f"Stress power (mean): {stress_power_avg:.2f}W")
print(f"Stress power (median): {stress_power_median:.2f}W")
print(f"Stress power (P95): {stress_power_p95:.2f}W")
print(f"Peak power: {peak_power:.2f}W")
print(f"Bus voltage (stress avg): {stress_voltage_avg:.2f}V")
print(f"Current (stress avg): {stress_current_avg:.0f} mA")
print()
print(f"Soft throttle onset: {throttle_soft_time:.0f}s after stress start" if throttle_soft_time else "No soft throttle")
print(f"Active throttle onset: {throttle_active_time:.0f}s after stress start" if throttle_active_time else "No active throttle")
print()
print("--- Simulation Parameters ---")
print(f"Default:    R_th = {R_TH_DEFAULT:.1f} C/W,  C_th = {C_TH_DEFAULT:.1f} J/C")
print(f"Calibrated: R_th = {R_TH_CAL:.2f} C/W,  C_th = {C_TH_CAL:.2f} J/C")
print(f"Time constant (default):    tau = {R_TH_DEFAULT * C_TH_DEFAULT:.1f}s")
print(f"Time constant (calibrated): tau = {R_TH_CAL * C_TH_CAL:.1f}s")
print(f"T_amb = {T_AMB:.0f}C")
print()
print("--- Prediction Error (stress phase) ---")
print(f"Default model:    MAE = {mae_default:.1f}C,  RMSE = {rmse_default:.1f}C,  Max = {max_err_default:.1f}C")
print(f"Calibrated model: MAE = {mae_calibrated:.1f}C,  RMSE = {rmse_calibrated:.1f}C,  Max = {max_err_calibrated:.1f}C")
print()

# Steady-state comparison
ss_default = T_AMB + stress_power_median * R_TH_DEFAULT
ss_calibrated = T_AMB + stress_power_median * R_TH_CAL
print(f"--- Steady-State Predictions (at median stress power {stress_power_median:.1f}W) ---")
print(f"Default model:    T_ss = {ss_default:.1f}C")
print(f"Calibrated model: T_ss = {ss_calibrated:.1f}C")
print(f"Measured plateau:  ~{stress_data['temp_c'].iloc[-20:].mean():.1f}C")
print()
print(f"Saved: {OUTPUT_FILE}, {OUTPUT_PDF}")
