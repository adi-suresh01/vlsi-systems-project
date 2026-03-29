#!/usr/bin/env python3
"""
Figure 3: DVFS Efficiency Frontier (Experiment 3)

Maps power vs throughput across 5 DVFS operating points to identify
the efficiency knee where power increase outpaces throughput gain.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../data"
DATA_FILE = f"{DATA_DIR}/experiment3_dvfs.json"
OUTPUT_FILE = f"{DATA_DIR}/figure3_dvfs_frontier.png"
OUTPUT_PDF = f"{DATA_DIR}/figure3_dvfs_frontier.pdf"

# Load JSON (file has a preamble text line before the JSON array)
with open(DATA_FILE) as f:
    lines = f.readlines()
# Find the line where JSON starts
json_start = next(i for i, line in enumerate(lines) if line.strip().startswith("["))
data = json.loads("".join(lines[json_start:]))

names = [d["name"] for d in data]
voltages = [d["voltage"] for d in data]
freqs = [d["frequency_mhz"] for d in data]
power_mw = [d["power_w"] * 1000 for d in data]
throughput = [1.0 / d["theoretical_time_s"] for d in data]  # samples/s
efficiency = [t / (p / 1000) for t, p in zip(throughput, power_mw)]  # samples/s per W

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# --- Panel 1: Power vs Throughput (Pareto frontier) ---
colors = ["#2563eb", "#3b82f6", "#f59e0b", "#ef4444", "#dc2626"]
for i, (name, pw, tp) in enumerate(zip(names, power_mw, throughput)):
    ax1.scatter(tp, pw, color=colors[i], s=120, zorder=5, edgecolors="white", linewidth=1.5)
    offset_x = 40 if i < 3 else -80
    offset_y = 0.15 if i % 2 == 0 else -0.3
    ax1.annotate(
        f"{name}\n({voltages[i]:.1f}V, {freqs[i]:.0f}MHz)",
        xy=(tp, pw),
        xytext=(tp + offset_x, pw + offset_y),
        fontsize=8,
        fontweight="bold",
        color=colors[i],
        ha="center",
    )

# Connect points to show frontier
ax1.plot(throughput, power_mw, color="#94a3b8", linewidth=1.5, linestyle="--", zorder=1)

ax1.set_xlabel("Throughput (samples/s)", fontsize=11)
ax1.set_ylabel("Power (mW)", fontsize=11)
ax1.set_title("DVFS Power-Throughput Frontier", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)

# --- Panel 2: Energy Efficiency across DVFS levels ---
x = np.arange(len(names))
bars = ax2.bar(x, efficiency, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

# Mark the most efficient point
best_idx = np.argmax(efficiency)
bars[best_idx].set_edgecolor("#000000")
bars[best_idx].set_linewidth(2.5)

for i, (eff, name) in enumerate(zip(efficiency, names)):
    label = f"{eff/1000:.1f}K" if eff >= 1000 else f"{eff:.0f}"
    ax2.text(i, eff + max(efficiency) * 0.02, label,
             ha="center", fontsize=9, fontweight="bold", color=colors[i])

ax2.set_xlabel("DVFS Level", fontsize=11)
ax2.set_ylabel("Energy Efficiency (samples/s per W)", fontsize=11)
ax2.set_title("Energy Efficiency by DVFS Level", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{n}\n{v:.1f}V/{f:.0f}MHz" for n, v, f in zip(names, voltages, freqs)],
                     fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("DVFS Operating Point Analysis: 20 Convolution Samples (28x28, 8 Kernels)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")

# Print summary
print(f"{'Level':<14} {'V':>5} {'MHz':>6} {'Power':>10} {'Throughput':>14} {'Efficiency':>14}")
for name, v, f, pw, tp, eff in zip(names, voltages, freqs, power_mw, throughput, efficiency):
    print(f"{name:<14} {v:>5.1f} {f:>6.0f} {pw:>9.2f}mW {tp:>12.0f} s/s {eff:>12.0f} s/s/W")

print(f"\nMost efficient: {names[best_idx]} ({efficiency[best_idx]:.0f} samples/s/W)")
print(f"Turbo vs Ultra-Low: {throughput[-1]/throughput[0]:.1f}x throughput, {power_mw[-1]/power_mw[0]:.1f}x power")

print(f"\nSaved: {OUTPUT_FILE}, {OUTPUT_PDF}")
