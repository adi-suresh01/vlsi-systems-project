#!/usr/bin/env python3
"""
Figure 2: Sequence Length vs Power -- Quadratic Attention Scaling (Experiment 2)

Shows how transformer attention's O(n^2) complexity drives nonlinear power
growth with sequence length across three model architectures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_DIR = "../data"
MODELS = [
    ("experiment2_tinybert_sweep.csv", "TinyBERT (6L, d=312)", "#2563eb", "o"),
    ("experiment2_distilgpt2_sweep.csv", "DistilGPT-2 (6L, d=768)", "#dc2626", "s"),
    ("experiment2_bertbase_sweep.csv", "BERT-Base (12L, d=768)", "#059669", "^"),
]

OUTPUT_FILE = f"{DATA_DIR}/figure2_sequence_scaling.png"
OUTPUT_PDF = f"{DATA_DIR}/figure2_sequence_scaling.pdf"

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# --- Panel 1: Total Power vs Sequence Length ---
ax1 = axes[0]
for fname, label, color, marker in MODELS:
    df = pd.read_csv(f"{DATA_DIR}/{fname}")
    ax1.plot(df["seq_length"], df["power_mw"], color=color, marker=marker,
             linewidth=2, markersize=7, label=label)

ax1.set_xlabel("Sequence Length", fontsize=11)
ax1.set_ylabel("Estimated Power (mW)", fontsize=11)
ax1.set_title("Total Power vs Sequence Length", fontsize=11, fontweight="bold")
ax1.set_yscale("log")
ax1.set_xticks([64, 128, 256, 512])
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.3, which="both")

# --- Panel 2: Attention MACs vs Linear MACs (stacked, one model) ---
ax2 = axes[1]
# Use DistilGPT-2 as the example
df_ex = pd.read_csv(f"{DATA_DIR}/experiment2_distilgpt2_sweep.csv")
x = np.arange(len(df_ex))
width = 0.6
seq_labels = df_ex["seq_length"].astype(str)

attn = df_ex["attention_macs"] / 1e9
linear = df_ex["linear_macs"] / 1e9

bars_linear = ax2.bar(x, linear, width, label="Linear MACs", color="#93c5fd")
bars_attn = ax2.bar(x, attn, width, bottom=linear, label="Attention MACs (O(n^2))", color="#dc2626", alpha=0.85)

ax2.set_xlabel("Sequence Length", fontsize=11)
ax2.set_ylabel("MACs (billions)", fontsize=11)
ax2.set_title("MAC Breakdown: DistilGPT-2", fontsize=11, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(seq_labels)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.3, axis="y")

# Add percentage labels on attention bars
for i, (a, l) in enumerate(zip(attn, linear)):
    total = a + l
    pct = a / total * 100
    ax2.text(i, total + total * 0.02, f"{pct:.0f}%", ha="center", fontsize=8,
             color="#dc2626", fontweight="bold")

# --- Panel 3: Steady-State Temperature ---
ax3 = axes[2]
for fname, label, color, marker in MODELS:
    df = pd.read_csv(f"{DATA_DIR}/{fname}")
    ax3.plot(df["seq_length"], df["steady_state_temp_c"], color=color, marker=marker,
             linewidth=2, markersize=7, label=label)

ax3.axhline(y=85, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.6, label="Throttle Threshold (85 C)")
ax3.set_xlabel("Sequence Length", fontsize=11)
ax3.set_ylabel("Steady-State Temperature (C)", fontsize=11)
ax3.set_title("Predicted Thermal Impact", fontsize=11, fontweight="bold")
ax3.set_xticks([64, 128, 256, 512])
ax3.legend(fontsize=8, loc="upper left")
ax3.grid(True, alpha=0.3)

fig.suptitle("Transformer Attention Scaling: O(n^2) Power Growth with Sequence Length",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")

# Print summary table
print("Sequence Length Scaling Summary (DistilGPT-2):")
print(f"{'Seq Len':>8} {'Total MACs':>14} {'Attn %':>8} {'Power (mW)':>12} {'Temp (C)':>10}")
for _, row in df_ex.iterrows():
    attn_pct = row["attention_macs"] / row["total_macs"] * 100
    print(f"{int(row['seq_length']):>8} {row['total_macs']:>14,.0f} {attn_pct:>7.1f}% {row['power_mw']:>11.1f} {row['steady_state_temp_c']:>9.1f}")

print(f"\n4x scaling check (128->256): MACs {df_ex.iloc[2]['total_macs']/df_ex.iloc[1]['total_macs']:.2f}x, "
      f"Attn MACs {df_ex.iloc[2]['attention_macs']/df_ex.iloc[1]['attention_macs']:.2f}x")
print(f"4x scaling check (256->512): MACs {df_ex.iloc[3]['total_macs']/df_ex.iloc[2]['total_macs']:.2f}x, "
      f"Attn MACs {df_ex.iloc[3]['attention_macs']/df_ex.iloc[2]['attention_macs']:.2f}x")

print(f"\nSaved: {OUTPUT_FILE}, {OUTPUT_PDF}")
