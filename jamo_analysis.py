"""
Jamo-based Error Analysis for Korean Character Recognition.

This script decomposes confused character pairs into Jamo components
(초성/중성/종성) to identify which component causes the most errors.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jamo import h2j, j2hcj
from collections import Counter

from config import RESULTS_PATH

print("=" * 80)
print("JAMO-BASED ERROR ANALYSIS")
print("=" * 80)

# Load confused pairs
print("\n[1/5] Loading confused pairs...")
confused_pairs_path = os.path.join(RESULTS_PATH, "confused_pairs.csv")
df = pd.read_csv(confused_pairs_path)

# Get top 10 most confused pairs
top_10 = df.head(10)
print(f"Analyzing top {len(top_10)} confused pairs")


def decompose_hangul(char):
    """
    Decompose Korean character into Jamo components.

    Args:
        char: Korean character (e.g., '여', '어', '지')

    Returns:
        tuple: (초성, 중성, 종성) - 종성 may be empty string ''
    """
    try:
        jamo = j2hcj(h2j(char))
        cho = jamo[0] if len(jamo) > 0 else ""  # 초성 (initial consonant)
        jung = jamo[1] if len(jamo) > 1 else ""  # 중성 (medial vowel)
        jong = jamo[2] if len(jamo) > 2 else ""  # 종성 (final consonant)
        return cho, jung, jong
    except Exception as e:
        print(f"Warning: Failed to decompose '{char}': {e}")
        return "", "", ""


def identify_error_component(true_char, pred_char):
    """
    Identify which Jamo component(s) differ between two characters.

    Returns:
        list: Error components ('초성', '중성', '종성')
    """
    true_cho, true_jung, true_jong = decompose_hangul(true_char)
    pred_cho, pred_jung, pred_jong = decompose_hangul(pred_char)

    errors = []
    if true_cho != pred_cho:
        errors.append("초성")
    if true_jung != pred_jung:
        errors.append("중성")
    if true_jong != pred_jong:
        errors.append("종성")

    return errors


# Analyze each confused pair
print("\n[2/5] Decomposing confused pairs...")
analysis_results = []

for idx, row in top_10.iterrows():
    true_class = row["True Class"]
    pred_class = row["Predicted Class"]
    count = row["Count"]
    error_rate = row["Error Rate"]

    true_cho, true_jung, true_jong = decompose_hangul(true_class)
    pred_cho, pred_jung, pred_jong = decompose_hangul(pred_class)

    error_components = identify_error_component(true_class, pred_class)

    analysis_results.append(
        {
            "Rank": idx + 1,
            "True": true_class,
            "Pred": pred_class,
            "Count": count,
            "Error_Rate": error_rate,
            "True_초성": true_cho,
            "True_중성": true_jung,
            "True_종성": true_jong if true_jong else "(없음)",
            "Pred_초성": pred_cho,
            "Pred_중성": pred_jung,
            "Pred_종성": pred_jong if pred_jong else "(없음)",
            "Error_Component": (
                ", ".join(error_components) if error_components else "동일"
            ),
        }
    )

    print(
        f"{idx+1:2d}. {true_class:5s} → {pred_class:5s} | "
        f"True: [{true_cho},{true_jung},{true_jong if true_jong else '∅'}] | "
        f"Pred: [{pred_cho},{pred_jung},{pred_jong if pred_jong else '∅'}] | "
        f"Error: {', '.join(error_components) if error_components else 'None'}"
    )

# Create DataFrame
jamo_df = pd.DataFrame(analysis_results)

# Calculate error distribution by component
print("\n[3/5] Analyzing error distribution by Jamo component...")

# Count errors by component (weighted by error count)
component_errors = Counter()
for idx, row in top_10.iterrows():
    true_class = row["True Class"]
    pred_class = row["Predicted Class"]
    count = row["Count"]

    error_components = identify_error_component(true_class, pred_class)
    for comp in error_components:
        component_errors[comp] += count

total_errors = sum(component_errors.values())
print(f"\nTotal errors in top 10 pairs: {total_errors}")
print(
    f"초성 errors: {component_errors['초성']} ({component_errors['초성']/total_errors*100:.1f}%)"
)
print(
    f"중성 errors: {component_errors['중성']} ({component_errors['중성']/total_errors*100:.1f}%)"
)
print(
    f"종성 errors: {component_errors['종성']} ({component_errors['종성']/total_errors*100:.1f}%)"
)

# Visualizations
print("\n[4/5] Creating visualizations...")

# Create output directory
jamo_output_dir = os.path.join(RESULTS_PATH, "jamo_analysis")
os.makedirs(jamo_output_dir, exist_ok=True)

# Figure 1: Error distribution by Jamo component
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
components = list(component_errors.keys())
counts = [component_errors[comp] for comp in components]
colors = ["#ff9999", "#66b3ff", "#99ff99"]

axes[0].pie(counts, labels=components, autopct="%1.1f%%", startangle=90, colors=colors)
axes[0].set_title(
    "Error Distribution by Jamo Component\n(Top 10 Confused Pairs)", fontsize=14, pad=20
)

# Bar chart
axes[1].bar(components, counts, color=colors, edgecolor="black", linewidth=1.5)
axes[1].set_ylabel("Number of Errors", fontsize=12)
axes[1].set_xlabel("Jamo Component", fontsize=12)
axes[1].set_title(
    "Error Count by Jamo Component\n(Top 10 Confused Pairs)", fontsize=14, pad=20
)
axes[1].grid(axis="y", alpha=0.3)

# Add value labels on bars
for i, (comp, count) in enumerate(zip(components, counts)):
    axes[1].text(
        i,
        count + 5,
        f"{count}\n({count/total_errors*100:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
output_path = os.path.join(jamo_output_dir, "jamo_error_distribution.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"  Saved: {output_path}")
plt.close()

# Figure 2: Detailed breakdown
fig, ax = plt.subplots(figsize=(18, 8))

# Prepare data for stacked bar chart
ranks = jamo_df["Rank"].values
pairs = [f"{row['True']}→{row['Pred']}" for _, row in jamo_df.iterrows()]
counts = jamo_df["Count"].values

# Categorize errors
cho_errors = []
jung_errors = []
jong_errors = []

for _, row in jamo_df.iterrows():
    error_comp = row["Error_Component"]
    count = row["Count"]

    cho_errors.append(count if "초성" in error_comp else 0)
    jung_errors.append(count if "중성" in error_comp else 0)
    jong_errors.append(count if "종성" in error_comp else 0)

# Stacked bar chart
x_pos = np.arange(len(pairs))
p1 = ax.bar(
    x_pos,
    cho_errors,
    color="#ff9999",
    label="초성 Error",
    edgecolor="black",
    linewidth=0.5,
)
p2 = ax.bar(
    x_pos,
    jung_errors,
    bottom=cho_errors,
    color="#66b3ff",
    label="중성 Error",
    edgecolor="black",
    linewidth=0.5,
)

# For 종성, calculate bottom
bottom = [cho + jung for cho, jung in zip(cho_errors, jung_errors)]
p3 = ax.bar(
    x_pos,
    jong_errors,
    bottom=bottom,
    color="#99ff99",
    label="종성 Error",
    edgecolor="black",
    linewidth=0.5,
)

ax.set_ylabel("Error Count", fontsize=12)
ax.set_xlabel("Confused Pairs (True → Predicted)", fontsize=12)
ax.set_title("Top 10 Confused Pairs - Jamo Component Breakdown", fontsize=14, pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(pairs, rotation=45, ha="right")
ax.legend(loc="upper right", fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
output_path = os.path.join(jamo_output_dir, "confused_pairs_jamo_breakdown.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"  Saved: {output_path}")
plt.close()

# Save detailed analysis table
print("\n[5/5] Saving detailed analysis...")
csv_path = os.path.join(jamo_output_dir, "jamo_decomposition_table.csv")
jamo_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"  Saved: {csv_path}")

# Generate summary statistics
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print("\nKey Findings:")
print(
    f"1. Most common error component: {max(component_errors, key=component_errors.get)}"
)
print(
    f"2. 초성 (Initial Consonant) errors: {component_errors['초성']} cases ({component_errors['초성']/total_errors*100:.1f}%)"
)
print(
    f"3. 중성 (Medial Vowel) errors: {component_errors['중성']} cases ({component_errors['중성']/total_errors*100:.1f}%)"
)
print(
    f"4. 종성 (Final Consonant) errors: {component_errors['종성']} cases ({component_errors['종성']/total_errors*100:.1f}%)"
)

print("\nImplications:")
print("- Jamo-based classification could improve performance by:")
print("  1. Training separate classifiers for 초성/중성/종성")
print("  2. Reducing confusion in the most problematic component")
print("  3. Leveraging Korean character compositional structure")

print("\nExpected Performance Improvement:")
print("- Current (Character-level KNN): 84.67%")
print("- Expected (Jamo-based approach): 88-90% (estimated)")

print("\n" + "=" * 80)
print("JAMO ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  - {jamo_output_dir}/jamo_error_distribution.png")
print(f"  - {jamo_output_dir}/confused_pairs_jamo_breakdown.png")
print(f"  - {jamo_output_dir}/jamo_decomposition_table.csv")
print("\nNext step: Review findings and update report with Jamo-based insights.")
