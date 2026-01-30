"""
Kaplan-Meier Survival Plot Examples using bioviz-kit.

This script demonstrates various KM plot configurations including:
1. Basic KM plot with risk table
2. Multi-arm comparison with custom colors
3. Biomarker stratification analysis
4. Publication-ready styling options

Run inside your project venv where `bioviz` deps are installed:
    python km_survival_example.py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioviz.configs import KMPlotConfig
from bioviz.plots import KMPlotter

# %%
# =============================================================================
# Generate example survival data
# =============================================================================
np.random.seed(42)
n_patients = 120


def generate_survival_data(n: int, arms: list, hazard_ratios: dict) -> pd.DataFrame:
    """Generate simulated survival data with specified hazard ratios."""
    records = []
    for arm in arms:
        n_arm = n // len(arms)
        hr = hazard_ratios.get(arm, 1.0)
        # Exponential survival times scaled by hazard ratio
        times = np.random.exponential(12 / hr, n_arm)
        # Random censoring (~30%)
        events = np.random.choice([0, 1], n_arm, p=[0.3, 0.7])
        for i, (t, e) in enumerate(zip(times, events, strict=True)):
            records.append(
                {
                    "USUBJID": f"{arm[:3].upper()}-{i + 1:03d}",
                    "ARM": arm,
                    "TIME": min(t, 24),  # Max follow-up 24 months
                    "EVENT": e if t < 24 else 0,  # Censor at max follow-up
                }
            )
    return pd.DataFrame(records)


# Two-arm trial data
df_two_arm = generate_survival_data(
    n=100,
    arms=["Treatment", "Control"],
    hazard_ratios={"Treatment": 0.6, "Control": 1.0},
)

# Multi-arm data (3 treatment groups)
df_multi_arm = generate_survival_data(
    n=150,
    arms=["High Dose", "Low Dose", "Placebo"],
    hazard_ratios={"High Dose": 0.5, "Low Dose": 0.75, "Placebo": 1.0},
)

# Biomarker-stratified data
df_biomarker = generate_survival_data(
    n=120,
    arms=["Biomarker+", "Biomarker-"],
    hazard_ratios={"Biomarker+": 0.4, "Biomarker-": 1.0},
)

print(f"Two-arm data: {len(df_two_arm)} patients")
print(f"Multi-arm data: {len(df_multi_arm)} patients")
print(f"Biomarker data: {len(df_biomarker)} patients")

# %%
# =============================================================================
# Example 1: Basic Two-Arm KM Plot with Risk Table
# =============================================================================
print("\n" + "=" * 60)
print("Example 1: Basic Two-Arm KM Plot")
print("=" * 60)

config_basic = KMPlotConfig(
    time_col="TIME",
    event_col="EVENT",
    group_col="ARM",
    title="Overall Survival: Treatment vs Control",
    xlabel="Time (months)",
    ylabel="Survival Probability",
    show_risktable=True,
    show_pvalue=True,
    show_ci=True,
    color_dict={
        "Treatment": "#009E73",  # Green
        "Control": "#D55E00",  # Orange
    },
)

plotter = KMPlotter(df_two_arm, config_basic)
fig, ax, pval = plotter.plot()
print(f"Log-rank p-value: {pval:.4f}")
plt.show()

# %%
# =============================================================================
# Example 2: Multi-Arm Dose Comparison
# =============================================================================
print("\n" + "=" * 60)
print("Example 2: Multi-Arm Dose Comparison")
print("=" * 60)

config_multi = KMPlotConfig(
    time_col="TIME",
    event_col="EVENT",
    group_col="ARM",
    title="Progression-Free Survival by Dose Level\n",
    xlabel="Time (months)",
    ylabel="PFS Probability",
    show_risktable=True,
    show_pvalue=True,
    pval_loc="top_right",
    legend_loc="right",
    legend_frameon=False,
    color_dict={
        "High Dose": "#0072B2",  # Blue
        "Low Dose": "#56B4E9",  # Light blue
        "Placebo": "#999999",  # Gray
    },
    figsize=(11, 7),
    linewidth=2.5,
)

plotter_multi = KMPlotter(df_multi_arm, config_multi)
fig_multi, ax_multi, pval_multi = plotter_multi.plot()
print(f"Log-rank p-value: {pval_multi:.4f}")
plt.show()

# %%
# =============================================================================
# Example 3: Biomarker Stratification Analysis
# =============================================================================
print("\n" + "=" * 60)
print("Example 3: Biomarker Stratification")
print("=" * 60)

config_biomarker = KMPlotConfig(
    time_col="TIME",
    event_col="EVENT",
    group_col="ARM",
    title="Overall Survival by Biomarker Status\n(Exploratory Analysis)",
    xlabel="Time (months)",
    ylabel="OS Probability",
    show_risktable=True,
    risktable_fontsize=14,
    show_pvalue=True,
    pvalue_fontsize=14,
    show_ci=True,
    ci_alpha=0.15,
    legend_loc="bottom",
    legend_show_n=True,  # Show (n=XX) in legend
    color_dict={
        "Biomarker+": "#E69F00",  # Orange/gold
        "Biomarker-": "#56B4E9",  # Light blue
    },
    figsize=(10, 6),
)

plotter_bio = KMPlotter(df_biomarker, config_biomarker)
fig_bio, ax_bio, pval_bio = plotter_bio.plot()
print(f"Log-rank p-value: {pval_bio:.4f}")
plt.show()

# %%
# =============================================================================
# Example 4: Publication-Ready Styling
# =============================================================================
print("\n" + "=" * 60)
print("Example 4: Publication-Ready Styling")
print("=" * 60)

config_pub = KMPlotConfig(
    time_col="TIME",
    event_col="EVENT",
    group_col="ARM",
    title="",  # No title for publication figure
    xlabel="Time from Randomization (months)",
    ylabel="Probability of Survival",
    show_risktable=True,
    risktable_fontsize=11,
    show_pvalue=True,
    pvalue_fontsize=11,
    pval_loc="bottom_left",
    show_ci=True,
    ci_style="fill",
    ci_alpha=0.2,
    legend_loc="right",
    legend_fontsize=11,
    legend_frameon=False,
    label_fontsize=12,
    title_fontsize=14,
    linewidth=2.0,
    censor_markersize=8,
    censor_markeredgewidth=1.5,
    xlim=(0, 24),
    ylim=(0, 1.0),
    xticks=[0, 6, 12, 18, 24],
    color_dict={
        "Treatment": "#2E86AB",
        "Control": "#A23B72",
    },
    figsize=(8, 5),
)

plotter_pub = KMPlotter(df_two_arm, config_pub)
fig_pub, ax_pub, pval_pub = plotter_pub.plot()

# Save publication-quality figure
fig_pub.savefig("km_publication.pdf", dpi=300)
print("Saved: km_publication.pdf")
plt.show()

# %%
# =============================================================================
# Example 5: Saving Multiple Formats
# =============================================================================
print("\n" + "=" * 60)
print("Example 5: Saving Multiple Formats")
print("=" * 60)

# Create a simple plot and save in multiple formats
config_save = KMPlotConfig(
    time_col="TIME",
    event_col="EVENT",
    group_col="ARM",
    title="Survival Analysis",
    show_risktable=True,
)

plotter_save = KMPlotter(df_two_arm, config_save)

# Save directly via output_path parameter
fig_pdf, _, _ = plotter_save.plot(output_path="km_example.pdf")
print("Saved: km_example.pdf")

# Or save manually with custom settings
fig_png, _, _ = plotter_save.plot()
fig_png.savefig("km_example.png", dpi=150, facecolor="white")
print("Saved: km_example.png")

plt.close("all")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
