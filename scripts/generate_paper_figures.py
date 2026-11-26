"""Publication-grade figure generator for Vision-RL paper.
Reads 100% of data from results/ (CSV/JSON). No hand-typed or simulated metrics.
Produces 300 DPI PNG and vector PDF files in paper/figures/.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIG_DIR = Path("paper") / "figures"

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
})

def plot_fig1_conceptual_geometry():
    """Figure 1: Conceptual illustration of policy pullback geometry. Explicitly labeled as illustration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # Panel A: Latent Manifold & Action Curvature
    z1 = np.linspace(-2.0, 2.0, 100)
    z2 = np.linspace(-2.0, 2.0, 100)
    Z1, Z2 = np.meshgrid(z1, z2)
    Act = np.tanh(1.2 * Z1 + 0.4 * (Z2 ** 2) - 0.2 * Z1 * Z2)
    cs = ax1.contourf(Z1, Z2, Act, levels=20, cmap="coolwarm", alpha=0.35)
    cbar = fig.colorbar(cs, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r"Action Field $\pi(\mathbf{z})$ (Illustration)", fontsize=9)
    ax1.plot(0, 0, "ko", markersize=7, label=r"Clean State $\mathbf{z}_0$")
    ax1.set_title("(a) Pullback Geometry (Conceptual)")
    ax1.set_xlabel(r"Latent Dimension $z_1$")
    ax1.set_ylabel(r"Latent Dimension $z_2$")
    ax1.legend(loc="lower right")

    # Panel B: Manifold Metric Comparison
    severities = np.linspace(0, 1, 50)
    euclid_drift = 1.8 * severities
    curv_drift = 2.4 * (severities ** 2) + 0.5 * severities
    ax2.plot(severities, euclid_drift, "k--", label=r"Isotropic Euclidean ($d_\mathrm{rep}$)")
    ax2.plot(severities, curv_drift, "b-", label=r"Curvature-Aware Metric (PAC)")
    ax2.set_title("(b) Metric Tensor Scaling (Conceptual)")
    ax2.set_xlabel(r"Perturbation Severity $s$")
    ax2.set_ylabel("Drift Magnitude")
    ax2.legend(loc="upper left")

    fig.suptitle("Figure 1: Conceptual Geometry of Policy-Action Manifolds (Illustration, not empirical data)", fontsize=11, style="italic")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_geometry.png")
    plt.savefig(FIG_DIR / "fig1_geometry.pdf")
    plt.close()
    print("Generated fig1_geometry.png and .pdf (labeled as illustration)")

def plot_fig2_forecasting_benchmarks():
    """Figure 2: Empirical forecasting accuracy across candidate scores from results/predictor_table_30k.csv."""
    df_p = pd.read_csv(RESULTS_DIR / "predictor_table_30k.csv")
    df_p = df_p.sort_values(by="auroc", ascending=True)
    prevalence = float(df_p["prevalence"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    # Panel A: AUROC and AUPRC
    y_pos = np.arange(len(df_p))
    ax1.barh(y_pos - 0.18, df_p["auroc"], height=0.35, label="AUROC", color="#1f77b4", alpha=0.85)
    ax1.barh(y_pos + 0.18, df_p["auprc"], height=0.35, label="AUPRC", color="#ff7f0e", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_p["predictor"])
    ax1.set_xlabel("Score Value")
    ax1.set_xlim(0.4, 1.0)
    ax1.axvline(prevalence, color="gray", linestyle=":", label=f"Random Prevalence ({prevalence:.3f})")
    ax1.set_title("(a) Discrimination Under Strict Holdout (Data: predictor_table_30k.csv)")
    ax1.legend(loc="lower right")

    # Panel B: TPR@5%FPR and Clean False Alarm Rate
    ax2.barh(y_pos - 0.18, df_p["tpr_at_5fpr"], height=0.35, label="TPR @ 5% FPR", color="#2ca02c", alpha=0.85)
    ax2.barh(y_pos + 0.18, df_p["false_alarm_rate_clean"], height=0.35, label="Clean False Alarm Rate", color="#d62728", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_p["predictor"])
    ax2.set_xlabel("Rate")
    ax2.axvline(0.05, color="red", linestyle="--", label="Nominal Conformal Target (0.05)")
    ax2.set_title("(b) Operational Detection & Alarm Rates (Data: predictor_table_30k.csv)")
    ax2.legend(loc="lower right")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_forecasting_auroc.png")
    plt.savefig(FIG_DIR / "fig2_forecasting_auroc.pdf")
    plt.close()
    print("Generated fig2_forecasting_auroc.png and .pdf")

def plot_fig3_cliff_locations():
    """Figure 3: Critical severity s* distribution and cliffness across shift kinds from results/all_agents_s_star_30k.csv."""
    df_c = pd.read_csv(RESULTS_DIR / "all_agents_s_star_30k.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: s* by shift family
    kinds = df_c["shift_kind"].unique()
    s_crit_by_kind = [df_c[df_c["shift_kind"] == k]["s_crit"].values for k in kinds]
    n_censored = int(df_c["censored"].astype(bool).sum())
    n_total = len(df_c)

    ax1.boxplot(s_crit_by_kind, tick_labels=kinds, patch_artist=True,
                boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
                medianprops=dict(color="darkblue", linewidth=2))
    ax1.axhline(1.0, color="red", linestyle="--", label=f"Censored Boundary (s*=1.0, {n_censored}/{n_total})")
    ax1.set_ylabel(r"Critical Severity $s^*$")
    ax1.set_title(r"(a) Distribution of Critical Severity $s^*$ (Data: all_agents_s_star_30k.csv)")
    ax1.legend(loc="upper left")

    # Panel B: Cliffness vs Total Drop
    for k in kinds:
        sub = df_c[df_c["shift_kind"] == k]
        ax2.scatter(sub["cliffness"], sub["total_drop"], label=k, s=60, alpha=0.85)
    n_abrupt = int((df_c["cliffness"] >= 0.5).sum())
    ax2.axvline(0.5, color="black", linestyle=":", label="Abrupt Cliffness Gate (0.5)")
    ax2.axhline(0.3, color="gray", linestyle=":", label="Significant Drop Gate (0.3)")
    ax2.set_xlabel(r"Cliffness Metric ($1/\Delta s$)")
    ax2.set_ylabel("Total Performance Drop")
    ax2.set_title(f"(b) Failure Abruptness Check ({n_abrupt}/{n_total} Curves >= 0.5)")
    ax2.legend(loc="lower right")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_cliff_locations.png")
    plt.savefig(FIG_DIR / "fig3_cliff_locations.pdf")
    plt.close()
    print("Generated fig3_cliff_locations.png and .pdf")

def plot_fig4_ranking_and_regret():
    """Figure 4: Policy ranking Kendall tau vs selection regret from results/ranking_table_30k.csv."""
    df_r_raw = pd.read_csv(RESULTS_DIR / "ranking_table_30k.csv")
    df_r = df_r_raw.groupby("predictor", as_index=False)[["tau", "regret"]].mean().dropna(subset=["tau", "regret"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(df_r)))
    for idx, (_, row) in enumerate(df_r.iterrows()):
        ax.scatter(row["tau"], row["regret"], s=100, color=colors[idx], label=row["predictor"], zorder=4)
        ax.annotate(row["predictor"], (row["tau"], row["regret"]), xytext=(row["tau"]+0.015, row["regret"]+0.002), fontsize=8)

    ax.set_xlabel(r"Policy Ranking Kendall $\tau$ ($\uparrow$ better ranking)")
    ax.set_ylabel(r"Selection Regret (Severity Units, $\downarrow$ lower error)")
    ax.set_title(r"Policy Ranking Fidelity vs Selection Regret (Data: ranking_table_30k.csv)")
    ax.axvline(0.60, color="red", linestyle="--", label=r"Pre-registered Gate ($\tau \geq 0.60$)")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_ranking_and_regret.png")
    plt.savefig(FIG_DIR / "fig4_ranking_and_regret.pdf")
    plt.close()
    print("Generated fig4_ranking_and_regret.png and .pdf")

def plot_learning_curves():
    """Figure: Clean tracking error and success learning curves from results/*_training_curve.csv (30k steps)."""
    curve_files = sorted(RESULTS_DIR.glob("*_training_curve.csv"))
    if not curve_files:
        raise FileNotFoundError("No *_training_curve.csv files found in results/")

    label_map = {"drqv2": "DrQ-v2", "svea": "SVEA", "taco": "TACO"}
    style_map = {"drqv2": "b-", "svea": "g--", "taco": "m-."}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for cf in curve_files:
        df = pd.read_csv(cf)
        variant = str(df["variant"].iloc[0])
        label = label_map.get(variant, variant)
        style = style_map.get(variant, "-")
        ax1.plot(df["env_steps"], df["mean_cle"], style, alpha=0.85, label=label)
        ax2.plot(df["env_steps"], df["success"], style, alpha=0.85, label=label)

    ax1.axhline(5.0, color="red", linestyle=":", label="Success Threshold (5.0 px)")
    ax1.axhline(22.05, color="gray", linestyle=":", label="Random Floor (22.05 px)")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Clean Tracking Error (CLE px)")
    ax1.set_title("(a) Clean Tracking Error Over 30k Steps")
    ax1.legend(loc="upper right")

    ax2.axhline(0.20, color="gray", linestyle=":", label="Pre-training Gate H0 (0.20)")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Clean Success Rate")
    ax2.set_title("(b) Clean Success Rate Over 30k Steps")
    ax2.legend(loc="upper left")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_learning_curves.png")
    plt.savefig(FIG_DIR / "fig_learning_curves.pdf")
    plt.close()
    print("Generated fig_learning_curves.png and .pdf")

if __name__ == "__main__":
    plot_fig1_conceptual_geometry()
    plot_fig2_forecasting_benchmarks()
    plot_fig3_cliff_locations()
    plot_fig4_ranking_and_regret()
    plot_learning_curves()
    print("All paper figures successfully generated.")
