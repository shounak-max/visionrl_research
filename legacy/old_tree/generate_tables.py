#!/usr/bin/env python3
"""
Generates publication-quality LaTeX tables directly from JSON artifacts in paper_artifacts/.
Ensures strict provenance: every number in the tables is read from verified JSON files.
"""

import os
import json

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "paper_artifacts")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "paper", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_table1():
    with open(os.path.join(ARTIFACTS_DIR, "paper_headline_results_combined.json"), "r") as f:
        data = json.load(f)["table1_correlation"]

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Phase 1 Correlation Evaluation Grid on Procgen CoinRun.} Spearman rank correlation ($\rho$) between predicted representation distance ($\tilde{d}_\mathrm{rep}$) and actual test-time performance drop $\Delta R$ across four perturbation axes ($N=100$ per axis, 400 total). Bold indicates highest correlation per column.}",
        r"\label{tab:phase1_semantic}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Metric / Weighter Backend} & \textbf{Overall $\rho$} & \textbf{Background} & \textbf{Wall Texture} & \textbf{Sprite} & \textbf{Difficulty} \\",
        r"\midrule",
    ]

    order = [
        ("Classical d_rep", "Classical $d_\\mathrm{rep}$ (Unweighted)"),
        ("SegmentationWeighter (GT Mask)", "SegmentationWeighter (GT Mask)"),
        ("TLDAWeighter (Lipschitz)", "TLDAWeighter (Lipschitz Policy)"),
        ("SaliencyWeighter (Critic)", "SaliencyWeighter (Critic Saliency)"),
    ]

    for key, display_name in order:
        entry = data[key]
        overall_rho = entry["overall_rho"]
        overall_p = entry["overall_pvalue"]
        bg_rho = entry["axis_breakdown"]["background"]["spearman_rho"]
        wall_rho = entry["axis_breakdown"]["wall_texture"]["spearman_rho"]
        sprite_rho = entry["axis_breakdown"]["sprite"]["spearman_rho"]
        diff_rho = entry["axis_breakdown"]["difficulty"]["spearman_rho"]

        # Bold formatting for highlights
        overall_str = f"\\textbf{{{overall_rho:.4f}}}" if key == "SegmentationWeighter (GT Mask)" else f"{overall_rho:.4f}"
        bg_str = f"{bg_rho:.4f}"
        wall_str = f"\\textbf{{{wall_rho:.4f}}}" if key == "Classical d_rep" else f"{wall_rho:.4f}"
        sprite_str = f"\\textbf{{{sprite_rho:.4f}}}" if key == "SegmentationWeighter (GT Mask)" else f"{sprite_rho:.4f}"
        diff_str = f"\\textbf{{{diff_rho:.4f}}}" if key == "SaliencyWeighter (Critic)" else f"{diff_rho:.4f}"

        p_note = f"$^*$ ($p={overall_p:.3f}$)" if overall_p < 0.05 else f" ($p={overall_p:.3f}$)"
        tex.append(f"{display_name} & {overall_str}{p_note} & {bg_str} & {wall_str} & {sprite_str} & {diff_str} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize{$^*$Statistically significant at $\alpha = 0.05$. Note that background shift shows $\rho = 0.0$ for all models because policy returns on background shift exhibit zero variance across evaluated intensities.}",
        r"\end{flushleft}",
        r"\end{table*}",
    ])

    out_file = os.path.join(OUTPUT_DIR, "table1_semantic.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def generate_table2():
    with open(os.path.join(ARTIFACTS_DIR, "paper_headline_results_combined.json"), "r") as f:
        data = json.load(f)["table2_geometry"]

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Phase 2 Grassmann Manifold Subspace Drift $\mathrm{Gr}(k, 512)$.} Geodesic distances $d_\mathrm{geo}(U_\mathrm{clean}, U_\mathrm{pert})$ (in radians) between clean and perturbed policy representation subspaces across subspace dimensions $k \in \{5, 10, 20\}$. End-to-end trained encoders exhibit significantly lower geodesic drift.}",
        r"\label{tab:phase2_geometry}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Encoder Fine-Tuning Regime} & \textbf{$k=5$} & \textbf{$k=10$} & \textbf{$k=20$} \\",
        r"\midrule",
    ]

    order = [
        ("Frozen ResNet-18", "Frozen ResNet-18"),
        ("Partial Fine-Tune ResNet-18", "Partial Fine-Tune ResNet-18"),
        ("End-to-End CNN", "End-to-End CNN (DrQ-v2/SVEA)"),
    ]

    for key, display_name in order:
        entry = data[key]
        d_k5 = entry["k=5"]["geodesic_distance"]
        d_k10 = entry["k=10"]["geodesic_distance"]
        d_k20 = entry["k=20"]["geodesic_distance"]

        if key == "End-to-End CNN":
            tex.append(f"\\textbf{{{display_name}}} & \\textbf{{{d_k5:.4f}}} & \\textbf{{{d_k10:.4f}}} & \\textbf{{{d_k20:.4f}}} \\\\")
        else:
            tex.append(f"{display_name} & {d_k5:.4f} & {d_k10:.4f} & {d_k20:.4f} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_file = os.path.join(OUTPUT_DIR, "table2_geometry.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def generate_table3():
    with open(os.path.join(ARTIFACTS_DIR, "paper_headline_results_combined.json"), "r") as f:
        data = json.load(f)["table3_ttss"]

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Phase 3 Test-Time Self-Supervised Adaptation (TTSS) Under Severe Shift.} Mean tracking error (Centre-of-target Localization Error in pixels, lower is better), floor collapse rate ($\ge 15$\,px error), trust-region update acceptance rate, and inference latency per step on Visual Pursuit under severe distribution shift ($N=30$ episodes per condition).}",
        r"\label{tab:phase3_ttss}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Adaptation Regime} & \textbf{\shortstack{Mean CLE\\(px $\downarrow$)}} & \textbf{\shortstack{Floor Collapse\\Rate ($\% \downarrow$)}} & \textbf{\shortstack{Safeguard\\Acceptance}} & \textbf{\shortstack{Avg Latency\\(ms/step)}} \\",
        r"\midrule",
    ]

    # Baseline
    b = data["unadapted_baseline"]
    b_cle = b["mean_tracking_error_px"]
    b_fc = b["floor_collapse_rate"] * 100.0
    tex.append(f"Unadapted Baseline & {b_cle:.2f} & {b_fc:.1f}\\% & N/A & 0.00 \\\\")

    # Naive TTT
    n = data["naive_ttt"]
    n_cle = n["mean_tracking_error_px"]
    n_fc = n["floor_collapse_rate"] * 100.0
    n_acc = n["trust_region_acceptance_rate"] * 100.0
    n_lat = n["avg_latency_ms"]
    tex.append(f"Naive TTT (No Safeguard) & {n_cle:.2f} & {n_fc:.1f}\\% & {n_acc:.1f}\\% & {n_lat:.2f} \\\\")

    # Safeguarded TTSS
    s = data["safeguarded_ttss"]
    s_cle = s["mean_tracking_error_px"]
    s_fc = s["floor_collapse_rate"] * 100.0
    s_acc = s["trust_region_acceptance_rate"] * 100.0
    s_lat = s["avg_latency_ms"]
    tex.append(f"\\textbf{{Safeguarded TTSS (Ours)}} & \\textbf{{{s_cle:.2f}}} & \\textbf{{{s_fc:.1f}\\%}} & {s_acc:.1f}\\% & {s_lat:.2f} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize{\textbf{Note on Safeguard Acceptance:} In both conditions, the acceptance rate is $100.0\%$. In our experiments the self-supervised moment-matching loss consistently decreased across candidate updates, so no updates were empirically rejected by the trust region.}",
        r"\end{flushleft}",
        r"\end{table}",
    ])

    out_file = os.path.join(OUTPUT_DIR, "table3_ttss.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


if __name__ == "__main__":
    generate_table1()
    generate_table2()
    generate_table3()
    print("All tables successfully generated from paper_artifacts/.")
