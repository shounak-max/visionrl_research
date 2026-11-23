"""Export authoritative experiment results from the executed notebook into results/ CSV and JSON."""

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Parse CLIFF TABLE from notebook outputs
cliff_data = """agent_id       kind  clean_success  s_crit  censored  cliffness  total_drop
drqv2_s0 background           0.85   0.194     False      0.471        0.85
drqv2_s0 color_gain           0.85   0.788     False      0.333        0.60
drqv2_s0 distractor           0.85   0.206     False      0.471        0.85
drqv2_s0      noise           0.85   0.077     False      0.647        0.85
drqv2_s0  viewpoint           0.85   1.000      True      0.286        0.35
drqv2_s1 background           0.75   0.210     False      0.400        0.75
drqv2_s1 color_gain           0.75   0.783     False      0.444        0.45
drqv2_s1 distractor           0.75   0.207     False      0.467        0.75
drqv2_s1      noise           0.75   0.094     False      0.533        0.75
drqv2_s1  viewpoint           0.75   1.000      True      0.250        0.20
drqv2_s2 background           0.80   0.162     False      0.500        0.80
drqv2_s2 color_gain           0.80   0.775     False      0.364        0.55
drqv2_s2 distractor           0.80   0.173     False      0.688        0.80
drqv2_s2      noise           0.80   0.062     False      0.812        0.80
drqv2_s2  viewpoint           0.80   1.000      True      0.200        0.25
 svea_s0 background           0.70   0.500     False      0.250        0.60
 svea_s0 color_gain           0.70   1.000      True      0.500        0.20
 svea_s0 distractor           0.70   0.188     False      0.571        0.70
 svea_s0      noise           0.70   0.136     False      0.786        0.70
 svea_s0  viewpoint           0.70   1.000      True      0.500        0.20
 svea_s1 background           0.80   0.600     False      0.231        0.65
 svea_s1 color_gain           0.80   1.000      True      0.750        0.20
 svea_s1 distractor           0.80   0.189     False      0.562        0.80
 svea_s1      noise           0.80   0.138     False      0.812        0.80
 svea_s1  viewpoint           0.80   1.000      True      0.429        0.35
 svea_s2 background           0.80   0.500     False      0.214        0.70
 svea_s2 color_gain           0.80   1.000      True      0.429        0.35
 svea_s2 distractor           0.80   0.180     False      0.625        0.80
 svea_s2      noise           0.80   0.053     False      0.938        0.80
 svea_s2  viewpoint           0.80   1.000      True      0.500        0.30"""

rows = []
for line in cliff_data.strip().split("\n")[1:]:
    parts = line.split()
    rows.append({
        "agent_id": parts[0],
        "kind": parts[1],
        "clean_success": float(parts[2]),
        "s_crit": float(parts[3]),
        "censored": parts[4] == "True",
        "cliffness": float(parts[5]),
        "total_drop": float(parts[6]),
    })
cliff_df = pd.DataFrame(rows)
cliff_df.to_csv(RESULTS_DIR / "cliff_table.csv", index=False)
cliff_df.to_json(RESULTS_DIR / "cliff_table.json", orient="records", indent=2)
print("Saved cliff_table.csv and cliff_table.json (30 curves, 9 censored)")

# 2. Parse PREDICTOR TABLE (strict protocol)
predictor_data = """predictor            n  prevalence  auroc  auprc  auprc_lift  tpr_at_5fpr  false_alarm_rate_clean  alarm_rate_on_failures status
knn_policy      3060       0.592  0.912  0.949       1.601        0.726                   0.117                   0.803     ok
maha_policy     3060       0.592  0.825  0.880       1.485        0.442                   0.017                   0.469     ok
cent_policy     3060       0.592  0.638  0.702       1.184        0.120                   0.033                   0.068     ok
sem_knn_policy  3060       0.592  0.792  0.874       1.475        0.509                   0.000                   0.517     ok
knn_random      3060       0.592  0.806  0.833       1.407        0.207                   0.017                   0.892     ok
knn_pixel       3060       0.592  0.834  0.854       1.442        0.260                   0.017                   0.882     ok
act_disagree    3060       0.592  0.884  0.928       1.566        0.656                   0.000                   0.989     ok
q_drop          3060       0.592  0.833  0.911       1.538        0.735                   0.000                   0.833     ok
critic_gap      3060       0.592  0.790  0.855       1.443        0.470                   0.017                   0.502     ok"""

p_rows = []
for line in predictor_data.strip().split("\n")[1:]:
    parts = line.split()
    p_rows.append({
        "predictor": parts[0],
        "n": int(parts[1]),
        "prevalence": float(parts[2]),
        "auroc": float(parts[3]),
        "auprc": float(parts[4]),
        "auprc_lift": float(parts[5]),
        "tpr_at_5fpr": float(parts[6]),
        "false_alarm_rate_clean": float(parts[7]),
        "alarm_rate_on_failures": float(parts[8]),
        "status": parts[9]
    })
predictor_df = pd.DataFrame(p_rows)
predictor_df.to_csv(RESULTS_DIR / "predictor_table.csv", index=False)
predictor_df.to_json(RESULTS_DIR / "predictor_table.json", orient="records", indent=2)
print("Saved predictor_table.csv and predictor_table.json")

# 3. Parse RANKING TABLE
ranking_data = """predictor       tau  regret  pred_spread  true_spread
act_disagree  0.581   0.032        0.113        0.055
cent_policy   0.120   0.068        0.015        0.055
critic_gap    0.245   0.052        0.124        0.055
knn_pixel     0.375   0.032        0.008        0.055
knn_policy    0.347   0.028        0.118        0.055
knn_random    0.132   0.053        0.005        0.055
maha_policy   0.052   0.045        0.217        0.055
q_drop        0.513   0.023        0.098        0.055
sem_knn_policy 0.375  0.035        0.095        0.055"""

r_rows = []
for line in ranking_data.strip().split("\n")[1:]:
    parts = line.split()
    r_rows.append({
        "predictor": parts[0],
        "tau": float(parts[1]),
        "regret": float(parts[2]),
        "pred_spread": float(parts[3]),
        "true_spread": float(parts[4]),
    })
ranking_df = pd.DataFrame(r_rows)
ranking_df.to_csv(RESULTS_DIR / "ranking_table.csv", index=False)
ranking_df.to_json(RESULTS_DIR / "ranking_table.json", orient="records", indent=2)
print("Saved ranking_table.csv and ranking_table.json")

# 4. Parse FORECAST MAE TABLE
mae_data = """predictor      method  mean_abs_err  curve_mae  rank_corr   n
act_disagree   iso            0.148      0.177      0.765  30
act_disagree   rule           0.450      0.348      0.475  30
cent_policy    iso            0.473      0.388      0.502  30
cent_policy    rule           0.446      0.369      0.058  30
critic_gap     iso            0.096      0.245      0.498  30
critic_gap     rule           0.133      0.183      0.608  30
knn_pixel      iso            0.230      0.236      0.696  30
knn_pixel      rule           0.215      0.173      0.695  30
knn_policy     iso            0.083      0.197      0.639  30
knn_policy     rule           0.078      0.112      0.532  30
knn_random     iso            0.345      0.295      0.721  30
knn_random     rule           0.229      0.187      0.661  30
maha_policy    iso            0.171      0.280      0.520  30
maha_policy    rule           0.159      0.208      0.562  30
q_drop         iso            0.070      0.165      0.716  30
q_drop         rule           0.409      0.190      0.528  30
sem_knn_policy iso            0.241      0.266      0.568  30
sem_knn_policy rule           0.191      0.161      0.674  30"""

m_rows = []
for line in mae_data.strip().split("\n")[1:]:
    parts = line.split()
    m_rows.append({
        "predictor": parts[0],
        "method": parts[1],
        "mean_abs_err": float(parts[2]),
        "curve_mae": float(parts[3]),
        "rank_corr": float(parts[4]),
        "n": int(parts[5]),
    })
mae_df = pd.DataFrame(m_rows)
mae_df.to_csv(RESULTS_DIR / "forecast_mae_table.csv", index=False)
mae_df.to_json(RESULTS_DIR / "forecast_mae_table.json", orient="records", indent=2)
print("Saved forecast_mae_table.csv and forecast_mae_table.json")

# 5. Parse TTA PAIRED TABLE
tta_data = """kind  sev     contrast  n  d_mean_cle  ci_lo  ci_hi  success_a  success_b  p_success  collapse_a  collapse_b  p_collapse  accepted_mean_a  p_success_holm  p_collapse_holm
background  0.6  ttt_vs_none 40      -5.269 -7.809 -2.247      0.075      0.000       0.25       0.550       0.675       0.227           49.000             1.0              1.0
background  0.6 ttss_vs_none 40       0.072 -1.053  1.056      0.025      0.000       1.00       0.675       0.675       1.000            0.875             1.0              1.0
background  0.6  ttss_vs_ttt 40       5.341  2.441  7.996      0.025      0.075       0.50       0.675       0.550       0.227            0.875             1.0              1.0
background  0.8  ttt_vs_none 40      -2.798 -5.872  0.452      0.000      0.000       1.00       0.650       0.700       0.727           49.000             1.0              1.0
background  0.8 ttss_vs_none 40      -0.397 -1.806  0.974      0.000      0.000       1.00       0.700       0.700       1.000            1.075             1.0              1.0
background  0.8  ttss_vs_ttt 40       2.401 -0.744  5.097      0.000      0.000       1.00       0.700       0.650       0.688            1.075             1.0              1.0
background  1.0  ttt_vs_none 40      -2.782 -6.061  0.910      0.025      0.000       1.00       0.750       0.750       1.000           49.000             1.0              1.0
background  1.0 ttss_vs_none 40      -0.947 -3.082  0.862      0.000      0.000       1.00       0.700       0.750       0.500            1.225             1.0              1.0
background  1.0  ttss_vs_ttt 40       1.835 -1.585  5.136      0.000      0.025       1.00       0.700       0.750       0.791            1.225             1.0              1.0
noise  0.6  ttt_vs_none 40      -5.774 -9.593 -1.424      0.000      0.000       1.00       0.750       0.800       0.804           49.000             1.0              1.0
noise  0.6 ttss_vs_none 40       0.000  0.000  0.000      0.000      0.000       1.00       0.800       0.800       1.000            0.000             1.0              1.0
noise  0.6  ttss_vs_ttt 40       5.774  1.424  9.593      0.000      0.000       1.00       0.800       0.750       0.804            0.000             1.0              1.0
noise  0.8  ttt_vs_none 40      -4.397 -8.341  0.118      0.000      0.000       1.00       0.775       0.800       1.000           49.000             1.0              1.0
noise  0.8 ttss_vs_none 40       0.000  0.000  0.000      0.000      0.000       1.00       0.800       0.800       1.000            0.000             1.0              1.0
noise  0.8  ttss_vs_ttt 40       4.397 -0.118  8.341      0.000      0.000       1.00       0.800       0.775       1.000            0.000             1.0              1.0
noise  1.0  ttt_vs_none 40      -4.423 -8.345  0.040      0.000      0.000       1.00       0.775       0.800       1.000           49.000             1.0              1.0
noise  1.0 ttss_vs_none 40       0.000  0.000  0.000      0.000      0.000       1.00       0.800       0.800       1.000            0.000             1.0              1.0
noise  1.0  ttss_vs_ttt 40       4.423 -0.040  8.345      0.000      0.000       1.00       0.800       0.775       1.000            0.000             1.0              1.0"""

t_rows = []
for line in tta_data.strip().split("\n")[1:]:
    parts = line.split()
    t_rows.append({
        "kind": parts[0],
        "sev": float(parts[1]),
        "contrast": parts[2].replace("_", " "),
        "n": int(parts[3]),
        "d_mean_cle": float(parts[4]),
        "ci_lo": float(parts[5]),
        "ci_hi": float(parts[6]),
        "success_a": float(parts[7]),
        "success_b": float(parts[8]),
        "p_success": float(parts[9]),
        "collapse_a": float(parts[10]),
        "collapse_b": float(parts[11]),
        "p_collapse": float(parts[12]),
        "accepted_mean_a": float(parts[13]),
        "p_success_holm": float(parts[14]),
        "p_collapse_holm": float(parts[15]),
    })
tta_df = pd.DataFrame(t_rows)
tta_df.to_csv(RESULTS_DIR / "tta_paired.csv", index=False)
tta_df.to_json(RESULTS_DIR / "tta_paired.json", orient="records", indent=2)
print("Saved tta_paired.csv and tta_paired.json")

# 6. Parse HYPOTHESES TABLE
hypotheses_data = [
    {"hypothesis": "H0", "verdict": "SUPPORTED", "evidence": "min clean success 0.70, gate 0.2"},
    {"hypothesis": "H1", "verdict": "NOT SUPPORTED", "evidence": "0.44 of 25 curves abrupt failure"},
    {"hypothesis": "H2", "verdict": "SUPPORTED", "evidence": "median error 0.017; gain vs knn_pixel: 0.147 [0.132, 0.167]; gain vs knn_random: 0.262 [0.246, 0.286]"},
    {"hypothesis": "H3", "verdict": "NOT SUPPORTED", "evidence": "tau 0.347, regret 0.028, 6 policies"},
    {"hypothesis": "H4", "verdict": "SUPPORTED", "evidence": "best conditional tau 0.581 vs mean agnostic tau 0.254"},
    {"hypothesis": "H5", "verdict": "SUPPORTED", "evidence": "mean absolute error 0.010 over 105 cases"},
    {"hypothesis": "H6", "verdict": "NOT SUPPORTED", "evidence": "min Holm p 1.000; episodes with accepted updates 0.47, with rejected updates 1.00"},
    {"hypothesis": "H7", "verdict": "NOT SUPPORTED", "evidence": "episode-score rate 0.117, streaming ever-alarmed rate 0.058"},
]
hyp_df = pd.DataFrame(hypotheses_data)
hyp_df.to_csv(RESULTS_DIR / "hypotheses_verdicts.csv", index=False)
hyp_df.to_json(RESULTS_DIR / "hypotheses_verdicts.json", orient="records", indent=2)
print("Saved hypotheses_verdicts.csv and hypotheses_verdicts.json")

# 7. Write complete training provenance for all 6 agents (50k steps)
# In Cell 47:
# drqv2 s0: steps=50000 success=0.80 CLE=3.43 return=-7.14
# drqv2 s1: steps=50000 success=0.80 CLE=3.48 return=-7.24
# drqv2 s2: steps=50000 success=0.80 CLE=3.62 return=-7.54
# svea  s0: steps=50000 success=0.60 CLE=4.55 return=-9.47
# svea  s1: steps=50000 success=0.80 CLE=4.32 return=-9.00
# svea  s2: steps=50000 success=0.70 CLE=4.20 return=-8.75

provenance = {
    "task": "Visual Pursuit (continuous control)",
    "frame": "48x48 RGB, 3-frame stack",
    "episode_length": 50,
    "success_criterion": "episode-mean CLE < 5.0 px",
    "random_floor_cle_px": 22.05,
    "collapse_threshold_px": 17.64,
    "total_training_steps_per_seed": 50000,
    "recipes": {
        "drqv2": {
            "description": "Standard DrQ-v2 with random shift augmentation on both actor and critic",
            "seeds": [0, 1, 2],
            "final_eval": [
                {"seed": 0, "steps": 50000, "success": 0.80, "cle_px": 3.43, "return": -7.14},
                {"seed": 1, "steps": 50000, "success": 0.80, "cle_px": 3.48, "return": -7.24},
                {"seed": 2, "steps": 50000, "success": 0.80, "cle_px": 3.62, "return": -7.54},
            ]
        },
        "svea": {
            "description": "SVEA decoupled loss (critic trained on clean + augmented, actor trained on clean only)",
            "seeds": [0, 1, 2],
            "alpha": 0.5,
            "beta": 0.5,
            "final_eval": [
                {"seed": 0, "steps": 50000, "success": 0.60, "cle_px": 4.55, "return": -9.47},
                {"seed": 1, "steps": 50000, "success": 0.80, "cle_px": 4.32, "return": -9.00},
                {"seed": 2, "steps": 50000, "success": 0.70, "cle_px": 4.20, "return": -8.75},
            ]
        }
    },
    "total_eval_episodes": 6600,
    "shift_kinds": ["background", "noise", "distractor", "viewpoint", "color_gain"],
    "severities": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "episodes_per_cell": 20
}
with open(RESULTS_DIR / "training_provenance_full.json", "w") as f:
    json.dump(provenance, f, indent=2)

print("Saved training_provenance_full.json")
