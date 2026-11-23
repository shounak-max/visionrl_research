# Legacy Heuristic & Preliminary Evaluation Scripts (QUARANTINED)

> [!WARNING]
> **Quarantine & Deprecation Notice:**
> The scripts in this directory (`legacy/heuristic_scripts/`) belong to an earlier exploratory prototype phase.
> They evaluated a hand-coded proportional-control gain policy (`VisualPursuitExpertPolicy`) rather than learned deep RL agents, contained hard-coded update acceptance rates (`trust_region_acceptance_rate: 1.0`), and had confounded learning-rate schedules between naive TTT and safeguarded TTSS.
> 
> **Authoritative Pipeline:**
> These scripts have been completely excised from the paper manuscript and evidence base. All reported experiments, tables, figures, and hypothesis decisions are derived exclusively from the 6,600-episode learned policy pipeline (`notebooks/notebook_revised.ipynb`, `src/visionrl/`, and verified data artifacts in `results/`).
