"""Main end-to-end experiment runner for VisionRL study."""

import json
import os
import pickle
import platform
import time
from dataclasses import asdict, replace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from visionrl.config import Cfg, PREREG, get_device, make_cfg, seed_for
from visionrl.environments import make_env
from visionrl.forecasting.metrics import auroc, cluster_bootstrap, spearman
from visionrl.forecasting.predictor import (
    PROPOSED,
    SCORE_COLS,
    add_flags_and_z,
    alarm_timing,
    cliff_table,
    correlation_vs_threshold,
    dose_response,
    fit_predict,
    forecast_cliffs,
    pooled_rows,
    predictor_table,
)
from visionrl.forecasting.prevalence import portable_check
from visionrl.forecasting.ranking import identifiability_table, ranking_table
from visionrl.models.drq_agent import DrQAgent
from visionrl.probes.adaptation import arm_config_table, compare_arms, run_tta
from visionrl.probes.feature_bank import collect_clean
from visionrl.experiments.claim_registry import ClaimRegistry, config_tables, git_commit
from visionrl.experiments.hypotheses import cross_env_summary, evaluate_hypotheses
from visionrl.experiments.latency import measure_latency
from visionrl.experiments.smoke_recipe import smoke_extra_recipe
from visionrl.experiments.study import run_study
from visionrl.experiments.train import (
    TrainingPaused,
    eval_cell,
    eval_discrete,
    random_floor_cle,
    train_agent,
    train_ppo,
    training_gate,
)


def run_pipeline(
    mode: Optional[str] = None,
    env_name: Optional[str] = None,
    out_dir: Optional[str] = None,
    train_budget: Optional[float] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute complete end-to-end VisionRL study pipeline."""
    # Bug 1 Fix: dynamically read environment variables without hardcoded override
    mode = mode or os.environ.get("VRL_MODE", "smoke")
    env_name = env_name or os.environ.get("VRL_ENV", "pursuit")
    out_dir = out_dir or os.environ.get("VRL_OUT", f"results_{mode}_{env_name}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "cache"), exist_ok=True)

    cfg = make_cfg(mode, env_name)
    env_ov = json.loads(os.environ.get("VRL_OVERRIDE", "{}"))
    if overrides:
        env_ov.update(overrides)
    if env_ov:
        cfg = replace(cfg, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in env_ov.items()})
        print("Overrides applied:", env_ov)

    # Bug 5 Fix: Ensure tta_lr is appropriately scaled in non-smoke modes
    if cfg.mode != "smoke" and cfg.tta_lr < 1e-3:
        cfg = replace(cfg, tta_lr=1e-3)

    device = get_device()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    print(f"\n[VisionRL Pipeline] Mode: {cfg.mode} | Env: {cfg.env} | Device: {device} | Output: {out_dir}")

    floor = random_floor_cle(cfg)
    print(f"Floor CLE: {floor:.2f} px | Success threshold: {cfg.success_cle_px} px | Collapse threshold: {cfg.collapse_frac * floor:.2f} px")

    # Step A: Train agents
    agents = {}
    curves = []
    t_budget = train_budget or (float(os.environ.get("VRL_TRAIN_BUDGET", "0")) or None)
    recipes = tuple(cfg.agents) + tuple(cfg.extra_recipes)

    for variant in recipes:
        for s in cfg.agent_seeds:
            aid = f"{variant}_s{s}"
            ckpt_path = os.path.join(out_dir, "ckpt", f"{aid}.pt")
            cpath = os.path.join(out_dir, "ckpt", f"{aid}_curve.csv")

            if os.path.exists(ckpt_path):
                ag = DrQAgent(cfg, variant, s)
                ag.load(torch.load(ckpt_path, map_location=device, weights_only=False))
                if os.path.exists(cpath):
                    curves.append(pd.read_csv(cpath))
                print(f"Loaded {aid}")
            else:
                print(f"Training {aid}...")
                ag, cur = train_agent(cfg, variant, s, state_path=ckpt_path + ".state", time_budget=t_budget)
                torch.save(ag.state(), ckpt_path)
                cur.to_csv(cpath, index=False)
                curves.append(cur)
            agents[aid] = ag

    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if len(curves_df):
        curves_df.to_csv(os.path.join(out_dir, "training_curves.csv"), index=False)

    gate_df = training_gate(agents, cfg)
    gate_df.to_csv(os.path.join(out_dir, "training_gate.csv"), index=False)
    print("Training Gate Results:\n", gate_df)

    if cfg.mode != "smoke" and cfg.train_gate_success > 0 and (not gate_df.gate_ok.all()):
        raise RuntimeError("Training gate failed: policies do not solve clean task.")

    # Step B: Closed loop evaluation and rollout-free forecast scores
    raw_df, grass_df, frame_scores, closed, clean_tau, cal = run_study(agents, cfg, floor, out_dir)
    episodes = fit_predict(add_flags_and_z(raw_df, cfg, cal), cfg, cal)
    cal.to_csv(os.path.join(out_dir, "clean_calibration.csv"), index=False)
    episodes.to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
    grass_df.to_csv(os.path.join(out_dir, "grassmann.csv"), index=False)

    # Step C: Analysis
    dr = dose_response(episodes)
    cliff = cliff_table(episodes)
    corr = correlation_vs_threshold(episodes)
    tab_strict = predictor_table(episodes, cfg, proto="strict")
    tab_seedout = predictor_table(episodes, cfg, proto="seed_out")

    # Paired AUROC difference
    x_pool = pooled_rows(episodes)
    diff_rows = []
    for base in [c for c in SCORE_COLS if c != PROPOSED]:
        def f_diff(df_, b_col=base):
            a_val = auroc(df_["z_" + PROPOSED].values, df_.fail.values)
            b_val = auroc(df_["z_" + b_col].values, df_.fail.values)
            return a_val - b_val
        lo, hi = cluster_bootstrap(f_diff, x_pool, "seed_idx", cfg.boot_B)
        diff_rows.append(dict(baseline=base, d_auroc=f_diff(x_pool), ci_lo=lo, ci_hi=hi))
    diff_df = pd.DataFrame(diff_rows)

    fc_iso = forecast_cliffs(episodes, PROPOSED, "strict")
    fc_base = pd.concat([forecast_cliffs(episodes, c, "strict").assign(predictor=c) for c in SCORE_COLS], ignore_index=True) if len(episodes) else pd.DataFrame()
    alarm, alarm_ep = alarm_timing(frame_scores, closed, cfg, clean_tau)

    # Grassmann correlation
    grass_summary = pd.DataFrame()
    if len(grass_df) and len(dr):
        _g = grass_df.merge(dr[["agent_id", "kind", "sev", "success_rate"]], on=["agent_id", "kind", "sev"])
        _g = _g.merge(dr[dr.sev == 0][["agent_id", "kind", "success_rate"]].rename(columns={"success_rate": "sr0"}), on=["agent_id", "kind"])
        _g["drop"] = _g.sr0 - _g.success_rate
        grass_summary = pd.DataFrame([dict(k=k, spearman_with_drop=spearman(t.grassmann, t["drop"]), n=len(t)) for k, t in _g.groupby("k")]).set_index("k")

    # Bug 2 Fix: Serialize analysis results to cache
    analysis_cache_path = os.path.join(out_dir, "cache", "analysis_results.pkl")
    with open(analysis_cache_path, "wb") as f_cache:
        pickle.dump(dict(
            DR=dr,
            CLIFF=cliff,
            CORR=corr,
            TAB_STRICT=tab_strict,
            TAB_SEEDOUT=tab_seedout,
            DIFF=diff_df,
            FC_BASE=fc_base,
            ALARM=alarm,
            ALARM_EP=alarm_ep,
            GRASS=grass_summary,
        ), f_cache)

    tab_strict.round(3).to_csv(os.path.join(out_dir, "tables", "predictors_strict.csv"))
    tab_seedout.round(3).to_csv(os.path.join(out_dir, "tables", "predictors_seed_out.csv"))
    cliff.round(3).to_csv(os.path.join(out_dir, "tables", "cliff_table.csv"), index=False)
    corr.round(3).to_csv(os.path.join(out_dir, "tables", "correlation_vs_threshold.csv"), index=False)
    alarm.round(3).to_csv(os.path.join(out_dir, "tables", "alarm_timing.csv"), index=False)

    # Step D: Test-time adaptation experiment
    tta_list = []
    for _aid, _ag in list(agents.items())[: cfg.tta_agents]:
        cp = os.path.join(out_dir, "cache", f"tta_{_aid}.pkl")
        if os.path.exists(cp):
            tta_list.append(pickle.load(open(cp, "rb")))
        else:
            ref_recs = collect_clean(_ag, cfg, "ref", cfg.ref_eps)
            t_ = run_tta(_ag, _aid, cfg, ref_recs, floor)
            pickle.dump(t_, open(cp, "wb"))
            tta_list.append(t_)
    tta_df = pd.concat(tta_list, ignore_index=True) if tta_list else pd.DataFrame()
    if len(tta_df):
        tta_df.to_csv(os.path.join(out_dir, "tta_episodes.csv"), index=False)
    tta_cmp = compare_arms(tta_df, B=max(200, cfg.boot_B)) if len(tta_df) else pd.DataFrame()
    if len(tta_cmp):
        tta_cmp.round(3).to_csv(os.path.join(out_dir, "tables", "tta_paired.csv"), index=False)

    # Step E: Claims registry and manifest
    reg = ClaimRegistry()
    clean_sr = dr[dr.sev == 0].groupby("agent_id").success_rate.mean()
    clean_cle = dr[dr.sev == 0].groupby("agent_id").mean_cle.mean()
    for _aid in agents:
        if _aid in clean_sr:
            reg.register(f"clean success {_aid}", clean_sr[_aid], "Table 1 (dose-response)")
            reg.register(f"clean success {_aid}", cliff[cliff.agent_id == _aid].clean_success.mean(), "Table 3 (cliff table)")
            reg.register(f"clean success {_aid}", episodes[(episodes.agent_id == _aid) & (episodes.sev == 0)].success.mean(), "abstract")
            reg.register(f"clean CLE {_aid}", clean_cle[_aid], "Table 1 (dose-response)")
            reg.register(f"clean CLE {_aid}", episodes[(episodes.agent_id == _aid) & (episodes.sev == 0)].mean_cle.mean(), "abstract")

    reg_items, reg_check = reg.check()
    ct = config_tables(cfg, floor)
    manifest = dict(
        mode=cfg.mode,
        cfg=asdict(cfg),
        floor_cle=floor,
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        device=str(device),
        commit=git_commit(),
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        claims_registry="OK" if (len(reg_check) == 0 or (reg_check.status == "OK").all()) else "MISMATCH",
        n_episode_rows=int(len(episodes)),
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f_man:
        json.dump(manifest, f_man, indent=2, default=str)

    # Step F: Hypotheses and replication
    rank_df = pd.concat([ranking_table(episodes, c) for c in SCORE_COLS], ignore_index=True) if len(episodes) else pd.DataFrame()
    ident_df = identifiability_table(episodes) if len(episodes) else pd.DataFrame()
    portable_df = portable_check(episodes) if len(episodes) else pd.DataFrame()

    hyp_df = evaluate_hypotheses(
        cfg, gate_df, cliff, fc_base, rank_df, portable_df, tta_cmp, tta_df, tab_strict, alarm, len(agents)
    )
    hyp_df.to_csv(os.path.join(out_dir, "hypotheses.csv"), index=False)
    if len(rank_df):
        rank_df.to_csv(os.path.join(out_dir, "tables", "ranking.csv"), index=False)
    if len(portable_df):
        portable_df.to_csv(os.path.join(out_dir, "tables", "portable_auroc.csv"), index=False)

    xenv_df, envs_found = cross_env_summary(current=(cfg.env, hyp_df), mode=cfg.mode, root=".")
    if xenv_df is not None:
        xenv_df.to_csv(os.path.join(out_dir, "cross_environment.csv"), index=False)

    # Latency benchmarking
    latency_df, hw_dict = measure_latency(cfg, next(iter(agents.values())))
    latency_df.to_csv(os.path.join(out_dir, "latency.csv"), index=False)
    with open(os.path.join(out_dir, "hardware.json"), "w") as f_hw:
        json.dump(hw_dict, f_hw, indent=2)

    # Smoke test extra recipe (TACO)
    smoke_ok, smoke_msg = smoke_extra_recipe(cfg, "taco")
    print(f"TACO Smoke Recipe Status: {'PASS' if smoke_ok else 'FAIL'} ({smoke_msg})")

    print(f"\n[VisionRL Pipeline Finished] Outputs written to: {out_dir}")
    return dict(
        cfg=cfg,
        agents=agents,
        episodes=episodes,
        hypotheses=hyp_df,
        manifest=manifest,
        latency=latency_df,
    )


if __name__ == "__main__":
    run_pipeline()
