"""Step B: Closed-loop truth evaluation and rollout-free forecast scores."""

import os
import pickle
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from visionrl.config import Cfg
from visionrl.environments import is_collapse, make_env
from visionrl.forecasting.metrics import conformal_threshold, ema_smooth
from visionrl.probes.metrics import grassmann_distance
from visionrl.forecasting.predictor import add_flags_and_z, fit_predict
from visionrl.probes import FeatureBank, RefBank, collect_clean, score_episode
from visionrl.experiments.train import eval_cell


def study_one_agent(
    aid: str,
    ag: Any,
    cfg: Cfg,
    floor: float,
    env: Any,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, float, int], np.ndarray], Dict[Tuple[str, str, float, int], int], List[Dict[str, Any]], Dict[str, Any]]:
    rows, frame_scores, closed, grass = [], {}, {}, []
    t0 = time.time()
    bank = FeatureBank(ag, cfg)
    ref = RefBank(bank, collect_clean(ag, cfg, "ref", cfg.ref_eps, env), cfg)
    clean = collect_clean(ag, cfg, "test", cfg.eps_per_cell, env)  # clean trajectories for post-hoc injection
    seeds = [r["seed"] for r in clean]

    cal_rows, cal_frames = [], []
    for rec in collect_clean(ag, cfg, "cal", cfg.cal_clean_eps, env):
        r_, fpol_, _ = score_episode(bank, ref, rec, "noise", 0.0, cfg)
        cal_rows.append(r_)
        cal_frames.append(fpol_)

    cache = {}
    for kind in cfg.shift_kinds:
        for sev in cfg.severities:
            key = (0.0,) if sev == 0 else (kind, sev)
            if key not in cache:
                truth = eval_cell(ag.act, cfg, kind, sev, seeds, env=env)
                sc, fpol_all, feats = [], [], []
                for rec in clean:
                    r, fpol, fe = score_episode(bank, ref, rec, kind, sev, cfg)
                    sc.append(r)
                    fpol_all.append(fpol)
                    feats.append(fe)
                cache[key] = (truth, sc, fpol_all, np.concatenate(feats))

            truth, sc, fpol_all, feats = cache[key]
            for i, (tr, r, fp) in enumerate(zip(truth, sc, fpol_all)):
                rows.append(dict(
                    agent_id=aid,
                    variant=ag.variant,
                    agent_seed=ag.seed,
                    kind=kind,
                    sev=float(sev),
                    seed=tr["seed"],
                    seed_idx=i,
                    success=bool(tr["success"]),
                    mean_cle=tr["mean_cle"],
                    ret=tr["ret"],
                    collapse=is_collapse(tr["mean_cle"], cfg, floor),
                    **r,
                ))
                frame_scores[aid, kind, float(sev), tr["seed"]] = fp
                closed[aid, kind, float(sev), tr["seed"]] = int(not tr["success"])

            for k in cfg.grassmann_ks:
                grass.append(dict(
                    agent_id=aid,
                    kind=kind,
                    sev=float(sev),
                    k=k,
                    grassmann=grassmann_distance(ref.F["policy"], feats, k) if len(feats) > 1 else float("nan"),
                ))

    print(f"  {aid}: {len(rows)} rows | {time.time() - t0:.0f}s")
    return rows, frame_scores, closed, grass, dict(rows=cal_rows, frames=cal_frames)


def run_study(
    agents: Dict[str, Any],
    cfg: Cfg,
    floor: float,
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Any, Any], Dict[Any, int], Dict[str, float], pd.DataFrame]:
    """Run full evaluation and forecasting sweep with disk caching per agent."""
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    rows, frame_scores, closed, grass, cal_rows, cal_frames = [], {}, {}, [], [], {}
    env = make_env(cfg)

    for _aid, _ag in agents.items():
        cp = os.path.join(cache_dir, f"study_{_aid}.pkl")
        r = pickle.load(open(cp, "rb")) if os.path.exists(cp) else None
        if r is not None and len(r) == 5:
            print(f"  {_aid}: loaded from cache")
        else:
            r = study_one_agent(_aid, _ag, cfg, floor, env)
            pickle.dump(r, open(cp, "wb"))

        rows += r[0]
        frame_scores.update(r[1])
        closed.update(r[2])
        grass += r[3]
        cal_rows += [dict(agent_id=_aid, **x) for x in r[4]["rows"]]
        cal_frames[_aid] = r[4]["frames"]

    df = pd.DataFrame(rows)
    cal = pd.DataFrame(cal_rows)
    clean_tau = {
        _aid: conformal_threshold([ema_smooth(x, cfg.ema).max() for x in cal_frames[_aid]], cfg.clean_fpr)
        for _aid in agents
    }
    return df, pd.DataFrame(grass), frame_scores, closed, clean_tau, cal
