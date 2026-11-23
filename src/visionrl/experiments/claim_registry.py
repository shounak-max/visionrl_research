"""Single source of truth for all numerical claims reported in manuscripts."""

import subprocess
from typing import Any, Dict, List, Tuple

import pandas as pd

from visionrl.config import Cfg, SEEDS, get_device


class ClaimRegistry:
    """Every number that appears in more than one place in the paper is registered with its place.
    check() fails if two places disagree. All values are read from result DataFrames.
    """

    def __init__(self):
        self.items: List[Tuple[str, float, str]] = []

    def register(self, name: str, value: float, place: str) -> None:
        self.items.append((name, float(value), place))

    def check(self, tol: float = 1e-9) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(self.items, columns=["quantity", "value", "place"])
        rows = []
        for q, g in df.groupby("quantity"):
            spread = float(g.value.max() - g.value.min())
            rows.append(dict(
                quantity=q,
                places=len(g),
                spread=spread,
                status="OK" if spread <= tol else "MISMATCH",
            ))
        return df, pd.DataFrame(rows)


def config_tables(cfg: Cfg, floor: float) -> Dict[str, pd.Series]:
    device = get_device()
    env = pd.Series(dict(
        task=cfg.env,
        frame=f"{cfg.img}x{cfg.img} RGB, {cfg.n_stack} stacked",
        episode_len=cfg.ep_len,
        max_step_px=cfg.max_step_px,
        target_speed_px_per_step=cfg.target_speed,
        success=f"episode-mean CLE < {cfg.success_cle_px} px",
        collapse=f"episode-mean CLE >= {cfg.collapse_frac} x random-policy CLE ({floor:.2f} px)",
    ), name="value")

    train = pd.Series(dict(
        env_steps=cfg.train_steps,
        seed_steps=cfg.seed_steps,
        batch=cfg.batch,
        lr=cfg.lr,
        discount=cfg.discount,
        n_step=cfg.n_step,
        tau=cfg.tau,
        update_every=cfg.update_every,
        feature_dim=cfg.feature_dim,
        hidden=cfg.hidden,
        aug_pad=cfg.aug_pad,
        std=f"{cfg.std_start} to {cfg.std_end} over {cfg.std_decay_steps} steps",
        svea_alpha_beta=f"{cfg.svea_alpha}, {cfg.svea_beta}",
        agent_seeds=list(cfg.agent_seeds),
    ), name="value")

    evalc = pd.Series(dict(
        shift_kinds=list(cfg.shift_kinds),
        severities=list(cfg.severities),
        episodes_per_cell=cfg.eps_per_cell,
        test_seeds=f"{SEEDS['test']} + i",
        ref_seeds=f"{SEEDS['ref']} + i",
        action_rule="deterministic mean action",
        knn_k=cfg.knn_k,
        clean_fpr=cfg.clean_fpr,
        bootstrap_B=cfg.boot_B,
        device=str(device),
    ), name="value")

    return {"environment": env, "training": train, "evaluation": evalc}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "not a git checkout"
