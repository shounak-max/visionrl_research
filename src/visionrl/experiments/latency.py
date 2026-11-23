"""Latency and hardware benchmarking protocol (Reviewer ykVG)."""

import os
import platform
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from visionrl.config import Cfg, get_device
from visionrl.environments import make_env, make_stacks
from visionrl.probes import FeatureBank, RefBank, collect_clean, score_episode


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn: Callable[[], Any], reps: int, warm: int = 5) -> Tuple[float, float, float]:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return float(np.median(ts)), float(np.percentile(ts, 90)), float(ts.min())


def measure_latency(cfg: Cfg, agent: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Benchmarking protocol: batch size 1, device synchronization before/after, median/p90/min ms."""
    device = get_device()
    env = make_env(cfg)
    ref_recs = collect_clean(agent, cfg, "ref", cfg.ref_eps, env=env)
    bank = FeatureBank(agent, cfg)
    ref = RefBank(bank, ref_recs, cfg)
    rec = collect_clean(agent, cfg, "test", 1, env=env)[0]
    stacks = make_stacks(rec["clean_u8"], cfg.n_stack)[:-1]
    obs, mk = stacks[0], rec["masks"][:1]

    ops = [
        ("policy action, 1 observation", lambda: agent.act(obs), cfg.latency_reps * 10),
        ("monitoring score, 1 observation (features + kNN)", lambda: ref.knn("policy", bank.extract(stacks[:1], mk)["policy"], cfg.knn_k), cfg.latency_reps * 5),
        ("forecast scores, 1 whole episode re-rendered under shift (all nine-plus scores)", lambda: score_episode(bank, ref, {k: v for k, v in rec.items() if k != "_cf"}, "noise", 0.5, cfg), cfg.latency_reps),
    ]

    rows = []
    for name, fn, reps in ops:
        med, p90, mn = _time(fn, reps)
        rows.append(dict(operation=name, median_ms=med, p90_ms=p90, min_ms=mn, reps=reps))

    hw = dict(
        platform=platform.platform(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        torch=torch.__version__,
        device=str(device),
        cuda=bool(torch.cuda.is_available()),
        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        torch_threads=torch.get_num_threads(),
        image_px=cfg.img,
        reference_rows=int(len(ref.F["policy"])),
        knn_k=cfg.knn_k,
        batch_size=1,
    )
    return pd.DataFrame(rows), hw
