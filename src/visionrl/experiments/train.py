"""Agent training loops, random floor reference, training gate, and PPO training."""

import copy
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from visionrl.config import Cfg, get_device, seed_for
from visionrl.environments import DiscretePursuit, make_env, rollout
from visionrl.models.drq_agent import DrQAgent
from visionrl.models.ppo_agent import PPOAgent
from visionrl.models.replay_buffer import EpisodeReplay
from visionrl.shifts.observation_shifts import Shift


def eval_cell(
    act_fn: Callable[[np.ndarray], Any],
    cfg: Cfg,
    kind: Optional[str],
    sev: float,
    seeds: List[int],
    record: bool = False,
    env: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Closed-loop evaluation of one policy in one (kind, severity) cell on a list of episode seeds."""
    env = env or make_env(cfg)
    recs = []
    for s in seeds:
        sh = Shift(kind, sev, s, cfg.img) if (kind is not None and sev > 0) else None
        recs.append(rollout(env, act_fn, s, cfg, shift=sh, record=record))
    return recs


def random_floor_cle(cfg: Cfg, n: int = 30) -> float:
    """CLE of a uniform-random policy on the clean task: reference point for floor collapse."""
    env = make_env(cfg)
    rng = np.random.default_rng(1234)
    recs = eval_cell(lambda o: rng.uniform(-1, 1, 2), cfg, None, 0.0, [seed_for("floor", i) for i in range(n)], env=env)
    return float(np.mean([r["mean_cle"] for r in recs]))


class TrainingPaused(Exception):
    """Raised when the time budget is used up. State is saved on disk; call train_agent again to resume."""


def train_agent(
    cfg: Cfg,
    variant: str,
    seed: int,
    verbose: bool = True,
    state_path: Optional[str] = None,
    time_budget: Optional[float] = None,
) -> Tuple[DrQAgent, pd.DataFrame]:
    """Train one continuous-action agent with pause/resume support."""
    env = make_env(cfg)
    agent = DrQAgent(cfg, variant, seed)
    rb = EpisodeReplay(max_eps=max(8, cfg.train_steps // cfg.ep_len + 2), cfg=cfg)
    rng = np.random.default_rng(seed)
    device = get_device()
    steps, ep, curves = 0, 0, []

    if state_path and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        agent.load(st["agent"])
        agent.critic_target.load_state_dict(st["target"])
        agent.enc_opt.load_state_dict(st["opt"][0])
        agent.act_opt.load_state_dict(st["opt"][1])
        agent.cri_opt.load_state_dict(st["opt"][2])
        rb.fr, rb.ac, rb.rw, rb.n = st["rb"]
        steps, ep, curves = st["steps"], st["ep"], st["curves"]
        rng.bit_generator.state = st["rng"]
        torch.manual_seed(seed * 7919 + ep)
        if verbose:
            print(f"  [{variant} s{seed}] resumed at step {steps}")

    t0 = time.time()
    while steps < cfg.train_steps:
        if time_budget and time.time() - t0 > time_budget:
            if state_path:
                torch.save(
                    dict(
                        agent=agent.state(),
                        target=agent.critic_target.state_dict(),
                        opt=(agent.enc_opt.state_dict(), agent.act_opt.state_dict(), agent.cri_opt.state_dict()),
                        rb=(rb.fr, rb.ac, rb.rw, rb.n),
                        steps=steps,
                        ep=ep,
                        curves=curves,
                        rng=rng.bit_generator.state,
                    ),
                    state_path,
                )
            raise TrainingPaused(f"{variant} s{seed} paused at step {steps}/{cfg.train_steps}")

        std = max(cfg.std_end, cfg.std_start - (cfg.std_start - cfg.std_end) * steps / max(1, cfg.std_decay_steps))
        if steps < cfg.seed_steps:
            policy = lambda o: rng.uniform(-1, 1, 2)
        else:
            policy = lambda o, s=std: agent.act(o, noise_std=s)

        rec = rollout(env, policy, seed_for("train", ep, seed), cfg)
        rb.add(rec)
        steps += cfg.ep_len
        ep += 1

        if steps >= cfg.seed_steps:
            for _ in range(cfg.ep_len // cfg.update_every):
                _b = rb.sample(cfg.batch, rng)
                aseq = torch.as_tensor(rb.last_actseq, dtype=torch.float32, device=device) if variant == "taco" else None
                agent.update(_b, std, aseq=aseq)

        if ep % cfg.eval_every_eps == 0 or steps >= cfg.train_steps:
            ev = eval_cell(agent.act, cfg, None, 0.0, [seed_for("eval_curve", i) for i in range(cfg.n_eval_eps)], env=env)
            row = dict(
                variant=variant,
                agent_seed=seed,
                env_steps=steps,
                success=float(np.mean([r["success"] for r in ev])),
                mean_cle=float(np.mean([r["mean_cle"] for r in ev])),
                ret=float(np.mean([r["ret"] for r in ev])),
                train_ret=rec["ret"],
            )
            curves.append(row)
            if verbose:
                print(f"  [{variant} s{seed}] steps {steps:>7d} | clean success {row['success']:.2f} | CLE {row['mean_cle']:.2f} | return {row['ret']:.2f}")

    if state_path and os.path.exists(state_path):
        os.remove(state_path)
    return agent, pd.DataFrame(curves)


def training_gate(agents: Dict[str, Any], cfg: Cfg, n: int = 20) -> pd.DataFrame:
    """Verifies that trained policies solve the clean task above the pre-registered threshold."""
    rows = []
    for _aid, _ag in agents.items():
        ev = eval_cell(_ag.act, cfg, None, 0.0, [seed_for("eval_curve", 500 + i) for i in range(n)])
        rows.append(dict(
            agent_id=_aid,
            clean_success=float(np.mean([r["success"] for r in ev])),
            clean_cle=float(np.mean([r["mean_cle"] for r in ev])),
        ))
    g = pd.DataFrame(rows)
    g["gate_ok"] = g.clean_success >= cfg.train_gate_success
    return g


def train_ppo(
    cfg: Cfg,
    iters: int = 3,
    eps_per_iter: int = 4,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.2,
    epochs: int = 3,
    mb: int = 64,
    ent: float = 0.01,
    seed: int = 0,
) -> PPOAgent:
    """Train a discrete-action PPO agent on DiscretePursuit."""
    device = get_device()
    torch.manual_seed(seed)
    net = PPOAgent(DiscretePursuit.n_actions, 3 * cfg.n_stack, cfg.img).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for it in range(iters):
        O, A, LP, ADV, RET = [], [], [], [], []
        for e in range(eps_per_iter):
            env = DiscretePursuit(cfg)
            o = env.reset(seed_for("train", it * eps_per_iter + e, seed + 100))
            done = False
            ob, ac, lp, vl, rw = [], [], [], [], []

            while not done:
                with torch.no_grad():
                    lg, v = net(torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0))
                    d = torch.distributions.Categorical(logits=lg)
                    a = d.sample()
                o2, r, done, _ = env.step(int(a))
                ob.append(o)
                ac.append(int(a))
                lp.append(float(d.log_prob(a)))
                vl.append(float(v))
                rw.append(r)
                o = o2

            vl = np.array(vl + [0.0])
            rw = np.array(rw)
            adv = np.zeros(len(rw))
            g = 0.0
            for t in reversed(range(len(rw))):
                delta = rw[t] + gamma * vl[t + 1] - vl[t]
                g = delta + gamma * lam * g
                adv[t] = g
            O += ob
            A += ac
            LP += lp
            ADV += list(adv)
            RET += list(adv + vl[:-1])

        O = torch.as_tensor(np.array(O), dtype=torch.float32, device=device)
        A = torch.as_tensor(A, device=device)
        LP = torch.as_tensor(LP, dtype=torch.float32, device=device)
        ADV = torch.as_tensor(ADV, dtype=torch.float32, device=device)
        RET = torch.as_tensor(RET, dtype=torch.float32, device=device)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        for _ in range(epochs):
            perm = torch.randperm(len(O), device=device)
            for i in range(0, len(O), mb):
                ix = perm[i : i + mb]
                lg, v = net(O[ix])
                d = torch.distributions.Categorical(logits=lg)
                ratio = (d.log_prob(A[ix]) - LP[ix]).exp()
                pl = -torch.min(ratio * ADV[ix], ratio.clamp(1 - clip, 1 + clip) * ADV[ix]).mean()
                loss = pl + 0.5 * F.mse_loss(v, RET[ix]) - ent * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    return net


def eval_discrete(net: PPOAgent, cfg: Cfg, seeds: List[int], kind: Optional[str] = None, sev: float = 0.0, rule: str = "greedy") -> pd.DataFrame:
    rows = []
    for s in seeds:
        env = DiscretePursuit(cfg, kind, sev)
        o = env.reset(s)
        done = False
        while not done:
            o, r, done, info = env.step(net.act(o, rule))
        mc = float(np.mean(env.cles))
        rows.append(dict(seed=s, mean_cle=mc, success=bool(mc < cfg.success_cle_px), rule=rule))
    return pd.DataFrame(rows)
