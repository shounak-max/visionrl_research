"""Base outcome definitions, conversion utilities, and episode rollout function."""

import collections
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from visionrl.config import Cfg


def is_success(mean_cle: float, cfg: Cfg) -> bool:
    """Strict success definition: episode-mean CLE < success_cle_px."""
    return bool(mean_cle < cfg.success_cle_px)


def is_collapse(mean_cle: float, cfg: Cfg, floor_cle: float) -> bool:
    """Floor collapse definition: episode-mean CLE >= collapse_frac * floor_cle."""
    return bool(mean_cle >= cfg.collapse_frac * floor_cle)


def to_u8(frame: np.ndarray) -> np.ndarray:
    """float HWC in [0,1] -> uint8 CHW in [0,255]."""
    return (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).transpose(2, 0, 1)


def from_u8(u8: np.ndarray) -> np.ndarray:
    """uint8 CHW in [0,255] -> float HWC in [0,1]."""
    return u8.astype(np.float32).transpose(1, 2, 0) / 255.0


def make_stacks(view_u8: np.ndarray, n_stack: int) -> np.ndarray:
    """(T+1, 3, H, W) -> (T+1, 3*n_stack, H, W); stack t holds frames t-n+1..t, clipped at episode start."""
    T1 = view_u8.shape[0]
    idx = np.clip(np.arange(T1)[:, None] - np.arange(n_stack - 1, -1, -1)[None, :], 0, None)
    return view_u8[idx].reshape(T1, -1, *view_u8.shape[2:])


def rollout(
    env: Any,
    act_fn: Callable[[np.ndarray], Any],
    seed: int,
    cfg: Cfg,
    shift: Optional[Callable[[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]]] = None,
    record: bool = False,
) -> Dict[str, Any]:
    """Run one episode under an environment and policy.
    
    act_fn maps a stacked uint8 observation (3*n_stack, H, W) to an action in [-1, 1]^2.
    If `shift` is given, the policy sees shifted frames while CLE is measured in the true state.
    record=True retains the clean frames and entity masks so shifts can be injected post-hoc.
    """
    f, m = env.reset(seed)
    clean = [to_u8(f)]
    masks = [m.copy()]
    view = [to_u8(f if shift is None else shift(f, m, 0)[0])]
    stack = collections.deque([view[0]] * cfg.n_stack, maxlen=cfg.n_stack)
    cles, rews, acts = [], [], []

    for t in range(1, cfg.ep_len + 1):
        obs = np.concatenate(list(stack), 0)
        a = np.asarray(act_fn(obs), np.float32)
        f, m, r, cle, done = env.step(a)
        v = to_u8(f if shift is None else shift(f, m, t)[0])
        stack.append(v)
        view.append(v)
        clean.append(to_u8(f))
        masks.append(m.copy())
        cles.append(cle)
        rews.append(r)
        acts.append(a)

    mean_cle = float(np.mean(cles))
    rec = dict(
        seed=int(seed),
        mean_cle=mean_cle,
        ret=float(np.sum(rews)),
        cle_t=np.array(cles, np.float32),
        success=is_success(mean_cle, cfg),
        view_u8=np.stack(view),
        actions=np.stack(acts),
        rewards=np.array(rews, np.float32),
    )
    if record:
        rec["clean_u8"] = np.stack(clean)
        rec["masks"] = np.stack(masks)
    return rec
