"""Smoke test for extra recipes (TACO auxiliary contrastive objective)."""

from dataclasses import replace
from typing import Tuple

import numpy as np
import torch

from visionrl.config import Cfg, get_device
from visionrl.models.drq_agent import DrQAgent


def smoke_extra_recipe(cfg: Cfg, variant: str = "taco") -> Tuple[bool, str]:
    """Checks that the recipe runs one update, yields finite losses, updates auxiliary heads,
    sends gradient into encoder, and survives state save and load.
    """
    device = get_device()
    try:
        c = replace(cfg, extra_recipes=(variant,))
        ag = DrQAgent(c, variant, 0)
        B, H, ch = 8, c.img, 3 * c.n_stack
        rnd = lambda *s: torch.rand(*s, device=device)
        batch = (
            rnd(B, ch, H, H) * 255.0,
            rnd(B, 2) * 2 - 1,
            rnd(B, 1),
            torch.full((B, 1), c.discount ** c.n_step, device=device),
            rnd(B, ch, H, H) * 255.0,
        )
        before = [p.detach().clone() for p in ag.aux.parameters()]
        lc, la = ag.update(batch, 0.2, aseq=rnd(B, c.n_step * 2) * 2 - 1)
        moved = any(float((p.detach() - b).abs().max()) > 0 for p, b in zip(ag.aux.parameters(), before))
        enc_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in ag.encoder.parameters())

        ag2 = DrQAgent(c, variant, 1)
        ag2.load(ag.state())
        same = all(torch.equal(a, b) for a, b in zip(ag.aux.state_dict().values(), ag2.aux.state_dict().values()))

        ok = bool(np.isfinite(lc) and np.isfinite(la) and moved and enc_grad and same)
        msg = f"critic loss {lc:.4f}, actor loss {la:.4f}, aux updated {moved}, encoder gradient {enc_grad}, state round-trip {same}"
    except Exception as e:
        ok, msg = False, f"{type(e).__name__}: {e}"

    return ok, msg
