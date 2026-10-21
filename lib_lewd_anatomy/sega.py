"""Semantic Guidance (SEGA) math, in the noise (epsilon) space of the paper.

Brack et al., "SEGA: Instructing Text-to-Image Models using Semantic
Guidance", NeurIPS 2023. The masking, momentum and warmup follow the
reference implementation (diffusers' SemanticStableDiffusionPipeline,
Apache-2.0; see NOTICE.md).

The WebUI's CFG hooks work on denoised predictions (x0). For the k-diffusion
parameterisation the WebUI uses, eps = (x - x0) / sigma, so

    eps_e - eps_u = (x0_u - x0_e) / sigma
    eps~ = eps_cfg + g   <=>   x0~ = x0_cfg - sigma * g

which is what `apply` does. Pure torch; no WebUI imports.
"""

from __future__ import annotations

import torch


def _per_sample(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """sigma as [B, 1, 1, ...] broadcastable against `like` ([B, C, ...])."""
    s = torch.as_tensor(sigma, device=like.device, dtype=torch.float32).reshape(-1)
    if s.numel() == 1:
        s = s.expand(like.shape[0])
    return s.reshape(-1, *([1] * (like.ndim - 1)))


def quantile_mask(diff: torch.Tensor, locality: float) -> torch.Tensor:
    """True where |diff| is at or above its `locality` quantile, per sample and channel.

    diff: [B, C, ...]. Spatial dims (and frames, for video latents) are flattened.
    """
    flat = diff.abs().flatten(start_dim=2).float()
    q = torch.quantile(flat, float(locality), dim=2, keepdim=True)
    return (flat >= q).reshape(diff.shape)


def concept_guidance(eps_base: torch.Tensor, eps_edit: torch.Tensor, scale: float, locality: float) -> torch.Tensor:
    """gamma = s * mask * (eps_edit - eps_base); a negative scale pushes away.

    eps_base is the user-prompt prediction, not the unconditional / negative one.
    """
    diff = (eps_edit - eps_base).float()
    return torch.where(quantile_mask(diff, locality), diff * float(scale), torch.zeros_like(diff))


def _norm(t: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm, shaped [B, 1, 1, ...]."""
    return t.flatten(start_dim=1).norm(dim=1).reshape(-1, *([1] * (t.ndim - 1)))


def budget_factor(total: torch.Tensor, parts: list, budget: float) -> torch.Tensor | float:
    """How much to shrink `total` so its size stays within `budget` times the largest
    single concept's size (per sample). 1 when within budget or budget <= 0.

    One concept is never shrunk. Many concepts acting on the same latent values
    would otherwise add up past anything a single slider does, which is what
    distorts anatomy with 8-12 concepts.
    """
    if budget is None or budget <= 0 or len(parts) < 2:
        return 1.0
    biggest = torch.stack([_norm(g) for g in parts]).amax(dim=0)
    size = _norm(total)
    limit = biggest * float(budget)
    return torch.where(size > limit, limit / size.clamp_min(1e-12), torch.ones_like(size))


def pixel_cap(t: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """t [B, C, ...] scaled down, pixel by pixel, to a channel norm of at most limit [B, 1, ...]."""
    norm = t.float().pow(2).sum(1, keepdim=True).sqrt()
    return (t.float() * (limit / norm.clamp_min(1e-12)).clamp(max=1.0)).to(t.dtype)


class Sega:
    """SEGA for one sampling pass: holds the momentum between steps."""

    def __init__(self, momentum_scale: float = 0.3, beta: float = 0.6, budget: float | None = None):
        self.momentum_scale = float(momentum_scale)
        self.beta = float(beta)
        self.budget = budget
        self.momentum = None
        self.overlap = None           # last step's overlap figures (for the console warning)

    def reset(self):
        self.momentum = None

    @staticmethod
    def diffs(eps_base: torch.Tensor, eps_concepts: list, refs: list | None = None) -> list:
        """eps_i - eps_ref(i) for every concept (ref None = eps_base): what each concept edits."""
        refs = refs or [None] * len(eps_concepts)
        return [(e - (eps_base if r is None else eps_concepts[r])).float() for e, r in zip(eps_concepts, refs)]

    def guidance_from_diffs(self, diffs: list, scales: list, localities: list, live: list,
                            owners: list | None = None, overlap_budget: float = 0.0) -> torch.Tensor | None:
        """The total edit guidance in eps space for one step, or None if nothing applies.

        diffs[i] is concept i's difference (fresh, or kept from an earlier step when
        concepts are rotated), or None when it is not available on this step.
        As in the reference implementation, a concept in warmup still feeds the
        momentum, and the momentum is only applied once every concept is live;
        until then only the live concepts are applied.

        Overlap budget: wherever several sliders (owners) edit the same pixels, their
        summed edit there is kept within overlap_budget x the strongest single slider at
        that pixel (norm over channels), before and after momentum. Nine sliders on one
        crotch no longer add up into something the model cannot draw.
        """
        self.overlap = None
        n = len(diffs)
        owners = owners if owners is not None else list(range(n))
        masked = {}
        for i, (d, s, lam) in enumerate(zip(diffs, scales, localities)):
            if s == 0 or d is None:
                continue
            masked[i] = torch.where(quantile_mask(d, lam), d * float(s), torch.zeros_like(d))
        if not masked:
            return None
        parts = [masked[i] for i in sorted(masked)]
        live_parts = [masked[i] for i in sorted(masked) if live[i]]
        total = torch.stack(parts).sum(dim=0)
        live_sum = torch.stack(live_parts).sum(dim=0) if live_parts else torch.zeros_like(total)
        limit = None
        if overlap_budget and overlap_budget > 0 and total.ndim >= 3:
            by_owner = {}
            for i in masked:
                by_owner.setdefault(owners[i], []).append(i)
            if len(by_owner) > 1:
                groups = list(by_owner.values())
                per = torch.stack([torch.stack([masked[i] for i in idx]).sum(0) for idx in groups]).float()
                norms = per.pow(2).sum(2, keepdim=True).sqrt()                      # [S, B, 1, ...]
                limit = float(overlap_budget) * norms.amax(0)
                # Sign election (as TIES merging): where sliders push one latent value opposite
                # ways, the minority's push is dropped instead of summed. Opposite pushes cancel
                # into mush and still use up the budget; dropping them leaves the agreeing
                # sliders more of it.
                elected = torch.sign(per.sum(0, keepdim=True))
                agree = ((torch.sign(per) == elected) | (per == 0)).to(per.dtype)
                total = (per * agree).sum(0).to(total.dtype)
                live_per = torch.stack([torch.stack([masked[i] if live[i] else torch.zeros_like(masked[i])
                                                     for i in idx]).sum(0) for idx in groups]).float()
                live_sum = (live_per * agree).sum(0).to(live_sum.dtype)
                capped = pixel_cap(total, limit)
                # what the warning in the console reports: how hard the cap bit, where
                tn = total.float().pow(2).sum(1, keepdim=True).sqrt()
                factor = (limit / tn.clamp_min(1e-12)).clamp(max=1.0)
                busy = (norms > 0.05 * norms.amax().clamp_min(1e-12)).sum(0)
                self.overlap = {"max_sliders": int(busy.max()),
                                "cut_share": float(((factor < 0.6) & (busy >= 2)).float().mean()),
                                "min_factor": float(factor.min())}
                total, live_sum = capped, pixel_cap(live_sum, limit)
        shrink = budget_factor(total, parts, self.budget)
        total, live_sum = total * shrink, live_sum * shrink
        if self.momentum is not None and self.momentum.shape == total.shape:
            total = total + self.momentum_scale * self.momentum
            if limit is not None:
                total = pixel_cap(total, limit)
            self.momentum = self.beta * self.momentum + (1.0 - self.beta) * total
        else:
            self.momentum = (1.0 - self.beta) * total
        if all(on for on, s, d in zip(live, scales, diffs) if s != 0 and d is not None):
            return total
        return live_sum

    def guidance(self, eps_u: torch.Tensor, eps_concepts: list, scales: list, localities: list, live: list,
                 refs: list | None = None) -> torch.Tensor | None:
        """Guidance from predictions: refs[i] (optional) is the index of the concept that
        concept i is measured against instead of eps_u (a contrastive pair). A scale-0
        concept is only a reference and adds nothing itself."""
        return self.guidance_from_diffs(self.diffs(eps_u, eps_concepts, refs), scales, localities, live)

    def apply(self, x, sigma, denoised, baseline_x0, concept_x0: list, scales, localities, live, refs=None):
        """x0-space wrapper: returns the new denoised prediction.

        ``baseline_x0`` is the model's prediction for the user's prompt (or the
        negative prompt, for baseline Negative).
        """
        sig = _per_sample(sigma, x)
        if not torch.all(sig > 0):
            return denoised
        eps_base = (x.float() - baseline_x0.float()) / sig
        eps_e = [(x.float() - c.float()) / sig for c in concept_x0]
        return self.apply_diffs(x, sig, denoised, self.diffs(eps_base, eps_e, refs), scales, localities, live)

    def apply_diffs(self, x, sigma, denoised, diffs, scales, localities, live, owners=None, overlap_budget=0.0):
        sig = _per_sample(sigma, x)
        if not torch.all(sig > 0):
            return denoised
        g = self.guidance_from_diffs(diffs, scales, localities, live, owners, overlap_budget)
        if g is None:
            return denoised
        return (denoised.float() - sig * g).to(denoised.dtype)


# --------------------------------------------------------------------- extra guidance math

def perpendicular(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """v without its component along ref, per sample (Perp-Neg, Armandpour et al. 2023)."""
    vf, rf = v.float().flatten(1), ref.float().flatten(1)
    coef = (vf * rf).sum(1, keepdim=True) / (rf * rf).sum(1, keepdim=True).clamp_min(1e-12)
    return (vf - coef * rf).view_as(v).to(v.dtype)


def cfg_rescale(x, sigma, denoised, cond_denoised, phi: float):
    """Rescaled CFG (Lin et al. 2024, "Common Diffusion Noise Schedules and Sample Steps
    are Flawed"): the guided prediction's standard deviation brought back toward the
    prompt prediction's, in v-space, mixed by phi. Lets high CFG run without burning.
    Same arithmetic as ComfyUI's RescaleCFG, applied to the final guided prediction."""
    if phi <= 0:
        return denoised
    sig = _per_sample(sigma, x)
    xs = x.float() / (sig * sig + 1.0)
    k = (sig * sig + 1.0) ** 0.5 / sig
    v_cond = (xs - cond_denoised.float()) * k
    v_cfg = (xs - denoised.float()) * k
    dims = tuple(range(1, x.ndim))
    ro_pos = v_cond.std(dim=dims, keepdim=True)
    ro_cfg = v_cfg.std(dim=dims, keepdim=True).clamp_min(1e-12)
    v = phi * (v_cfg * ro_pos / ro_cfg) + (1.0 - phi) * v_cfg
    return (xs - v / k).to(denoised.dtype)
