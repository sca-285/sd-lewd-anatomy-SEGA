"""What differs between Forge, reForge and Forge Classic (Neo), in one place.

Checked against the three source trees (Forge 2025-06, reForge 2026-04,
Neo 2026-10):

                     Forge / Neo                          reForge
  batch runner       backend.sampling.sampling_function   ldm_patched.modules.samplers
                     .calc_cond_uncond_batch(model,       .calc_cond_batch(model, [conds],
                      cond, uncond, x, sigma, options)     x, sigma, options)
  condition classes  backend.sampling.condition           ldm_patched.modules.conds
                     (Condition, ConditionCrossAttn)      (CONDRegular, CONDCrossAttn)
  post-CFG args      denoised, cond, uncond, model, uncond_denoised, cond_denoised,
                     sigma, model_options, input   (same keys on all three)

Both condition families have process_cond(batch_size=...) that repeats a
smaller batch, and _copy_with(tensor), which is how a concept conditioning
is put into a copy of the user's own condition entry (so ControlNet, the
inpainting concat and any other key on that entry still apply).
"""

from __future__ import annotations

import functools
import math

import torch


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def name() -> str:
    if _importable("ldm_patched.modules.samplers"):
        return "reforge"
    if _importable("modules_forge.packages.k_diffusion.sampling"):
        return "neo"
    return "forge"


def is_reforge() -> bool:
    return name() == "reforge"


def predict(model, cond_entries, x, sigma, model_options):
    """Denoised prediction for one condition list, through the fork's own batch runner
    (so model_options patches, ControlNet and wrappers apply as in the main call)."""
    if is_reforge():
        from ldm_patched.modules.samplers import calc_cond_batch
        return calc_cond_batch(model, [cond_entries], x, sigma, model_options)[0]
    from backend.sampling.sampling_function import calc_cond_uncond_batch
    return calc_cond_uncond_batch(model, cond_entries, None, x, sigma, model_options)[0]


# keys of a condition entry that restrict or reweight it; a concept covers the whole image
_DROP = ("area", "mask", "mask_strength", "strength", "timestep_start", "timestep_end", "default")


def _wrap(old, tensor):
    if hasattr(old, "_copy_with"):
        return old._copy_with(tensor)
    return type(old)(tensor)


def _fit_tokens(tensors: list) -> list:
    """Same token length for every concept: repeat shorter ones up to the lcm (as the WebUIs do)."""
    lengths = [t.shape[1] for t in tensors]
    if len(set(lengths)) == 1:
        return tensors
    target = functools.reduce(lambda a, b: a * b // math.gcd(a, b), lengths)
    return [t.repeat(1, target // t.shape[1], *([1] * (t.ndim - 2))) for t in tensors]


def concept_entries(template: dict, crossattn: torch.Tensor, vector: torch.Tensor | None) -> list:
    """A one-entry condition list like `template` but with the concept conditioning.

    crossattn: [N, T, D] (N = concepts x batch, already in the row order of x).
    vector:    [N, V] for SDXL-type models, else None.
    """
    entry = {k: v for k, v in template.items() if k not in _DROP}
    model_conds = dict(template["model_conds"])
    model_conds["c_crossattn"] = _wrap(model_conds["c_crossattn"], crossattn)
    entry["cross_attn"] = crossattn
    if vector is not None and "y" in model_conds:
        model_conds["y"] = _wrap(model_conds["y"], vector)
        entry["pooled_output"] = vector
    entry["model_conds"] = model_conds
    return [entry]


def stack_concepts(conds: list, batch: int, device, dtype=None):
    """Concept conditionings -> (crossattn, vector) with rows [c0 x batch, c1 x batch, ...]
    to match x.repeat(n_concepts, 1, ...).

    Each item is one conditioning ({"crossattn", "vector"}, repeated over the batch)
    or a list with one conditioning per batch row (a different prompt per image).
    """
    rows = []
    for c in conds:
        per_row = c if isinstance(c, list) else [c]
        if len(per_row) != batch:
            per_row = [per_row[0]] * batch
        rows.extend(per_row)
    cross = _fit_tokens([r["crossattn"] for r in rows])
    cross = torch.cat(cross).to(device=device, dtype=dtype or cross[0].dtype)
    vector = None
    if all(r.get("vector") is not None for r in rows):
        vector = torch.cat([r["vector"] for r in rows]).to(device=device, dtype=dtype or rows[0]["vector"].dtype)
    return cross, vector


def tile(t: torch.Tensor, n: int) -> torch.Tensor:
    return t.repeat(n, *([1] * (t.ndim - 1)))


def checkpoint_name(p) -> str:
    info = getattr(getattr(p, "sd_model", None), "sd_checkpoint_info", None)
    for attr in ("name_for_extra", "model_name", "title", "filename"):
        value = getattr(info, attr, None)
        if value:
            return str(value)
    try:
        from modules import shared
        return str(shared.opts.sd_model_checkpoint or "")
    except Exception:
        return ""


def checkpoint_key(p) -> str:
    info = getattr(getattr(p, "sd_model", None), "sd_checkpoint_info", None)
    for attr in ("sha256", "shorthash", "hash", "filename"):
        value = getattr(info, attr, None)
        if value:
            return str(value)
    return checkpoint_name(p)


# --------------------------------------------------------------------- tokens

def text_engine(sd_model):
    """The object whose tokenize_line() builds the 77-token chunks of the conditioning.

    Forge / Neo: sd_model.text_processing_engine_l (SDXL) or .text_processing_engine
    (SD 1.x). reForge: sd_model.cond_stage_model (SD 1.x) or a hijacked
    sd_model.conditioner.embedders[i] (SDXL). CLIP-L and CLIP-G split text the same
    way, so the first one found gives the positions for both.
    """
    candidates = [getattr(sd_model, "text_processing_engine_l", None),
                  getattr(sd_model, "text_processing_engine", None),
                  getattr(sd_model, "cond_stage_model", None)]
    conditioner = getattr(sd_model, "conditioner", None)
    candidates += list(getattr(conditioner, "embedders", []) or [])
    for c in candidates:
        if c is not None and callable(getattr(c, "tokenize_line", None)) and callable(getattr(c, "tokenize", None)):
            return c
    return None


def concept_token_positions(engine, full_text: str, concept: str):
    """(positions of the concept's tokens in full_text's conditioning, token length of that
    conditioning), or None when they cannot be found."""
    from lib_lewd_anatomy.attention import token_positions
    try:
        chunks, _ = engine.tokenize_line(full_text)
        flat = [t for c in chunks for t in c.tokens]
        concept_tokens = list(engine.tokenize([concept])[0])
        skip = {getattr(engine, "id_start", None), getattr(engine, "id_end", None), getattr(engine, "id_pad", None)}
        skip.discard(None)
        positions = token_positions(flat, concept_tokens, skip)
        return (positions, len(flat)) if positions else None
    except Exception:
        return None
