"""A small opt-in API for other extensions (meant for Impact ADetailer).

Why it is needed: Impact ADetailer repaints with a nested img2img job whose
script runner is empty. On Forge, reForge and Neo, every sampling pass starts
by resetting p.sd_model.forge_objects from the unpatched copy
(processing.py: `forge_objects = forge_objects_original.shallow_copy()` per
batch, then `forge_objects_after_applying_lora.shallow_copy()` before each
pass), so the patched UNet of the main job is NOT on the nested job's model.
During the main job's postprocess_image it is still there (until the next
batch resets it), but the nested job replaces it first.

Usage from another extension, inside the nested job's
process_before_every_sampling (or right before its sampling):

    try:
        from lib_lewd_anatomy.api import current_patch
    except ImportError:
        current_patch = None
    patch = current_patch(main_p) if current_patch else None
    if patch is not None:
        nested_p.sd_model.forge_objects.unet = patch(nested_p.sd_model.forge_objects.unet)

The patch re-uses the main job's concepts and settings with a fresh momentum.
The concept conditionings keep the main image's size embedding (SDXL).
"""

from __future__ import annotations

from lib_lewd_anatomy import engine


def current_patch(p):
    """A function unet -> patched unet clone for the job `p`, or None if Lewd Anatomy
    did nothing on that job."""
    job = getattr(p, "_lewd_anatomy_job", None)
    if not job:
        return None

    def patch(unet):
        clone, _ = engine.patch_unet(unet, job["plan"], job["conds"], job["settings"], label="nested job")
        return clone

    return patch
