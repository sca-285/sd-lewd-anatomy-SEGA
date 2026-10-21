"""Putting SEGA on a clone of the WebUI's UNet patcher for one sampling pass.

Hook: a sampler *post*-CFG function (model_options["sampler_post_cfg_function"]
is a list on all three forks). It runs after the CFG function, whoever set
it (plain CFG, Steer's FDG / APG / CHG, Dynamic Thresholding, ...), and
after the post-CFG functions registered before it (Steer's PAG / SEG / SAG
...). SEGA's correction is additive, so it chains with all of them instead
of replacing any.

Per step inside the step range: the concept predictions, through the fork's
own batch runner, `per_call` concepts per model call (x repeated once per
concept in the call, concept conditionings stacked on the batch dimension,
rows [c0 x B, c1 x B, ...]). per_call = 1 keeps every extra call at batch B,
half of a normal CFG step (2B), which is what avoids the out-of-memory crash
of one big call on mid-range cards.
"""

from __future__ import annotations

import math
import re
import time

import torch
import torch.nn.functional as F

from lib_lewd_anatomy import attention as A
from lib_lewd_anatomy import concepts as K
from lib_lewd_anatomy import host
from lib_lewd_anatomy import isolate as I
from lib_lewd_anatomy.sega import Sega, cfg_rescale, perpendicular

MOMENTUM_SCALE = 0.3
MOMENTUM_BETA = 0.6

_cond_cache = {}            # (checkpoint, text, width, height) -> {"crossattn", "vector"} on the CPU
_CACHE_LIMIT = 128           # entries at most
_CACHE_BYTES = 64 * 2 ** 20  # and at most this much RAM (a long SDXL prompt is ~1 MB per entry)
_cache_bytes = 0
_cache_key = None            # the checkpoint the cache belongs to
ENCODE_BATCH = 4


class Engine:
    """SEGA state for one sampling pass.

    Per step, inside the step range:
      1. which concepts are due: inside their window (Staggered ranges), and with
         Rotate N, about N passes' worth of concept groups in turn (a group is a
         toward concept plus the concepts measured against it, so a contrastive
         pair is always predicted on the same step);
      2. predict the due concepts through the fork's batch runner, `chunk` concepts
         per call (Auto batch: as many as free VRAM allows, halved on out-of-memory
         and retried, never below 1);
      3. difference of each concept against its reference, kept for the steps on
         which a rotated concept is not recomputed;
      4. SEGA masks and sums (where sliders overlap, within the overlap budget per pixel), and
         momentum (off while concepts take turns: it would carry stale edits along).
    """

    def __init__(self, plan: K.Plan, conds: list, settings: dict, label: str = "pass", log=print,
                 table: dict | None = None):
        table = table if table is not None else _table()
        self.plan = plan
        self.conds = conds
        n = len(plan.concepts)
        # Control mode scales every edit, nothing else (it also sets Orthogonal, below)
        self.control = K.control_mode(settings)
        self.control_gain = K.control_gain(table, self.control)
        self.scales = [c.signed_scale * self.control_gain for c in plan.concepts]
        self.refs = K.ref_indices(plan.concepts)
        self.warmups = [c.warmup for c in plan.concepts]
        self.windows = K.windows(plan.concepts, settings, table)
        self.groups = K.groups(plan.concepts)
        self.locality = float(settings.get("locality", K.DEFAULTS["locality"]))
        self.start = float(settings.get("start", K.DEFAULTS["start"]))
        self.end = float(settings.get("end", K.DEFAULTS["end"]))
        self.per_call = max(1, int(settings.get("per_call", K.DEFAULTS["per_call"]) or 1))
        self.auto_batch = bool(settings.get("auto_batch", K.DEFAULTS["auto_batch"]))
        self.rotate = K.effective_rotate(settings, n)
        self.baseline = K.baseline(settings)
        # Concept mask: each edit only where its own concept words attend (LEDITS++)
        self.mask_on = bool(settings.get("attn_mask", K.DEFAULTS["attn_mask"]))
        self.mask_area = float(settings.get("attn_area", K.DEFAULTS["attn_area"]))
        self.mask_areas = K.mask_areas(plan.concepts, table, self.mask_area)
        self.whole = K.global_scope(plan.concepts, table)    # Pose tab: no mask, no face protection
        self.keepface = K.keep_face(plan.concepts, table, settings)   # ... except Pose energy and Camera height under Protect face
        self.face_policy = K.face_policy(plan.concepts, table, settings)   # full / features / none
        self.face_core = None            # eyes, eyebrows, nose and lips: the core of the face mask
        # where several sliders edit the same pixels, their sum is capped there (overlap budget)
        self.owners = [c.slider for c in plan.concepts]
        self.overlap_budget = float((table or {}).get("overlap_budget", 1.5) or 0)
        self.overlap_warn = float((table or {}).get("overlap_warn", 0.01))     # share of the image
        self.overlap_warned = False
        # absolute cap: the summed edit never past edit_cap x the prompt's own CFG push (per pixel)
        self.edit_cap = float((table or {}).get("edit_cap", 1.0) or 0) if not settings.get("_no_edit_cap") else 0.0
        self.cap_warned = False
        self.attn = [None] * n           # last attention map of each concept, [B, H, W]
        self.blocks = None
        self.orthogonal = K.orthogonal_on(settings)
        self.by_sigma = str(settings.get("range_by", K.DEFAULTS["range_by"])).lower() != "steps"
        self.rescale = float(settings.get("rescale", K.DEFAULTS["rescale"]) or 0.0)
        # Protect face: a probe pass ("<prompt>, face, eyes, eyebrows, nose, lips") whose attention marks the
        # face; every slider edit is zeroed there. Refreshed every `face_every` steps.
        self.protect_face = bool(settings.get("protect_face", K.DEFAULTS["protect_face"]))
        self.face_conds = None
        self.face_mask = None
        self.face_age = 0
        self.face_every = int((table or {}).get("face_every", 4) or 4)
        self.face_threshold = float((table or {}).get("face_threshold", 0.35) or 0.35)
        self.face_max_area = float((table or {}).get("face_max_area", 0.4) or 0.4)
        self.face_core_threshold = float((table or {}).get("face_core_threshold", 0.65) or 0.65)
        # each concept's edit fades out over the last `taper` share of its window
        self.taper = float((table or {}).get("taper", 0.3) or 0.0)
        self.job = None                  # the processing job this engine belongs to
        self.cfg = float(settings.get("_cfg", 0) or 0)
        self.isolated = None             # per-image isolated main conditioning (Isolate words)
        self.sigma_bounds = None
        # one budget only: the overlap budget (per pixel). Momentum is off while concepts take
        # turns, since it would keep pushing edits measured on earlier latents.
        self.sega = Sega(MOMENTUM_SCALE if self.rotate <= 0 else 0.0, MOMENTUM_BETA, budget=0.0)
        self.stale = [None] * n          # last difference of each concept (eps space)
        self.stale_age = [0] * n         # steps since it was computed
        # a kept difference was measured on an earlier latent; applied as is it lands a little
        # off target (ghosting, smearing where several sliders share a body part): it fades
        self.stale_decay = float((table or {}).get("stale_decay", 0.6) or 0.0)
        self.turn = 0                    # Rotate: next group to recompute
        self.chunk_cap = None            # Auto batch: largest chunk that has not run out of memory
        self.label = label
        self.log = log
        self.step, self.total, self.progress = 0, 1, 0.0
        self.calls = 0
        self.passes = 0
        self.seconds = 0.0
        self.off = False
        self.concepts_off = False
        self.notes = set()

    # updated by the WebUI's CFG denoiser callback (scripts/lewd_anatomy.py)
    def on_step(self, step: int, total: int):
        self.step, self.total = int(step), max(int(total), 1)
        self.progress = min(max(self.step / max(self.total - 1, 1), 0.0), 1.0)

    def note(self, text):
        if text not in self.notes:
            self.notes.add(text)
            self.log(f"[Lewd Anatomy] {text}")

    def in_range(self) -> bool:
        """Inside some concept's window (detail sliders may run past the range end)."""
        if self.windows:
            return any(lo <= self.progress <= hi for lo, hi in self.windows)
        return self.start <= self.progress <= self.end

    # ------------------------------------------------------------ what to compute

    def due(self) -> tuple:
        """(active concepts, concepts to predict now) for the current step."""
        active = [i for i, (lo, hi) in enumerate(self.windows) if lo <= self.progress <= hi]
        if not active:
            return [], []
        live_groups = [g for g in self.groups if any(i in active for i in g)]
        # a group's reference must be predicted with it even if only its dependents are active
        if self.rotate <= 0:
            chosen = live_groups
        else:
            # every active concept needs one fresh difference before stale ones can be used
            missing = [g for g in live_groups if any(self.stale[i] is None for i in g if i in active)]
            others = [g for g in live_groups if g not in missing]
            chosen, used = list(missing), sum(len(g) for g in missing)
            if others:
                k = self.turn % len(others)
                order = others[k:] + others[:k]
                for g in order:
                    if chosen and used + len(g) > self.rotate:
                        break
                    chosen.append(g)
                    used += len(g)
                    self.turn += 1
        predict = sorted({i for g in chosen for i in g})
        return active, predict

    # ------------------------------------------------------------ prediction

    def _chunk(self, model, x, n: int) -> int:
        """Concepts per UNet call."""
        if not self.auto_batch:
            return self.per_call
        cap = min(n, 4, self.chunk_cap or 4)
        try:
            if x.device.type != "cuda":
                return max(1, min(self.per_call, cap))
            free, _ = torch.cuda.mem_get_info(x.device)
            batch = x.shape[0]
            for k in range(cap, 1, -1):
                try:
                    need = model.memory_required([batch * k, *x.shape[1:]])
                except TypeError:
                    need = model.memory_required([batch * k, *x.shape[1:]], cond_shapes={})
                if need * 1.25 < free:
                    return k
        except Exception:
            return self.per_call
        return 1

    def _predict(self, args, x, sig, which: list, conds: list | None = None) -> dict:
        """{index: x0 prediction} for `which` (indices into `conds`, the concepts by default),
        chunked, backing off on out-of-memory. Attention is recorded for concepts only."""
        concepts = conds is None
        conds = self.conds if conds is None else conds
        out = {}
        cond_list = args["cond"]
        model = args["model"]
        chunk = self._chunk(model, x, len(which))
        at = 0
        with torch.no_grad():
            while at < len(which):
                part = which[at:at + chunk]
                try:
                    cross, vector = host.stack_concepts([conds[i] for i in part], x.shape[0], x.device)
                    entries = host.concept_entries(cond_list[0], cross, vector)
                    options, recorder = (self._recording_options(args, x, part) if concepts
                                         else (args["model_options"], None))
                    pred = host.predict(model, entries, host.tile(x, len(part)), host.tile(sig, len(part)),
                                        options)
                except Exception as e:
                    if not _is_oom(e) or chunk == 1:
                        raise
                    _free_cache()
                    chunk = max(1, chunk // 2)
                    self.chunk_cap = chunk
                    self.note(f"out of VRAM with {len(part)} concepts per call; now {chunk} per call")
                    continue
                for i, o in zip(part, pred.chunk(len(part))):
                    out[i] = o.detach()
                if recorder is not None:
                    maps = recorder.maps()
                    if maps is not None:
                        for j, i in enumerate(part):
                            m = maps[j * x.shape[0]:(j + 1) * x.shape[0]]
                            self.attn[i] = m if bool(torch.any(m > 0)) else None
                self.calls += 1
                self.passes += len(part)
                at += len(part)
                del pred, cross, vector, entries
        return out

    def _row_tokens(self, i: int, batch: int) -> list:
        c = self.conds[i]
        rows = c if isinstance(c, list) else [c] * batch
        if len(rows) != batch:
            rows = [rows[0]] * batch
        return [r.get("tokens") for r in rows]

    def _recording_options(self, args, x, part):
        """model_options for a concept call: with the attention recorder when the mask is on."""
        if not self.mask_on:
            return args["model_options"], None
        if self.blocks is None:
            self.blocks = A.attention_blocks(getattr(args["model"], "diffusion_model", None))
            if not self.blocks:
                self.note("concept mask: no cross-attention blocks found; edits are not masked")
                self.mask_on = False
        if not self.blocks:
            return args["model_options"], None
        rows = [spec for i in part for spec in self._row_tokens(i, x.shape[0])]
        if not any(rows):
            return args["model_options"], None
        recorder = A.Recorder(rows, tuple(x.shape[-2:]))
        return A.with_recorder(args["model_options"], self.blocks, recorder), recorder

    def release(self):
        """The job is done: drop the per-step tensors (differences, maps, masks, momentum) so they
        do not sit on the GPU until the next job. The plan and conditionings stay (CPU), for
        lib_lewd_anatomy.api.current_patch."""
        n = len(self.plan.concepts)
        self.stale, self.stale_age, self.attn = [None] * n, [0] * n, [None] * n
        self.face_mask = self.face_core = None
        self.isolated = None
        self.sega.reset()
        self.sega.overlap = None

    def _face(self, args, xq, sig):
        """[B, 1, h, w] soft face mask (1 = face), recomputed every `face_every` calls."""
        if not (self.protect_face and self.face_conds):
            return None
        # while a kept-face Pose slider is active the head moves: the mask follows every step
        every = 1 if any(k and lo <= self.progress <= hi for k, (lo, hi) in zip(self.keepface, self.windows)) \
            else self.face_every
        if self.face_mask is not None and self.face_age < every:
            self.face_age += 1
            return self.face_mask
        if self.blocks is None:
            self.blocks = A.attention_blocks(getattr(args["model"], "diffusion_model", None))
        if not self.blocks:
            self.note("protect face: no cross-attention blocks found; face not protected")
            self.protect_face = False
            return None
        batch = xq.shape[0]
        conds = self.face_conds if len(self.face_conds) == batch else [self.face_conds[0]] * batch
        recorder = A.Recorder([c.get("tokens") for c in conds], tuple(xq.shape[-2:]))
        cross, vector = host.stack_concepts([conds], batch, xq.device)
        entries = host.concept_entries(args["cond"][0], cross, vector)
        with torch.no_grad():
            host.predict(args["model"], entries, xq, sig, A.with_recorder(args["model_options"], self.blocks, recorder))
        self.calls += 1
        self.passes += 1
        maps = recorder.maps()
        if maps is None:
            self.note("protect face: no attention recorded for the face words; face not protected")
            self.protect_face = False
            return None
        maps = maps / maps.flatten(1).amax(1).clamp_min(1e-12).view(-1, 1, 1)
        # above face_threshold of the peak, but never more than face_max_area of the image:
        # a flat map (blurry or tiny face) must not switch every edit off
        cap = torch.quantile(maps.flatten(1).float(), 1.0 - self.face_max_area, dim=1).view(-1, 1, 1)
        if bool((cap > self.face_threshold).any()):
            self.note(f"protect face: the face words attend to a large area; the mask is capped at "
                      f"{self.face_max_area:.0%} of the image")
        face = (maps >= torch.clamp(cap, min=self.face_threshold)).float()[:, None]
        # the core: where "face, eyes, eyebrows, nose, lips" peak, the features; only lightly softened.
        # Facial hair and skin sliders are kept off it and may act on the rest of the face
        core = (maps >= torch.clamp(cap, min=self.face_core_threshold)).float()[:, None]
        self.face_core = F.avg_pool2d(core, 3, stride=1, padding=1, count_include_pad=False).clamp(0, 1).to(xq.device)
        grow = max(3, int(round(min(face.shape[-2:]) / 16)) | 1)     # take in hairline, jaw and ears
        face = F.max_pool2d(face, grow, stride=1, padding=grow // 2)
        face = F.avg_pool2d(face, grow, stride=1, padding=grow // 2, count_include_pad=False)
        self.face_mask = face.clamp(0, 1).to(xq.device)
        self.face_age = 1
        return self.face_mask

    def _cap_to_prompt(self, denoised, out, sig4, eps_c, eps_u):
        """Edit cap: at every pixel the summed slider edit stays within edit_cap x the prompt's
        own CFG push there (smoothed; with a floor of a quarter of its mean, so flat areas can
        still be edited). Past what the model itself pushes, an edit leaves what it can draw: a
        strong push on a small part (Penis -2 on a neon scene) drew a glowing ring."""
        edit = (denoised.float() - out.float()) / sig4
        push = self.cfg * (eps_c - eps_u).float()
        pn = push.pow(2).sum(1, keepdim=True).sqrt()
        k = max(3, int(round(min(pn.shape[-2:]) / 16)) | 1)
        pn = F.avg_pool2d(pn, k, stride=1, padding=k // 2, count_include_pad=False)
        floor = 0.25 * pn.flatten(1).mean(1).view(-1, *([1] * (pn.ndim - 1)))
        limit = self.edit_cap * torch.maximum(pn, floor)
        en = edit.pow(2).sum(1, keepdim=True).sqrt()
        factor = (limit / en.clamp_min(1e-12)).clamp(max=1.0)
        if not self.cap_warned and float((factor < 0.6).float().mean()) > self.overlap_warn:
            self.cap_warned = True
            self.note(f"⚠ edit cap: pushes past the model's own held to {float(factor.min()):.0%} "
                      f"over {float((factor < 0.6).float().mean()):.0%} of the image")
        return (denoised.float() - sig4 * edit * factor).to(denoised.dtype)

    def _taper(self, i: int) -> float:
        """1 inside concept i's window, fading to 0 over its last `taper` share: no abrupt
        edits in the low-noise steps, where they would smear fine detail."""
        if self.taper <= 0 or not self.windows:
            return 1.0
        lo, hi = self.windows[i]
        span = (hi - lo) * self.taper
        return 1.0 if span <= 0 else min(1.0, max(0.0, (hi - self.progress) / span))

    def _mask(self, i: int, device):
        """Soft mask of concept i (and its reference, for a contrastive pair), [B, 1, H, W]."""
        maps = [self.attn[i]]
        r = self.refs[i]
        if r is not None:
            maps.append(self.attn[r])
        maps = [m / m.flatten(1).amax(1).clamp_min(1e-12).view(-1, 1, 1) for m in maps if m is not None]
        if not maps:
            return None
        attn = torch.stack(maps).amax(0)
        mask = A.soft_mask(attn, self.mask_areas[i] if i < len(self.mask_areas) else self.mask_area).unsqueeze(1)
        return mask.to(device)

    # ------------------------------------------------------------ hook

    def _progress_from_sigma(self, args, sig):
        """Range by noise level: 0 at the schedule's largest sigma, 1 at its smallest, in
        log-sigma. The same start / end then means the same noise on every sampler and
        scheduler (steps-based ranges move with Karras / exponential / beta)."""
        if self.sigma_bounds is None:
            model = args.get("model")
            bounds = None
            for src in (getattr(model, "model_sampling", None), getattr(model, "predictor", None)):
                try:
                    hi, lo = float(src.sigma_max), float(src.sigma_min)
                    if hi > lo > 0:
                        bounds = (math.log(hi), math.log(lo))
                        break
                except Exception:
                    continue
            self.sigma_bounds = bounds or False
            if not bounds:
                self.note("range by sigma: the model has no sigma schedule; using steps")
        if not self.sigma_bounds:
            return
        hi, lo = self.sigma_bounds
        s0 = float(sig.reshape(-1)[0])
        if s0 > 0:
            self.progress = min(max((hi - math.log(s0)) / (hi - lo), 0.0), 1.0)

    def post_cfg(self, args):
        denoised = args["denoised"]
        if self.off or not (self.plan.concepts or self.rescale > 0):
            return denoised
        try:
            t0 = time.perf_counter()
            x = args["input"]
            sig = torch.as_tensor(args["sigma"], device=x.device).reshape(-1)
            if sig.numel() == 1:
                sig = sig.expand(x.shape[0]).contiguous()
            if self.by_sigma:
                self._progress_from_sigma(args, sig)
            sig4 = sig.float().reshape(-1, *([1] * (x.ndim - 1)))
            uncond, cond = args.get("uncond_denoised"), args.get("cond_denoised")
            has_u = _usable(uncond, denoised)
            has_c = _usable(cond, denoised)
            eps_u = (x.float() - uncond.float()) / sig4 if has_u else None
            eps_c = (x.float() - cond.float()) / sig4 if has_c else None
            result = denoised
            if self.plan.concepts and self.in_range() and not self.concepts_off:
                try:
                    result = self._concepts(args, x, sig, sig4, result, eps_u, eps_c)
                except Exception as e:  # the slider edits stop; anti-burn and keep apart go on
                    self.concepts_off = True
                    result = denoised
                    self.note(f"slider edits turned off for this pass: {type(e).__name__}: {e}")
            if self.rescale > 0 and has_u and has_c:
                result = cfg_rescale(x, sig, result, cond, self.rescale)
            if result is not denoised:
                self.seconds += time.perf_counter() - t0
            return result
        except Exception as e:  # never break a generation
            self.off = True
            self.note(f"turned off for this pass: {type(e).__name__}: {e}")
            return denoised

    def _concepts(self, args, x, sig, sig4, denoised, eps_u, eps_c):
        """The slider edits (SEGA), for one step inside the range."""
        # Baseline "Prompt" (default) is the user's prompt prediction, not the negative one.
        # (eps_edit - eps_uncond) for a short body phrase points at a stock photo
        # of that phrase, which is usually clothed — shirts and underwear appear
        # on a character the prompt left bare. (eps_edit - eps_prompt) cancels
        # everything the prompt already decided, clothes included.
        # Baseline "Negative" is classic SEGA: eps_concept - eps_uncond, with the bare phrase.
        eps_base = eps_u if self.baseline == "Negative" else eps_c
        if eps_base is None:
            if self.baseline == "Negative":
                self.note("baseline Negative: no negative prediction on this step (CFG 1, or skipped by "
                          "another extension); edit skipped there. Use baseline Prompt at CFG 1")
            else:
                self.note("no prompt prediction on this step; edit skipped there")
            return denoised
        if not (args.get("cond") or []):
            return denoised
        n = len(self.conds)
        active, predict = self.due()
        if not active:
            return denoised
        # Perp-Neg: an edit's part along the prompt's own guidance (eps_c - eps_u) is
        # "more (or less) of the prompt"; only the part perpendicular to it is the concept.
        direction = (eps_c - eps_u) if (self.orthogonal and eps_c is not None and eps_u is not None) else None
        x0 = self._predict(args, x, sig, predict)
        eps = {i: (x.float() - p.float()) / sig4 for i, p in x0.items()}
        face = self._face(args, x, sig)
        diffs = [None] * n
        for i in active:
            if i in eps:
                r = self.refs[i]
                if r is None or r in eps:
                    d = eps[i] - (eps_base if r is None else eps[r])
                    if direction is not None and self.scales[i] != 0:
                        d = perpendicular(d, direction)
                    whole = i < len(self.whole) and self.whole[i]
                    policy = self.face_policy[i] if i < len(self.face_policy) else "full"
                    if self.mask_on and self.scales[i] != 0 and not whole:
                        m = self._mask(i, d.device)
                        if m is None:
                            if any(self._row_tokens(i, x.shape[0])):
                                self.note(f"concept mask: no attention for '{self.plan.concepts[i].text}'; "
                                          "that edit is not masked")
                        else:
                            d = d * m.to(d.dtype)
                    if face is not None and self.scales[i] != 0 and policy != "none":
                        f = self.face_core if (policy == "features" and self.face_core is not None) else face
                        f = f if tuple(f.shape[-2:]) == tuple(d.shape[-2:]) else F.interpolate(
                            f, size=d.shape[-2:], mode="bilinear", align_corners=False)
                        d = d * (1.0 - f.to(d.dtype))
                    self.stale[i] = d
                    self.stale_age[i] = 0
                    diffs[i] = d
                    continue
            # Rotate: the difference from the last time it ran, fading with its age
            if self.stale[i] is not None:
                self.stale_age[i] += 1
                diffs[i] = self.stale[i] * (self.stale_decay ** self.stale_age[i])
            else:
                diffs[i] = None
        for i in active:
            if diffs[i] is not None:
                t = self._taper(i)
                diffs[i] = diffs[i] * t if t < 1.0 else diffs[i]
        live = [w is None or self.progress >= w for w in self.warmups]
        out = self.sega.apply_diffs(x, sig, denoised, diffs, self.scales, [self.locality] * n, live,
                                    self.owners, self.overlap_budget)
        if self.edit_cap > 0 and out is not denoised and eps_c is not None and eps_u is not None and self.cfg > 1.0:
            out = self._cap_to_prompt(denoised, out, sig4, eps_c, eps_u)
        ov = self.sega.overlap
        if ov and not self.overlap_warned and ov["cut_share"] > self.overlap_warn:
            # once per pass: a spot where several sliders meet and the cap had to bite hard
            self.overlap_warned = True
            self.note(f"⚠ heavy overlap: {ov['max_sliders']} sliders on one spot, cut to {ov['min_factor']:.0%} "
                      f"over {ov['cut_share']:.0%} of the image")
        return out

    def apply_isolation(self, params):
        """CFG denoiser callback: swap the main prompt's conditioning for the isolated one."""
        if not self.isolated:
            return
        tc = getattr(params, "text_cond", None)
        cross = tc.get("crossattn") if isinstance(tc, dict) else tc
        if not torch.is_tensor(cross) or cross.shape[0] != len(self.isolated):
            self.note("keep apart: the prompt's conditioning does not match (AND, prompt editing or "
                      "a batch layout); not applied")
            self.isolated = None
            return
        if any(iso.shape[1:] != cross.shape[1:] for iso in self.isolated):
            self.note("keep apart: token length differs from the WebUI's encoding; not applied")
            self.isolated = None
            return
        new = torch.cat(self.isolated).to(device=cross.device, dtype=cross.dtype)
        if isinstance(tc, dict):
            tc["crossattn"] = new
        else:
            params.text_cond = new

    def summary(self) -> str:
        return (f"{self.label}: {len(self.plan.concepts)} concept(s), {self.passes} concept pass(es) in "
                f"{self.calls} UNet call(s), {self.seconds:.1f} s")


def _is_oom(e: Exception) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", None) or getattr(torch, "OutOfMemoryError", None)
    return (oom is not None and isinstance(e, oom)) or "out of memory" in str(e).lower()


def _free_cache():
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _table():
    try:
        return K.load_table()
    except Exception:
        return {}


CURRENT: Engine | None = None


def on_cfg_denoiser(params):
    # The callback is global: a nested job sampling while this one is still open
    # (ADetailer's face repaint runs in postprocess_image) must not be touched.
    job = getattr(getattr(params, "denoiser", None), "p", None)
    if CURRENT is not None and CURRENT.job is not None and job is not None and job is not CURRENT.job:
        return
    if CURRENT is not None:
        CURRENT.on_step(getattr(params, "sampling_step", 0) or 0, getattr(params, "total_sampling_steps", 1) or 1)
        try:
            CURRENT.apply_isolation(params)
        except Exception as e:
            CURRENT.note(f"keep apart: {type(e).__name__}: {e}; not applied")
            CURRENT.isolated = None


def _usable(t, like) -> bool:
    """A prediction the WebUI really computed (Forge passes zeros for a skipped pass)."""
    return (torch.is_tensor(t) and t.shape == like.shape and bool(torch.any(t != 0))
            and bool(torch.isfinite(t).all()))


# --------------------------------------------------------------------- encoding

def _nbytes(item: dict) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in (item.get("crossattn"), item.get("vector"))
               if t is not None)


def clear_cache():
    """Drop every cached concept conditioning (RAM)."""
    global _cache_bytes, _cache_key
    _cond_cache.clear()
    _cache_bytes, _cache_key = 0, None


def encode(p, texts: list, width: int, height: int) -> list:
    """Concept conditionings with the model's own text encoder(s), shaped like the main
    conditioning (SDXL: crossattn + vector, with this width / height in the size embedding).
    Cached on the CPU by (checkpoint, text, width, height)."""
    global _cache_bytes, _cache_key
    key = host.checkpoint_key(p)
    if key != _cache_key:
        # another checkpoint: its entries can never be used again until it is loaded back
        _cond_cache.clear()
        _cache_bytes, _cache_key = 0, key
    found = {t: _cond_cache[(key, t, width, height)] for t in dict.fromkeys(texts)
             if (key, t, width, height) in _cond_cache}
    missing = [t for t in dict.fromkeys(texts) if t not in found]
    if missing:
        from modules import prompt_parser
    for at in range(0, len(missing), ENCODE_BATCH):     # small batches: long prompts x many concepts
        part = missing[at:at + ENCODE_BATCH]
        sd = prompt_parser.SdConditioning(part, width=width, height=height)
        with torch.no_grad(), _autocast():
            c = p.sd_model.get_learned_conditioning(sd)
        for i, t in enumerate(part):
            if isinstance(c, dict):
                item = {"crossattn": c["crossattn"][i:i + 1].detach().cpu(),
                        "vector": c["vector"][i:i + 1].detach().cpu() if c.get("vector") is not None else None}
            else:
                item = {"crossattn": c[i:i + 1].detach().cpu(), "vector": None}
            size = _nbytes(item)
            while _cond_cache and (len(_cond_cache) >= _CACHE_LIMIT or _cache_bytes + size > _CACHE_BYTES):
                _cache_bytes -= _nbytes(_cond_cache.pop(next(iter(_cond_cache))))
            _cond_cache[(key, t, width, height)] = item
            _cache_bytes += size
            found[t] = item      # this call's own texts, even if the cache evicted them already
    return [found[t] for t in texts]


def _autocast():
    try:
        from modules import devices
        return devices.autocast()
    except Exception:
        import contextlib
        return contextlib.nullcontext()


# --------------------------------------------------------------------- install

def _scene_prompt(p) -> str:
    """Every prompt in the pass, for Who / scene detection only. Never written back."""
    raw = getattr(p, "prompts", None)
    if raw:
        return "\n".join(str(x or "") for x in raw)
    return str(getattr(p, "prompt", "") or "")


def _edit_prompts(p, hr: bool = False) -> list:
    """One prompt per image of the batch about to be sampled (hires prompts on the hires pass).

    Each image's concepts are appended to that image's own prompt. Using the first
    prompt for every image (as before) meant an image whose prompt differs (a
    batch from Dynamic Prompts, prompts from file, a hires prompt) was compared
    with another scene: the difference, clothes included, became part of the edit.
    """
    raw = (getattr(p, "hr_prompts", None) if hr else None) or getattr(p, "prompts", None)
    if raw:
        return [str(x or "") for x in raw]
    return [str(getattr(p, "prompt", "") or "")]


def _seeds(p, n: int) -> list:
    """One seed per image of the batch about to be sampled (picks a Pose style variant)."""
    seeds = [s for s in (getattr(p, "seeds", None) or []) if s is not None]
    if len(seeds) >= n:
        return [int(s) for s in seeds[:n]]
    first = int(getattr(p, "seed", 0) or 0)
    return [first + i for i in range(n)]


def _is_unet(unet) -> bool:
    dm = getattr(getattr(unet, "model", None), "diffusion_model", None)
    return dm is not None and hasattr(dm, "middle_block") and hasattr(dm, "output_blocks")


def _size(p, hr):
    if hr:
        return int(getattr(p, "hr_upscale_to_x", 0) or p.width), int(getattr(p, "hr_upscale_to_y", 0) or p.height)
    return int(p.width), int(p.height)


_SCHEDULED = re.compile(r"\[[^\]]*:[^\]]*\]|\bAND\b")


def _isolator(p, settings, width, height, log):
    """text -> isolated crossattn ([1, T, D]) or None; None when Isolate words is off."""
    words = I.parse_words(settings.get("isolate", ""))
    strength = float(settings.get("isolate_strength", K.DEFAULTS["isolate_strength"]) or 0)
    if not words or strength <= 0:
        return None
    engine = host.text_engine(p.sd_model)
    if engine is None:
        log("[Lewd Anatomy] keep apart: no tokenizer found on this model; not applied")
        return None

    def one(text):
        if _SCHEDULED.search(text or ""):
            return None                               # AND / [from:to:when] change the encoding per step
        try:
            return I.isolate(text, words, strength, lambda t: encode(p, [t], width, height)[0], engine)
        except Exception as e:
            log(f"[Lewd Anatomy] keep apart: {type(e).__name__}: {e}; not applied")
            return None
    return one


def prepare(p, settings: dict, table: dict, hr: bool = False, log=print):
    """Plan and encoded concepts for this pass (the plan may have no concepts)."""
    profile = K.resolve_profile(settings.get("profile"), host.checkpoint_name(p), table)
    plan = K.plan(settings, table, profile, _scene_prompt(p))
    for n in plan.notes:
        log(f"[Lewd Anatomy] {n}")
    if not plan.concepts:
        return plan, []
    width, height = _size(p, hr)
    prompts = _edit_prompts(p, hr)
    seeds = _seeds(p, len(prompts))
    texts = [[K.concept_text(settings, prompt, K.variant(c, seed)) for prompt, seed in zip(prompts, seeds)]
             for c in plan.concepts]
    flat = encode(p, [t for row in texts for t in row], width, height)
    engine = host.text_engine(p.sd_model) if settings.get("attn_mask", K.DEFAULTS["attn_mask"]) else None
    if settings.get("attn_mask", K.DEFAULTS["attn_mask"]) and engine is None:
        log("[Lewd Anatomy] concept mask: no tokenizer found on this model; edits are not masked")
    isolate_one = _isolator(p, settings, width, height, log) if K.baseline(settings) == "Prompt" else None
    conds, at = [], 0
    for c, row in zip(plan.concepts, texts):
        items = []
        for full, item in zip(row, flat[at:at + len(row)]):
            item = dict(item)                     # the cache entry itself stays untouched
            if engine is not None and c.text:
                item["tokens"] = mask_tokens(engine, full, c, table)
            if isolate_one is not None:
                iso = isolate_one(full)           # measured against the isolated prompt, so isolate too
                if iso is not None:
                    item["crossattn"] = iso
            items.append(item)
        conds.append(items if len(set(row)) > 1 else items[0])
        at += len(row)
    return plan, conds


def mask_tokens(engine, full: str, c, table: dict):
    """Token positions the concept's mask is read from: its slider's 'mask_words' that occur in
    its text (pubic, armpit, chest...), else every token of the concept. A bare 'hair' attends
    to the hair on the head, so the Hair tab's masks took in head and face."""
    words = K._conf(table, c.slider).get("mask_words") or []
    found = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", c.text, re.I)]
    positions, length = [], 0
    for w in found:
        got = host.concept_token_positions(engine, full, w)
        if got:
            positions += got[0]
            length = got[1]
    if positions:
        return sorted(set(positions)), length
    return host.concept_token_positions(engine, full, c.text)


FACE_WORDS = ("face", "eyes", "eyebrows", "nose", "lips")


def prepare_face(p, settings: dict, hr: bool, width: int, height: int, log=print):
    """Probe conditionings for Protect face: each image's prompt + "face, eyes, eyebrows, nose, lips", with
    the positions of those face words. None when the tokenizer is not available."""
    engine = host.text_engine(p.sd_model)
    if engine is None:
        log("[Lewd Anatomy] protect face: no tokenizer found on this model; face not protected")
        return None
    out = []
    for prompt in _edit_prompts(p, hr):
        text = K.edit_prompt(prompt, ", ".join(FACE_WORDS))
        positions, length = [], 0
        for word in FACE_WORDS:
            found = host.concept_token_positions(engine, text, word)
            if found:
                positions += found[0]
                length = found[1]
        item = dict(encode(p, [text], width, height)[0])
        item["tokens"] = (sorted(set(positions)), length) if positions else None
        out.append(item)
    return out if any(o.get("tokens") for o in out) else None


def patch_unet(unet, plan, conds, settings, label="pass", log=print, table=None, isolated=None):
    """A clone of `unet` with SEGA as a post-CFG function; the new Engine becomes CURRENT."""
    global CURRENT
    if CURRENT is not None and CURRENT.calls:
        log(f"[Lewd Anatomy] {CURRENT.summary()}")
    if CURRENT is not None:
        CURRENT.release()                   # the previous pass is over (first pass -> hires pass)
    engine = Engine(plan, conds, settings, label=label, log=log, table=table)
    engine.isolated = isolated
    clone = unet.clone()
    clone.set_model_sampler_post_cfg_function(engine.post_cfg)
    CURRENT = engine
    return clone, engine


def install(p, settings: dict, table: dict, log=print):
    """Patch p's UNet for this pass. Returns the Engine, or None if nothing was installed."""
    hr = bool(getattr(p, "is_hr_pass", False))
    unet = p.sd_model.forge_objects.unet
    if not _is_unet(unet):
        log("[Lewd Anatomy] needs a UNet model (SD 1.x / 2.x / SDXL family); skipped")
        _release(log)
        return None
    settings = {**settings, "_cfg": float(getattr(p, "hr_cfg", 0) or p.cfg_scale) if hr else float(p.cfg_scale)}
    plan, conds = prepare(p, settings, table, hr=hr, log=log)
    slow = K.time_warning(plan.concepts, settings, table)
    if slow:
        log(f"[Lewd Anatomy] {slow}")
    width, height = _size(p, hr)
    isolated = None
    isolate_one = _isolator(p, settings, width, height, log)
    if isolate_one is not None:
        prompts = _edit_prompts(p, hr)
        if any(_SCHEDULED.search(t) for t in prompts):
            log("[Lewd Anatomy] keep apart: AND or prompt editing ([from:to:when]) in the prompt; not applied")
        else:
            rows = [isolate_one(t) for t in prompts]
            if any(r is not None for r in rows):
                # an image whose prompt has no isolate word keeps its normal encoding
                isolated = [r if r is not None else encode(p, [t], width, height)[0]["crossattn"]
                            for r, t in zip(rows, prompts)]
            else:
                log("[Lewd Anatomy] keep apart: none of them is in the prompt; not applied")
    if not (plan.concepts or isolated or float(settings.get("rescale", 0) or 0) > 0):
        _release(log)
        return None
    clone, engine = patch_unet(unet, plan, conds, settings, label="hires pass" if hr else "pass", log=log,
                               table=table, isolated=isolated)
    engine.job = p
    local = [f != "none" and c.scale > 0 for f, c in zip(K.face_policy(plan.concepts, table, settings), plan.concepts)]
    if any(local) and settings.get("protect_face", K.DEFAULTS["protect_face"]):
        engine.face_conds = prepare_face(p, settings, hr, width, height, log)
    p.sd_model.forge_objects.unet = clone
    # for lib_lewd_anatomy.api.current_patch (Impact ADetailer opt-in)
    p._lewd_anatomy_job = {"plan": plan, "conds": conds, "settings": dict(settings)}
    return engine


def _release(log=print):
    """Nothing installed for this pass: the previous engine must not go on applying its keep-apart
    conditioning (it still matches the job) to this pass."""
    global CURRENT
    if CURRENT is not None and CURRENT.calls:
        log(f"[Lewd Anatomy] {CURRENT.summary()}")
    if CURRENT is not None:
        CURRENT.release()
    CURRENT = None


def finish():
    global CURRENT
    engine, CURRENT = CURRENT, None
    if engine is not None:
        engine.release()
    return engine
