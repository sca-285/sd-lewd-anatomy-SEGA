"""Slider values -> SEGA concepts, profiles, scene detection and infotext.

Pure Python (no torch, no WebUI imports) so it can be unit tested anywhere.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_PATH = os.path.join(ROOT, "data", "concepts.json")

WHO_CHOICES = ("Everyone", "Women", "Men")
WHO_CODES = {"Everyone": "e", "Women": "w", "Men": "m"}
WHO_FROM_CODE = {v: k for k, v in WHO_CODES.items()}
WHO_SCENE = {"Women": "women", "Men": "men"}
PROFILE_CHOICES = ("Auto", "Realistic", "Anime", "Stylized")
# What a concept's edit is measured against:
#   Prompt   - the concept is appended to your prompt and compared with your prompt's own
#              prediction; clothes / pose / background cancel (default).
#   Negative - classic SEGA (the paper): the bare concept phrase compared with the
#              negative / unconditional prediction. Stronger, but a bare phrase can drag
#              in a default outfit, and it does nothing at CFG 1 (no negative pass).
BASELINE_CHOICES = ("Prompt", "Negative")
RANGE_CHOICES = ("Sigma", "Steps")
# Who wins when the prompt and a slider disagree (as ControlNet's control modes):
#   Balanced - Strength as set; an edit's part along the prompt's own guidance is dropped (Perp-Neg).
#   Prompt   - the same, edits x 0.7: the prompt has the last word.
#   Sliders  - edits x 1.3, nothing dropped: a slider may overrule the prompt ("large breasts" vs -2).
CONTROL_CHOICES = ("Balanced", "Prompt", "Sliders")
CONTROL_CODES = {"Balanced": "balanced", "Prompt": "prompt", "Sliders": "sliders"}
_CONTROL_OLD = {"my prompt is more important": "prompt", "sega is more important": "sliders", "sega": "sliders"}

DEFAULTS = {
    "enabled": False,
    "profile": "Auto",
    "strength": 1.0,
    "locality": 0.90,       # with the concept mask placing each edit, a little less pixel selection
    "start": 0.10,          # in log-sigma: about where 15 % / 70 % of Karras steps were
    "end": 0.60,
    "allow": False,
    "hires": False,
    "baseline": "Prompt",
    "control": "Balanced",
    "max_concepts": 6,      # most expensive knob: each concept is one more UNet pass per step
    "attn_mask": True,      # each edit only where its concept words attend: keeps face, background
    "protect_face": True,   # a face probe; every slider edit is zeroed on the face
    "attn_area": 0.2,       # share of the image a concept's mask covers (before feathering)
    "rescale": 0.0,         # CFG rescale phi (anti-burn at high CFG); 0 = off
    "isolate": "",          # Keep apart: words kept inside their own comma phrase; "" = off
    # fixed, not in the UI (2.0): kept as settings so the engine and tests can reach them
    "per_call": 1,          # concepts per extra model call; 1 keeps the peak below a CFG step
    "auto_batch": True,     # raise per_call while VRAM is free, halve it on out-of-memory
    "rotate": 0,            # 0 = automatic: concepts take turns only above Max concepts
    "staggered": True,      # shape sliders early in the range, detail sliders late
    "orthogonal": True,     # Perp-Neg in Balanced / Prompt; Sliders turns it off
    "range_by": "Sigma",    # Range start / end in noise level (same on every sampler)
    "isolate_strength": 0.6,
}
DEFAULT_PHASES = {"shape": [0.0, 0.8], "detail": [0.3, 1.15]}
STEPS = 2                   # every slider: -2 -1 0 +1 +2


# --------------------------------------------------------------------- table

_cache = {"path": None, "mtime": None, "table": None}


def load_table(path: str = TABLE_PATH) -> dict:
    """The concept table, re-read when the file changes on disk."""
    mtime = os.path.getmtime(path)
    if _cache["path"] != path or _cache["mtime"] != mtime:
        with open(path, encoding="utf-8") as f:
            _cache.update(path=path, mtime=mtime, table=json.load(f))
    return _cache["table"]


def sliders(table: dict) -> list:
    return table["sliders"]


def slider_by_key(table: dict) -> dict:
    return {s["key"]: s for s in (table or {}).get("sliders", [])}


def one_sided(s: dict) -> bool:
    """A slider with no 'below' steps (Anatomy fix): 0, +1, +2 only."""
    return not s.get("below")


def slider_range(s: dict) -> tuple:
    """(minimum, maximum) of the UI slider: -2..+2, 0 neutral; 0..+2 for a one-sided slider."""
    return (0 if one_sided(s) else -STEPS), STEPS


def step(value) -> int:
    """A slider value as one of -2, -1, 0, +1, +2."""
    try:
        v = int(round(float(value or 0)))
    except (TypeError, ValueError):
        return 0
    return max(-STEPS, min(STEPS, v))


def slider_step(s: dict, value) -> int:
    """step(), kept inside the slider's own range (a one-sided slider has no - side)."""
    lo, hi = slider_range(s)
    return max(lo, min(hi, step(value)))


def step_names(s: dict) -> dict:
    """{-2: name, -1: name, 1: name, 2: name} from the slider's 'below' and 'levels'
    (only {1, 2} for a one-sided slider)."""
    below, levels = list(s.get("below") or []), list(s["levels"])
    if len(below) not in (0, STEPS) or len(levels) != STEPS:
        raise ValueError(f"slider {s['key']!r} needs {STEPS} 'levels' and {STEPS} (or no) 'below' names")
    names = {1: levels[0], 2: levels[1]}
    if below:
        names.update({-2: below[0], -1: below[1]})
    return names


def level_name(s: dict, value) -> str | None:
    """The level a slider value stands for, or None when neutral."""
    v = slider_step(s, value)
    return None if v == 0 else step_names(s)[v]


def level_value(s: dict, name: str):
    """Inverse of level_name; accepts a level name or a number. None if unknown."""
    name = str(name).strip().lower()
    if name in ("", "off", "neutral", "0"):
        return 0
    for v, n in step_names(s).items():
        if n.lower() == name:
            return v
    for alias, v in (s.get("aliases") or {}).items():     # step names of an earlier version
        if alias.lower() == name:
            return slider_step(s, v)
    try:
        v = int(round(float(name)))
    except (ValueError, OverflowError):
        return None
    return slider_step(s, v)


def entry(raw) -> dict | None:
    """Normalise one concept entry."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"text": raw}
    variants = [" ".join(str(v).split()) for v in (raw.get("variants") or []) if str(v).strip()]
    text = str(raw.get("text", "") or (variants[0] if variants else "")).strip()
    if not text:
        return None
    direction = raw.get("direction", "toward")
    if direction not in ("toward", "away"):
        raise ValueError(f"direction must be 'toward' or 'away', not {direction!r}")
    out = {"text": text, "direction": direction, "scale": float(raw.get("scale", 1.0))}
    if raw.get("unless"):
        out["unless"] = list(raw["unless"])     # regions: skip this entry while a slider there is moved
    if raw.get("warmup") is not None:
        out["warmup"] = float(raw["warmup"])
    if raw.get("ref"):
        out["ref"] = True                       # a measuring point only: no push toward it
    if len(variants) > 1:
        out["variants"] = tuple(variants)       # one per image, picked by its seed (Pose style)
    return out


def entries(raw) -> list:
    """A step's concept entries: one entry, or a list of them (Anatomy fix pushes away
    from several separate things; one long phrase would blur them together)."""
    if isinstance(raw, list):
        return [e for e in (entry(r) for r in raw) if e is not None]
    e = entry(raw)
    return [e] if e is not None else []


def profile_concepts(table: dict, s: dict, profile: str) -> dict:
    """The slider's concept entries for a profile, following the profile's fallback."""
    seen = set()
    while profile and profile not in seen:
        seen.add(profile)
        found = s["concepts"].get(profile)
        if found is not None:
            return found
        profile = table["profiles"].get(profile, {}).get("fallback")
    return {}


# --------------------------------------------------------------------- profile and scene

def resolve_profile(setting: str, checkpoint_name: str, table: dict) -> str:
    """'Auto' picks from the checkpoint name; otherwise the chosen profile (lower case)."""
    setting = (setting or "Auto").strip().lower()
    setting = table.get("profile_aliases", {}).get(setting, setting)
    if setting != "auto":
        return setting if setting in table["profiles"] else "realistic"
    name = (checkpoint_name or "").lower()
    for rule in table.get("auto_profile", []):
        if re.search(rule["pattern"], name):
            return rule["profile"]
    return "realistic"


def step_gain(table: dict, value) -> float:
    """How hard a step pushes: ±2 is a steady step up from ±1, not a jump."""
    gains = table.get("step_gain") or {"1": 0.6, "2": 1.2}
    v = abs(step(value))
    return float(gains.get(str(v), 0.0)) if v else 0.0


def scene_genders(prompt: str, table: dict) -> set:
    """Which of 'women' / 'men' the user's prompt mentions (read only, never edited)."""
    text = (prompt or "").lower().replace("_", " ")
    found = set()
    for who, pattern in table.get("scene_tokens", {}).items():
        if re.search(pattern, text):
            found.add(who)
    return found


def scene_key(scene: set) -> str:
    """'women', 'men' or 'any': which of a 'by_scene' slider's tables a prompt uses. Futa read as
    women and cuntboys as men (their build); a scene with both, or neither, is 'any'."""
    women = bool({"women", "futa"} & scene)
    men = bool({"men", "cuntboy"} & scene)
    return "women" if women and not men else "men" if men and not women else "any"


def variant(c, seed) -> str:
    """The text one image uses for concept c: its seed picks one of c's variants."""
    if not c.variants:
        return c.text
    return c.variants[int(seed or 0) % len(c.variants)]


def edit_prompt(user_prompt: str, concept: str) -> str:
    """The text actually encoded for one concept.

    A bare phrase such as "medium natural breasts" or "masculine man" is, to the
    checkpoint, a stock photo — very often a clothed one. Steering from the
    unconditional prediction toward that phrase paints the outfit on. Encoding
    the phrase *after the user's prompt* and guiding against the prompt
    prediction (see engine) keeps the scene: clothes, pose and background are
    in both predictions and cancel, and only the anatomy moves.
    """
    concept = " ".join((concept or "").split()).strip().strip(",")
    user = " ".join((user_prompt or "").split()).strip().strip(",")
    if not concept:
        return user
    if not user:
        return concept
    if concept.lower() in user.lower():
        return user
    return f"{user}, {concept}"


def toward_allowed(who: str | None, scene: set) -> bool:
    """Toward pushes are global in v1; they are only safe if nobody else is in the scene.

    A futa / shemale tag means a woman has male anatomy, so Men sliders (penis,
    testicles, foreskin) are allowed on that body; a cuntboy tag is the mirror case,
    Women sliders (vulva) on a man.
    """
    if who in (None, "Everyone"):
        return True
    genders = set(scene) - {"futa", "cuntboy"}
    if who == "Men" and "futa" in scene:
        return genders <= {"women", "men"}
    if who == "Women" and "cuntboy" in scene:
        return genders <= {"women", "men"}
    return genders <= {WHO_SCENE[who]}


# --------------------------------------------------------------------- planning

@dataclass
class Concept:
    slider: str
    text: str
    sign: float          # +1 toward, -1 away
    scale: float         # s_e (>= 0; 0 = only computed as another concept's reference)
    warmup: float | None = None
    ref: str | None = None   # text of the (toward) concept this one is measured against; None = baseline
    phase: str | None = None  # 'shape' / 'detail' (Staggered ranges) / 'fix'; None = the whole range
    role: str | None = None   # 'toward' / 'opposite' / 'cap' / 'away' (planning only)
    variants: tuple | None = None   # texts one of which each image uses, by its seed; text = the first

    @property
    def signed_scale(self) -> float:
        return self.sign * self.scale


@dataclass
class Plan:
    profile: str
    concepts: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def plan(settings: dict, table: dict, profile: str, prompt: str = "") -> Plan:
    """The concepts to compute for these slider values.

    settings: {"strength", "allow", "max_concepts", "sliders": {key: value}, "who": {key: who}}

    A step v of a slider computes at most three concepts:
      * toward the entry named by v (or away, if that entry is written as 'away'),
      * away from the first step on the other side (-1 for +v, +1 for -v), which is
        where the checkpoint drifts by default,
      * on the + side, away from the 'cap' unless Allow exaggerated.
    Their strength grows with |v| through step_gain, the same for every slider.
    """
    roles = table["roles"]
    base = float(table["base_scale"]) * float(settings.get("strength", 1.0))
    allow = bool(settings.get("allow", False))
    scene = scene_genders(prompt, table)
    mixed = {"women", "men"} <= scene
    out = Plan(profile=profile)
    if base <= 0:
        return out

    def add(slider, e, sign, factor, ref=None, keep_zero=False, role=None):
        if e is None or (factor <= 0 and not keep_zero):
            return
        out.concepts.append(Concept(slider, e["text"], sign, base * e["scale"] * max(factor, 0.0),
                                    e.get("warmup"), ref, phase_of.get(slider), role, e.get("variants")))

    phase_of = {s["key"]: s.get("phase") for s in table["sliders"]}
    moved_regions = {s.get("region") for s in table["sliders"]
                     if s.get("region") and slider_step(s, (settings.get("sliders") or {}).get(s["key"], 0))}
    for s in table["sliders"]:
        key = s["key"]
        v = slider_step(s, (settings.get("sliders") or {}).get(key, 0))
        if v == 0:
            continue
        if mixed and s.get("mixed_scene") == "off":
            out.notes.append(f"• {s['label']}: off (women + men)")
            continue
        names = step_names(s)
        found = profile_concepts(table, s, profile)
        if found and s.get("by_scene"):
            # the same step reads differently on a man and on a woman (Pose style)
            found = found.get(scene_key(scene)) or {}
        if not found:
            out.notes.append(f"• {s['label']}: off in {profile} (no tag)")
            continue
        who = (settings.get("who") or {}).get(key) or s.get("who")
        toward_ok = toward_allowed(who, scene)
        gain = step_gain(table, v)

        # Away pushes are contrastive whenever the step has a 'toward' entry:
        # measured from the step's own concept, not from the bare baseline.
        # "prompt + small breasts" vs "prompt + large breasts" differ only in size;
        # "prompt" vs "prompt + large breasts" also differs in how much breast is
        # shown, and pushing away from that is pushing toward covering it up.
        raw = found.get(names[v])
        if isinstance(raw, list):
            # several separate concepts on one step (Anatomy fix): each on its own
            for e in entries(raw):
                if moved_regions & set(e.get("unless") or ()):
                    # e.g. Anatomy fix "away from extra breasts" on a chest a Breasts slider is
                    # shaping: the two would fight over the same pixels and smear them
                    out.notes.append(f"• {s['label']}: skips {e['text']}")
                    continue
                add(key, e, -1.0 if e["direction"] == "away" else 1.0, roles["away"] * gain, role="away")
            continue
        target = entry(raw)
        ref = None
        if target is not None:
            if target["direction"] == "away":
                add(key, target, -1.0, roles["away"] * gain, role="away")
            else:
                ref = target["text"]
                # kept at scale 0 when Who blocks the toward push: still the reference.
                # A 'ref' step (-1 "sparse chest hair") is only where the push is measured from:
                # pushed toward, it would add sparse hair to a hairless chest; measured from, it
                # takes hair away from a hairy one and leaves a hairless one alone
                push = toward_ok and not target.get("ref")
                add(key, target, 1.0, roles["toward"] * gain if push else 0.0, keep_zero=True, role="toward")
        # Only contrastive: without a toward concept to measure from, "away from
        # pectorals" is away from showing a chest at all, which is how clothes come back.
        opposite = entry(found.get(names.get(-1 if v > 0 else 1)))
        if ref is not None and opposite is not None and opposite["direction"] == "toward":
            # from a 'ref' step the opposite push is the whole edit: at toward strength
            role_gain = roles["toward"] if target.get("ref") else roles["opposite"]
            add(key, opposite, -1.0, role_gain * gain, ref=ref, role="opposite")
        if v > 0 and not allow:
            cap = entry(found.get("cap"))
            if cap is not None:
                add(key, cap, -1.0, roles["cap"] * gain, ref=ref, role="cap")

        if not toward_ok and who:
            out.notes.append(f"• {s['label']}: {who} only, push away only (women + men)")

    # a cap must not fight another slider's goal: Hips + is capped away from "huge ass",
    # which is exactly what Butt +2 asks for
    def tags(text):
        return {t.strip().lower() for t in text.split(",") if t.strip()}
    wanted = {}
    for c in out.concepts:
        if c.role == "toward" and c.scale > 0:
            wanted.setdefault(c.slider, set()).update(tags(c.text))
    out.concepts = [c for c in out.concepts if not (c.role == "cap" and any(
        tags(c.text) & w for k, w in wanted.items() if k != c.slider))]
    moved = [s for s in table["sliders"] if slider_step(s, (settings.get("sliders") or {}).get(s["key"], 0))]
    if len(moved) >= 10:
        out.notes.append(f"⚠ {len(moved)} sliders: capped where they overlap")
    lock = table.get("nudity_lock") or {}
    if lock and not settings.get("_no_lock") and nudity_lock(prompt):
        e = entry(((lock.get("concepts") or {}).get(profile)
                   or (lock.get("concepts") or {}).get("realistic")))
        if e is not None:
            out.concepts.append(Concept(NUDITY_LOCK, e["text"], -1.0, base * e["scale"] * float(lock.get("gain", 0.6)),
                                        None, None, lock.get("phase", "shape"), "away"))
            out.notes.append("• Nudity lock: on")
    by_region = {}
    for s in moved:
        if s.get("region"):
            # a fuse group is one description: it counts once
            if not s.get("spots"):          # scars, moles... dot the skin, they do not pile up
                by_region.setdefault(s["region"], set()).add(s.get("fuse") or s["key"])
    for region, labels in by_region.items():
        if len(labels) >= int(table.get("region_warn", 4)):
            out.notes.append(f"⚠ {len(labels)} on {region}: capped, may smear")
    clashing = clashes(settings, table)
    if clashing:
        # Two sliders that pull the same body part opposite ways: each slider's 'opposite'
        # push (away from the other side of its own scale) is exactly what the other slider
        # moves toward, so the two fight twice over the same pixels. Keep both toward
        # pushes, drop the opposite pushes of both; the overlap budget limits the rest.
        involved = {k for pair in clashing for k in pair}
        out.concepts = [c for c in out.concepts if not (c.slider in involved and c.role == "opposite")]
        by_key = slider_by_key(table)
        for a, b in clashing:
            out.notes.append(f"⚠ {by_key[a]['label']} vs {by_key[b]['label']}: clash, softened")
    out.concepts, fused_notes = _fuse(out.concepts, table)
    out.notes += fused_notes
    if baseline(settings) == "Negative" and nudity_context(prompt):
        # Negative baseline: each concept is led by the prompt's nudity words. An away
        # concept measured against the negative prediction would then push away from
        # nudity itself (= toward clothes). Measure it against the nudity words alone.
        loose = [c for c in out.concepts if c.sign < 0 and c.ref is None]
        if loose:
            for c in loose:
                c.ref = NUDITY_REF
            out.concepts.append(Concept("", NUDITY_REF, 1.0, 0.0))
    out.concepts = _drop_unused_refs(_merge(out.concepts))
    limit = int(settings.get("max_concepts", DEFAULTS["max_concepts"]) or 0)
    if limit > 0 and len(out.concepts) > limit and int(settings.get("rotate", 0) or 0) <= 0:
        out.concepts, shed = _shed(out.concepts, limit)
        if shed:
            out.notes.append(f"• Max {limit}: {shed} caps / counter-pushes dropped")
    if limit > 0 and len(out.concepts) > limit and int(settings.get("rotate", 0) or 0) <= 0:
        # nothing is dropped: above Max concepts they take turns (Rotate = Max), so the
        # cost per step stays at about Max passes and every slider still acts
        out.notes.append(f"• {len(out.concepts)} concepts > Max {limit}: take turns")
    return out


FUSE_PREFIX = "fuse:"


def _fuse(concepts: list, table: dict) -> tuple:
    """Sliders that describe one body part (a slider's 'fuse' group: breasts + breast shape;
    nipple size and melanin; penis and erection; veins, wetness and foreskin...) become ONE
    concept per role, within one phase, when two or more of them are moved: "large breasts, perky breasts",
    "large areolae, erect nipples, brown nipples". A checkpoint draws one coherent part from
    one description; three separate edits on the same few pixels drew a smeared or doubled
    nipple. Strength: the strongest member's, not the sum. Members whose toward push is
    blocked (Who) or that only push away stay on their own."""
    by_key = slider_by_key(table)
    groups = {}
    for c in concepts:
        conf = by_key.get(c.slider) or {}
        g = conf.get("fuse")
        if g and c.role == "toward" and c.scale > 0:
            # within one phase only: a size push (shape) fused with veins (detail) would run on
            # into the last, low-noise steps and smear the part
            groups.setdefault(f"{g}@{conf.get('phase') or 'shape'}", []).append(c)
    out, notes = list(concepts), []
    for g, towards in groups.items():
        members = []
        for c in towards:
            if c.slider not in members:
                members.append(c.slider)
        if len(members) < 2:
            continue
        member_texts = {c.text for c in towards}
        fused_key = FUSE_PREFIX + g
        phase = g.split("@", 1)[1]

        def joined(cs):
            seen = []
            for c in cs:
                for part in (p.strip() for p in c.text.split(",")):
                    if part and part not in seen:
                        seen.append(part)
            return ", ".join(seen)

        toward = Concept(fused_key, joined(towards), 1.0, max(c.scale for c in towards),
                         None if any(c.warmup is None for c in towards) else min(c.warmup for c in towards),
                         None, phase, "toward")
        new = [toward]
        for role in ("opposite", "cap"):
            group = [c for c in out if c.slider in members and c.role == role and c.ref in member_texts]
            if group:
                new.append(Concept(fused_key, joined(group), -1.0, max(c.scale for c in group), None,
                                   toward.text, phase, role))
        drop = {id(c) for c in out if c.slider in members and (c.role == "toward" and c.scale > 0
                                                               or c.role in ("opposite", "cap") and c.ref in member_texts)}
        out = [c for c in out if id(c) not in drop] + new
        labels = [by_key[k]["label"] for k in members]
        notes.append(f"• Fused: {' + '.join(labels)}")
    return out, notes


def clashes(settings: dict, table: dict) -> list:
    """[(slider a, slider b)] of the table's 'clashes' that the current values hit:
    e.g. Belly + with Abs + (a pot belly and a six-pack on the same belly)."""
    values = settings.get("sliders") or {}
    by_key = slider_by_key(table)
    out = []
    for rule in table.get("clashes", []):
        a, b = rule["sliders"]
        if a not in by_key or b not in by_key:
            continue
        va, vb = slider_step(by_key[a], values.get(a, 0)), slider_step(by_key[b], values.get(b, 0))
        if va == 0 or vb == 0:
            continue
        sa, sb = (1 if va > 0 else -1), (1 if vb > 0 else -1)
        if [sa, sb] in rule["signs"]:
            out.append((a, b))
    return out


NUDITY_LOCK = "nudity-lock"


def _conf(table: dict, key: str) -> dict:
    """The table entry behind a concept: its slider, or the nudity lock block."""
    if key == NUDITY_LOCK:
        return table.get("nudity_lock") or {}
    if key.startswith(FUSE_PREFIX):
        group, _, phase = key[len(FUSE_PREFIX):].partition("@")
        members = [s for s in table.get("sliders", []) if s.get("fuse") == group
                   and (not phase or (s.get("phase") or "shape") == phase)]
        conf = {"region": next((s.get("region") for s in members if s.get("region")), None),
                "phase": phase or None}
        words = []
        for s in members:
            for w in s.get("mask_words") or []:
                if w not in words:
                    words.append(w)
        if words:
            conf["mask_words"] = words
        waits = sorted({w for s in members for w in (s.get("wait_for") or [])})
        if waits:
            conf["wait_for"] = waits
        areas = [float(s["area"]) for s in members if s.get("area")]
        if areas:
            conf["area"] = max(areas)
        faces = {s.get("face") for s in members}
        if faces == {"features"}:
            conf["face"] = "features"
        return conf
    return slider_by_key(table).get(key) or {}


def face_policy(concepts: list, table: dict, settings: dict) -> list:
    """How each concept's edit treats the face, with Protect face on:
      'full'     - zeroed on the whole face (body sliders, Pose energy, Camera height);
      'features' - zeroed only on eyes, eyebrows, nose and lips, the core of the face mask (Facial hair,
                   Skin realism, Cum: they have to reach jaw, chin, cheeks, forehead, hair);
      'none'     - not at all (Head turn: turning the face is its job).
    """
    keep = keep_face(concepts, table, settings)
    out = []
    for c, k in zip(concepts, keep):
        conf = _conf(table, c.slider)
        if conf.get("scope") == "global" and not k:
            out.append("none")
        else:
            out.append("features" if conf.get("face") == "features" else "full")
    return out


def global_scope(concepts: list, table: dict) -> list:
    """Per concept: True for a slider with 'scope': 'global' (the Pose tab). A pose or a
    camera angle is the whole picture: no concept mask, and no Protect face (a face held
    in place while the body turns would tear)."""
    return [_conf(table, c.slider).get("scope") == "global" for c in concepts]


def keep_face(concepts: list, table: dict, settings: dict) -> list:
    """Per concept: a Pose slider marked 'keep_face' (Pose energy, Camera height) while Protect
    face is on. Its edit is zeroed on the face like a body slider's; the face mask follows the
    head every step. Head turn is never one of them: turning the face is its job."""
    if not settings.get("protect_face", DEFAULTS["protect_face"]):
        return [False] * len(concepts)
    return [bool(_conf(table, c.slider).get("keep_face")) for c in concepts]


def mask_areas(concepts: list, table: dict, default: float) -> list:
    """Concept mask area per concept: the slider's own 'area' (Anatomy fix: small, on the
    hands and limbs only) or the global Mask area."""
    return [float(_conf(table, c.slider).get("area") or default) for c in concepts]


def effective_rotate(settings: dict, n_concepts: int) -> int:
    """Rotate as set; or, when it is 0 and there are more concepts than Max concepts,
    Max concepts (the concepts take turns instead of being left out)."""
    r = int(settings.get("rotate", 0) or 0)
    if r > 0:
        return r
    limit = int(settings.get("max_concepts", DEFAULTS["max_concepts"]) or 0)
    return limit if (limit > 0 and n_concepts > limit) else 0


def _ref_index(concepts: list) -> dict:
    return {c.text: i for i, c in enumerate(concepts) if c.sign > 0}


def ref_indices(concepts: list) -> list:
    """For each concept, the index of its reference concept, or None (= the baseline)."""
    by_text = _ref_index(concepts)
    return [by_text.get(c.ref) if c.ref is not None else None for c in concepts]


def _drop_unused_refs(concepts: list) -> list:
    """A scale-0 concept is only worth its UNet pass if something is measured against it."""
    used = {c.ref for c in concepts if c.ref is not None and c.scale > 0}
    return [c for c in concepts if c.scale > 0 or (c.sign > 0 and c.text in used)]


def _earliest(a, b):
    """Of two warmups, the earlier; None (live from the first step) wins."""
    return None if a is None or b is None else min(a, b)


def _shed(concepts: list, limit: int) -> tuple:
    """Above Max concepts, before anything takes turns: leave out what detail sliders can best
    do without, first their caps, then their counter-pushes ('opposite'), then shape sliders'
    caps (a cap only matters past the size a step asks for). Every slider is then
    still computed on every step: two hair sliders (3 concepts each) no longer take turns at
    Max 6, which reused stale, fading differences and blurred both. A counter-push that IS
    the edit (from a 'ref' step, whose toward push is 0) stays."""
    out = list(concepts)
    shed = 0
    for role, phases in (("cap", ("detail",)), ("opposite", ("detail",)), ("cap", ("shape", "fix"))):
        while len(out) > limit:
            pick = [c for c in out if c.role == role and c.phase in phases and c.scale > 0
                    and not (role == "opposite" and any(t.text == c.ref and t.scale == 0 for t in out))]
            if not pick:
                break
            out.remove(min(pick, key=lambda c: c.scale))
            out = _drop_unused_refs(out)
            shed += 1
    return out, shed


def _merge(concepts: list) -> list:
    """One concept per (text, direction, phase): identical texts would cost an extra forward
    each. The strongest scale, not the sum (as a fused part): Breasts -2 and Pectorals -2 both
    asking for a flat chest push once, not twice as hard. Different phases stay apart."""
    merged = {}
    for c in concepts:
        k = (c.text, c.sign, c.ref, c.phase)
        if k in merged:
            m = merged[k]
            m.scale = max(m.scale, c.scale)
            m.warmup = _earliest(m.warmup, c.warmup)
        else:
            merged[k] = Concept(c.slider, c.text, c.sign, c.scale, c.warmup, c.ref, c.phase, c.role, c.variants)
    return list(merged.values())


def expected_extra(concepts: list, settings: dict, table: dict) -> float:
    """Expected extra time as a fraction (2.5 = +250 %): concept passes per step over the whole
    schedule, next to the 2 passes of a CFG step."""
    if not concepts:
        return 0.0
    # over the whole schedule: detail windows run past the range end, Pose ones start before it
    wins = windows(concepts, settings, table)
    rotate = effective_rotate(settings, len(concepts))
    samples, total = 101, 0.0
    for k in range(samples):
        t = k / (samples - 1)
        n = sum(1 for lo, hi in wins if lo <= t <= hi)
        total += min(n, rotate) if rotate > 0 else n
    return total / samples / 2.0


def time_warning(concepts: list, settings: dict, table: dict) -> str | None:
    """A warning when the expected extra time passes `warn_time` (default +800 %)."""
    extra = expected_extra(concepts, settings, table)
    if extra * 100 >= float(table.get("warn_time", 8.0)) * 100:
        return f"⚠ Slow: +{extra * 100:.0f} % time"
    return None


def slowdown(n_concepts: int, start: float, end: float, cfg_passes: int = 2) -> float:
    """Expected extra time as a fraction: n concept rows next to the cfg_passes rows of a step,
    over the share of steps between start and end."""
    if n_concepts <= 0:
        return 0.0
    share = max(0.0, min(1.0, float(end)) - max(0.0, float(start)))
    return n_concepts / float(max(cfg_passes, 1)) * share


# --------------------------------------------------------------------- infotext

def _num(v) -> str:
    return f"{float(v):.2f}".rstrip("0").rstrip(".") if float(v) != int(float(v)) else f"{float(v):.1f}"


def to_infotext(settings: dict, table: dict, resolved_profile: str) -> str:
    """'profile=anime, strength=1.0, breasts=medium(w), v-line=defined, range=0.15-0.70'"""
    profile = resolved_profile
    if (settings.get("profile") or "Auto").lower() == "auto":
        profile = f"auto({resolved_profile})"
    parts = [f"profile={profile}", f"strength={_num(settings.get('strength', 1.0))}"]
    for s in table["sliders"]:
        level = level_name(s, (settings.get("sliders") or {}).get(s["key"], 0))
        if level is None:
            continue
        item = f"{s['key']}={level}"
        who = (settings.get("who") or {}).get(s["key"]) or s.get("who")
        if s.get("who") and who in WHO_CODES:
            item += f"({WHO_CODES[who]})"
        parts.append(item)
    parts.append(f"locality={_num(settings.get('locality', DEFAULTS['locality']))}")
    parts.append(f"range={float(settings.get('start', DEFAULTS['start'])):.2f}-{float(settings.get('end', DEFAULTS['end'])):.2f}")
    if baseline(settings) != DEFAULTS["baseline"]:
        parts.append(f"baseline={baseline(settings).lower()}")
    limit = int(settings.get("max_concepts", DEFAULTS["max_concepts"]) or 0)
    if limit != DEFAULTS["max_concepts"]:
        parts.append(f"max={limit}")
    if control_mode(settings) != "balanced":
        parts.append(f"control={control_mode(settings)}")
    if float(settings.get("rescale", 0) or 0) > 0:
        parts.append(f"rescale={_num(settings['rescale'])}")
    words = [w for w in re.split(r"[,|\n]", str(settings.get("isolate", "") or "")) if w.strip()]
    if words:
        parts.append("isolate=" + "|".join(" ".join(w.split()) for w in words))
    if not settings.get("protect_face", DEFAULTS["protect_face"]):
        parts.append("face=off")
    if not settings.get("attn_mask", DEFAULTS["attn_mask"]):
        parts.append("mask=off")
    elif abs(float(settings.get("attn_area", DEFAULTS["attn_area"])) - DEFAULTS["attn_area"]) > 1e-9:
        parts.append(f"area={_num(settings['attn_area'])}")
    if settings.get("allow"):
        parts.append("exaggerated=on")
    if settings.get("hires"):
        parts.append("hires=on")
    return ", ".join(parts)


_ITEM = re.compile(r"^\s*([\w-]+)\s*=\s*([^()]*?)\s*(?:\(\s*([a-z]+)\s*\))?\s*$")


def from_infotext(text: str, table: dict) -> dict:
    """Parse to_infotext's output back into settings (enabled, every slider present)."""
    by_key = slider_by_key(table)
    out = {**DEFAULTS, "enabled": True,
           "sliders": {k: 0 for k in by_key},
           "who": {k: s.get("who") for k, s in by_key.items() if s.get("who")}}
    text = (text or "").strip().strip('"')
    for raw in text.split(","):
        try:
            _read_item(raw, out, by_key, table)
        except (ValueError, OverflowError):
            continue                     # one malformed item does not spoil the paste
    return out


def _read_item(raw: str, out: dict, by_key: dict, table: dict) -> None:
    if raw.strip().lower().startswith("isolate="):
        # keep-apart words may hold prompt weights: "(red hair:1.2)|blue eyes"
        value = raw.split("=", 1)[1]
        out["isolate"] = ", ".join(w.strip() for w in value.split("|") if w.strip())
        return
    m = _ITEM.match(raw)
    if not m:
        return
    key, value, extra = m.group(1).lower(), m.group(2), m.group(3)
    if key == "profile":
        p = value.lower()
        p = table.get("profile_aliases", {}).get(p, p)     # old "pony" -> anime
        if p.startswith("auto"):
            out["profile"] = "Auto"
        else:
            out["profile"] = next((c for c in PROFILE_CHOICES if c.lower() == p), "Auto")
        # "auto(anime)": the regex leaves the resolved profile in `extra`
    elif key == "strength":
        out["strength"] = float(value)
    elif key == "locality":
        out["locality"] = float(value)
    elif key == "range":
        a, _, b = value.partition("-")
        out["start"], out["end"] = float(a), float(b)
    elif key == "baseline":
        out["baseline"] = next((c for c in BASELINE_CHOICES if c.lower() == value.lower()), DEFAULTS["baseline"])
    elif key == "control":
        code = control_mode({"control": value})
        out["control"] = next(c for c, k in CONTROL_CODES.items() if k == code)
    elif key == "max":
        out["max_concepts"] = int(float(value))
    elif key == "rescale":
        out["rescale"] = float(value)
    elif key == "isolate":
        out["isolate"] = ", ".join(w.strip() for w in value.split("|") if w.strip())
    elif key == "face":
        out["protect_face"] = value.lower() not in ("off", "false", "0", "no")
    elif key == "mask":
        out["attn_mask"] = value.lower() not in ("off", "false", "0", "no")
    elif key == "area":
        out["attn_area"] = float(value)
    elif key == "exaggerated":
        out["allow"] = value.lower() in ("on", "true", "1", "yes")
    elif key == "hires":
        out["hires"] = value.lower() in ("on", "true", "1", "yes")
    elif key in by_key:
        v = level_value(by_key[key], value)
        if v is not None:
            out["sliders"][key] = v
        if extra and extra in WHO_FROM_CODE and by_key[key].get("who"):
            out["who"][key] = WHO_FROM_CODE[extra]


def control_mode(settings: dict) -> str:
    """'balanced', 'prompt' or 'sliders' (also accepts the codes and the 1.x labels)."""
    v = str(settings.get("control") or DEFAULTS["control"]).strip()
    if v in CONTROL_CODES:
        return CONTROL_CODES[v]
    v = v.lower()
    v = _CONTROL_OLD.get(v, v)
    return v if v in ("prompt", "sliders") else "balanced"


def control_gain(table: dict, mode: str) -> float:
    """The one thing a Control mode scales: every edit (table 'control': {mode: gain})."""
    defaults = {"balanced": 1.0, "prompt": 0.7, "sliders": 1.3}
    return float(((table or {}).get("control") or {}).get(mode, defaults.get(mode, 1.0)))


def orthogonal_on(settings: dict) -> bool:
    """Perp-Neg in Balanced and Prompt; off in Sliders (a slider may then overrule the prompt)."""
    if control_mode(settings) == "sliders":
        return False
    return bool(settings.get("orthogonal", DEFAULTS["orthogonal"]))


def baseline(settings: dict) -> str:
    b = str(settings.get("baseline") or DEFAULTS["baseline"]).strip().capitalize()
    return b if b in BASELINE_CHOICES else DEFAULTS["baseline"]


NUDITY = re.compile(
    r"\b(completely nude|fully nude|full nudity|nude|naked|nudity|topless|bottomless|"
    r"no clothes|unclothed|undressed|bare skin|nsfw|explicit|uncensored|nipples|pussy|vulva|penis|testicles)\b",
    re.I)


NUDITY_REF = ""      # concept text of the nudity-only reference (encodes to the nudity words alone)


_FULL_NUDE = re.compile(r"\b(completely nude|fully nude|full nudity|stark naked|naked|nude|nudity|no clothes|"
                        r"unclothed|undressed)\b(?![\s-]*(lipstick|makeup|nails|stockings|pantyhose|heels|"
                        r"colou?r|tone|bodysuit|dress|bra|panties|leotard))", re.I)
# partial nudity, or garments / clothed people in the prompt: clothes are meant somewhere
_KEEP_CLOTHES = re.compile(
    r"\b(partially nude|partial nudity|semi[- ]?nude|(?:half|nearly|almost|mostly)[- ]?(?:nude|naked)|topless|bottomless|shirtless|"
    r"bare[- ]chested|clothed|dressed|wearing|clothes|clothing|outfit|shirt|t-shirt|blouse|dress|skirt|"
    r"pants|jeans|shorts|panties|bra|underwear|lingerie|bikini|swimsuit|suit|uniform|jacket|coat|hoodie|"
    r"sweater|apron|kimono|cmnf|cfnm|open clothes|clothes lift|clothes pull|stockings|thighhighs|"
    r"pantyhose|socks|gloves|heels|shoes|boots|hat|choker|collar|garter|garter belt)\b", re.I)
# "no clothes", "without panties", "wearing nothing": asks for less, not for a garment
_NEGATED = re.compile(r"\b(?:no|without)\s+[\w-]+|\bwearing\s+(?:only\s+)?nothing\b|\bbirthday suit\b", re.I)


def nudity_lock(prompt: str) -> bool:
    """Full nudity asked for, nothing to keep on: 'nude' / 'completely nude' / 'naked' with no
    partial-nudity word and no garment or clothed person in the prompt. Then the plan pushes
    away from clothes. topless, shirtless, bare-chested, a skirt, a clothed man (CMNF)... leave
    the clothes alone."""
    text = (prompt or "").replace("_", " ")
    return bool(_FULL_NUDE.search(text)) and not _KEEP_CLOTHES.search(_NEGATED.sub(" ", text))


def nudity_context(user_prompt: str) -> str:
    """The prompt's own nudity words, in order, deduplicated: 'completely nude, nipples'."""
    seen, out = set(), []
    for m in NUDITY.finditer((user_prompt or "").replace("_", " ")):
        w = m.group(1).lower()
        if w not in seen:
            seen.add(w)
            out.append(w)
    return ", ".join(out)


def concept_text(settings: dict, user_prompt: str, concept: str) -> str:
    """The text encoded for a concept under the chosen baseline.

    Prompt: your prompt + the concept (everything else cancels against your prompt).
    Negative: the concept phrase, led by the prompt's own nudity words. A bare
    "small breasts" is, to these checkpoints, a clothed stock photo; measured
    against the negative prediction that clothing is part of the edit.
    """
    concept = " ".join((concept or "").split()).strip().strip(",")
    if baseline(settings) == "Negative":
        context = nudity_context(user_prompt)
        return f"{context}, {concept}" if context and concept else (concept or context)
    return edit_prompt(user_prompt, concept)


def model_calls(n_concepts: int, per_call: int) -> int:
    """Extra UNet calls per step for n concepts, per_call concepts at a time."""
    per_call = max(1, int(per_call or 1))
    return -(-int(n_concepts) // per_call)


def windows(concepts: list, settings: dict, table: dict) -> list:
    """(start, end) progress window of each concept.

    'detail' sliders (hair, abs, V-line, back muscles, anus...) always reach past the
    range end, up to the phase's upper fraction (1.15 of the range by default, clipped
    at 1): strands and fine texture form in the last, low-noise steps. With Staggered
    ranges, 'shape' sliders also keep to the early part of the range. A reference
    shared by several phases (the nudity reference) spans every phase that uses it.
    """
    a, b = float(settings.get("start", DEFAULTS["start"])), float(settings.get("end", DEFAULTS["end"]))
    phases = table.get("phases") or DEFAULT_PHASES
    staggered = bool(settings.get("staggered", DEFAULTS["staggered"]))

    def win(phase):
        if phase == "pose":
            # Pose, camera and head turn are decided by the composition, in the first steps,
            # before Range start: a fixed window of the whole schedule, not of the range
            lo, hi = phases.get("pose", (0.0, 0.35))
            return float(lo), float(hi)
        if phase == "fix":
            # Anatomy fix never runs in the first, composition-setting steps (moving a limb
            # then re-poses the whole body, face included)
            lo, hi = phases.get("fix", (0.3, 0.85))
            return a + (b - a) * float(lo), min(1.0, a + (b - a) * float(hi))
        if phase == "detail" or (staggered and phase):
            lo, hi = phases.get(phase, (0.0, 1.0))
            if not staggered:
                lo = 0.0
            return a + (b - a) * float(lo), min(1.0, a + (b - a) * float(hi))
        return a, b

    out = [win(c.phase) for c in concepts]
    # In a region where a shape concept is running (Breasts), a detail concept (Nipples) waits
    # until the shape has mostly settled: drawn on a part still changing size it lands off
    # target (smeared or doubled nipples).
    confs = [_conf(table, c.slider) for c in concepts]
    shaped = {f.get("region") for f, c in zip(confs, concepts)
              if f.get("region") and c.phase == "shape" and c.scale > 0}
    any_shape = any(c.phase == "shape" and c.scale > 0 and c.slider != NUDITY_LOCK for c in concepts)
    after = float(phases.get("detail_after_shape", 0.6))
    for i, (f, c) in enumerate(zip(confs, concepts)):
        if c.phase != "detail":
            continue
        waits = set(f.get("wait_for") or [])
        if f.get("region"):
            waits.add(f["region"])
        if waits & shaped or ("*" in waits and any_shape):
            lo, hi = out[i]
            out[i] = (max(lo, a + (b - a) * after), hi)
    # while a Pose slider moves the body, shape sliders wait until the composition mostly
    # settled: a waist or breast edit on a body still turning lands off target
    posed = [w for w, c in zip(out, concepts) if c.phase == "pose" and c.scale > 0]
    if posed:
        settle = max(hi for _, hi in posed) * float(phases.get("shape_after_pose", 0.7))
        for i, c in enumerate(concepts):
            if c.phase == "shape":
                lo, hi = out[i]
                if settle < hi:                  # a short range ending early keeps its own window
                    out[i] = (max(lo, settle), hi)
    for i, c in enumerate(concepts):
        users = [out[j] for j, d in enumerate(concepts) if d.ref is not None and d.ref == c.text and c.sign > 0]
        if c.phase is None and users:
            out[i] = (min(u[0] for u in users), max(u[1] for u in users))
    return out


def groups(concepts: list) -> list:
    """Concepts that must be predicted on the same step: each toward concept with the
    concepts measured against it; a concept measured against the baseline alone."""
    refs = ref_indices(concepts)
    out, at = [], {}
    for i, r in enumerate(refs):
        root = i if r is None else r
        if root not in at:
            at[root] = len(out)
            out.append([])
        out[at[root]].append(i)
    for g in out:
        g.sort()
    return out


def passes_per_step(concepts: list, settings: dict, table: dict, samples: int = 41) -> float:
    """Average UNet passes per step inside the range, after Staggered ranges and Rotate."""
    if not concepts:
        return 0.0
    a, b = float(settings.get("start", DEFAULTS["start"])), float(settings.get("end", DEFAULTS["end"]))
    if b <= a:
        return 0.0
    wins = windows(concepts, settings, table)
    rotate = effective_rotate(settings, len(concepts))
    total = 0.0
    for k in range(samples):
        t = a + (b - a) * k / (samples - 1)
        n = sum(1 for lo, hi in wins if lo <= t <= hi)
        total += min(n, rotate) if rotate > 0 else n
    return total / samples


def extras_active(settings: dict) -> bool:
    """Something to do even with every slider at 0: anti-burn or keep-apart words."""
    return bool(settings.get("enabled")) and (
        float(settings.get("rescale", 0) or 0) > 0 or bool(str(settings.get("isolate", "") or "").strip()))


def active(settings: dict, table: dict, prompt: str = "") -> bool:
    if not settings.get("enabled"):
        return False
    if table.get("nudity_lock") and nudity_lock(prompt):
        return True
    return any(level_name(s, (settings.get("sliders") or {}).get(s["key"], 0)) is not None for s in table["sliders"])
