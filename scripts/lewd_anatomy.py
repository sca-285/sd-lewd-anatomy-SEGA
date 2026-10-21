"""Lewd Anatomy · SEGA Semantic Guidance: body proportion sliders for Forge,
reForge and Forge Classic (Neo). The prompt is never edited."""

import html
import os
import re
import sys

import gradio as gr

from modules import script_callbacks, scripts, shared

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)

from lib_lewd_anatomy import concepts as K  # noqa: E402
from lib_lewd_anatomy import engine  # noqa: E402

try:
    from modules.ui_components import InputAccordion
except Exception:
    InputAccordion = None

try:
    GRADIO4 = int(str(gr.__version__).split(".")[0]) >= 4
except Exception:
    GRADIO4 = False

INFOTEXT_KEY = "Lewd Anatomy"
LEGACY_INFOTEXT_KEYS = ("Physique", "Body Sliders")     # earlier names; still pasted back
LABEL = "Lewd Anatomy · SEGA Semantic Guidance"
TOP_KEYS = ("enabled", "profile", "control", "options", "strength", "start", "end",
            "baseline", "attn_area", "max_concepts", "rescale", "isolate")

# on / off settings, shown as one row of toggles: (label, settings key)
OPTIONS = (("Protect face", "protect_face"), ("Concept mask", "attn_mask"), ("Allow exaggerated", "allow"),
           ("Hires fix", "hires"))
OPTION_KEYS = {label: key for label, key in OPTIONS}


def options_value(settings: dict) -> list:
    return [label for label, key in OPTIONS if settings.get(key, K.DEFAULTS.get(key))]

# slider tabs; a slider added to data/concepts.json and not listed here goes to the last tab
# Tabs in the order the layers run (see the Guide): the composition first, then shapes, then
# details; inside a tab, shape sliders before detail sliders.
TABS = (
    ("Pose", ("pose-energy", "pose-style", "camera-height", "head-turn")),
    ("Body", ("masc-fem", "shoulders", "waist", "hips", "thighs", "butt", "anatomy-fix", "nipples",
                "nipple-color", "anus")),
    ("Muscle", ("muscle", "body-fat", "belly", "lats", "biceps", "abs", "v-line", "back")),
    ("Female", ("breasts", "breast-shape", "vulva", "vulva-puffy", "vulva-texture")),
    ("Male", ("penis", "penis-girth", "testicles", "pectorals", "penis-veins", "scrotum", "foreskin")),
    ("Skin", ("skin-tone", "skin-realism", "flush", "sweat", "cellulite")),
    ("Arousal", ("erection", "nipple-state", "vulva-wetness", "penis-wetness", "anus-wetness", "cum")),
    ("Hair & marks", ("facial-hair", "body-hair", "armpit-hair", "pubic-hair", "limb-hair", "scars", "stretch-marks", "moles",
                      "birthmarks", "freckles", "wrinkles")),
)

GUIDE = """
**Body preset** · fills the body sliders for the chosen Profile (not Pose, Arousal, skin tone) and sets *Mask area*, *Max concepts* (nothing takes turns, up to 16) and *Anti-burn* (from your CFG) to match; tweak any of them after.

**Sliders** · 0 = off · − smaller than the prompt would give · + larger. Don't also write that size in the prompt.

**Pose** · *Pose energy*: how much the body moves · *Pose style*: − powerful, confident · + graceful, sensual, hers or his by the prompt, a different pose per image seed (off with both in the prompt). Exact poses: prompt or ControlNet OpenPose.

**Tabs run left to right** · Pose (composition, first steps) → Body, Muscle, Female, Male (shapes) → Skin, Arousal, Hair & marks (details, last steps). Inside a tab, shapes come first. Skin tone and Erection are shapes too.

**Start with the defaults.** Too weak → raise *Strength*. Size still exaggerated → lower CFG (4 – 6 on SDXL).

**Profile** must match the checkpoint: Realistic (photo), Stylized (2.5D / 3D / cartoon in sentences), Anime (Pony, Illustrious, NoobAI and their merges: tags). Auto reads the checkpoint name.

**Control mode** · *Balanced* as set · *Prompt* gentler (× 0.7), the prompt wins · *Sliders* stronger (× 1.3), a slider may overrule the prompt.

**Protect face** · body sliders leave the whole face alone, Pose energy, Pose style and Camera height too (fewer changes, not none). Facial hair, Skin tone, Skin realism, Cum, the marks and Wrinkles keep off only eyes, eyebrows, nose and lips. Head turn turns it. Same face in every image: IP-Adapter FaceID / PuLID.

**Other options** · *Concept mask*: each edit on its own body part · *Allow exaggerated*: no size cap on + steps · *Hires fix*: sliders in the hires pass too (never Pose).

**Range** · where in the steps the sliders act. Face drifts → start later (0.15 – 0.20).

**Overlap** · sliders of one part are fused into one description; details wait for shapes; where sliders meet, the sum is capped; the status line warns when it gets heavy or slow. Fewer sliders = stronger, cleaner edits.

**Nudity lock** · automatic: *nude* / *naked* with no partial nudity, garment or clothed person pushes clothes away.

**Anatomy fix** · only pushes away from extra fingers, limbs, nipples, genitals. **Cum** lands where the prompt says, face and hair too. Blush on the face is the prompt's.

**Advanced** · *Baseline*: Prompt keeps scene and clothes, Negative is classic SEGA (CFG > 1) · *Mask area*: how much one edit may touch · *Max concepts*: above it they take turns (softer) · *Anti-burn*: 0.5 – 0.7 for CFG 8 – 12 · *Keep apart*: words of one person in a couple, so hair colour or size does not leak.

Full guide: `docs/DETAILS.md`.
"""

def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)


def guide_html() -> str:
    """GUIDE (a little Markdown: paragraphs, **bold**, *italic*, `code`) as a <details> pop-over."""
    paras = [p.strip() for p in GUIDE.strip().split("\n\n") if p.strip()]
    body = "".join(f"<p>{_inline(' '.join(p.split()))}</p>" for p in paras)
    return f'<details class="bs-guide-pop"><summary>Guide</summary><div class="bs-guide-body">{body}</div></details>'


TABLE = K.load_table()
SLIDERS = K.sliders(TABLE)          # the UI is built from the table as it is at startup
BY_KEY = K.slider_by_key(TABLE)


def _load_presets():
    import json
    try:
        with open(os.path.join(EXT_ROOT, "data", "presets.json"), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return [], {}
    scope = [k for k in data.get("scope", []) if k in BY_KEY]
    return scope, {p["name"]: p for p in data.get("presets", [])}


PRESET_SCOPE, PRESETS = _load_presets()
NO_PRESET = "None"


def preset_values(name, variant=None, profile="realistic") -> list:
    """Slider values (in PRESET_SCOPE order) for a preset, variant and profile, or None to leave
    them. The preset's per-profile overrides apply; a slider off on that profile stays 0."""
    p = PRESETS.get(name)
    if not p:
        return None
    v = next((v for v in p["variants"] if v["name"] == variant), p["variants"][0])
    values = {**v["sliders"], **((p.get("profiles") or {}).get(profile) or {})}
    out = []
    for key in PRESET_SCOPE:
        s = BY_KEY[key]
        lo, hi = K.slider_range(s)
        value = max(lo, min(hi, int(values.get(key, 0) or 0)))
        out.append(value if value and K.profile_concepts(TABLE, s, profile) else 0)
    return out


BIG_PARTS = ("shoulders", "waist", "hips", "thighs", "butt", "lats", "back", "belly")


def preset_settings(values: list, profile: str, prompt: str = "", cfg=None) -> tuple:
    """(Mask area, Max concepts, Anti-burn) a preset's slider values call for.
    Mask area: bigger with more big parts moved (waist, hips, shoulders...: they cover more of the
    image than the default 0.2 lets an edit touch). Max concepts: as many as the plan computes, so
    nothing takes turns, 6 to 16. Anti-burn: by CFG (None = leave it), many sliders push harder."""
    sliders = {k: v for k, v in zip(PRESET_SCOPE, values) if v}
    big = sum(1 for k in BIG_PARTS if sliders.get(k))
    area = round(min(0.35, 0.2 + 0.03 * big), 2)
    who = {k: "Everyone" for k in sliders if BY_KEY[k].get("who")}
    plan = K.plan({**K.DEFAULTS, "enabled": True, "sliders": sliders, "who": who, "max_concepts": 0},
                  TABLE, profile, prompt or "")
    max_c = max(int(K.DEFAULTS["max_concepts"]), min(16, len(plan.concepts)))
    rescale = None
    if cfg is not None:
        cfg = float(cfg)
        rescale = 0.0 if cfg <= 7.5 else 0.3 if cfg <= 8.5 else 0.5 if cfg <= 10.5 else 0.7
        if rescale and len(sliders) >= 7:
            rescale = min(0.7, rescale + 0.1)
    return area, max_c, rescale


def _panel_profile(choice) -> str:
    return K.resolve_profile(choice or "Auto", _checkpoint_name(), TABLE)

script_callbacks.on_cfg_denoiser(engine.on_cfg_denoiser)

PROMPTS = {}                        # elem_id -> prompt textbox, for the status line


def _remember_prompt(component, **kwargs):
    eid = kwargs.get("elem_id")
    if eid in ("txt2img_prompt", "img2img_prompt", "txt2img_cfg_scale", "img2img_cfg_scale"):
        PROMPTS[eid] = component


script_callbacks.on_after_component(_remember_prompt)


def _make(cls, *args, **kwargs):
    """Gradio 3.41 and 4: drop layout keywords an older component does not take."""
    try:
        return cls(*args, **kwargs)
    except TypeError:
        for k in ("scale", "min_width", "info", "show_label"):
            kwargs.pop(k, None)
        return cls(*args, **kwargs)


def _no_ui_config(*components):
    """Keep these out of the WebUI's ui-config.json.

    The WebUI saves every slider's minimum / maximum / value there on the first
    start and forces them back on every later start. The first versions of
    these sliders went 0..N; ui-config kept minimum = 0, which is why the new
    -2..+2 sliders could not be moved below 0. Not saving them fixes that for
    good, whatever the table says later.
    """
    for c in components:
        try:
            c.do_not_save_to_config = True
        except Exception:
            pass


# --------------------------------------------------------------------- args <-> settings

def arg_layout():
    """(kind, key) for every UI argument, in the order process() receives them."""
    layout = [("top", k) for k in TOP_KEYS]
    for s in SLIDERS:
        layout.append(("slider", s["key"]))
        if s.get("who"):
            layout.append(("who", s["key"]))
    return layout


LAYOUT = arg_layout()


def from_args(args) -> dict:
    out = {**K.DEFAULTS, "sliders": {}, "who": {}}
    for (kind, key), value in zip(LAYOUT, args):
        if kind == "top" and key == "options":
            if isinstance(value, (list, tuple, set)):
                chosen = set(value)
            elif isinstance(value, str):                     # an API caller sending one label
                chosen = {value}
            else:                                            # anything else: keep the defaults
                chosen = set(options_value(K.DEFAULTS))
            for label, opt in OPTIONS:
                out[opt] = label in chosen
        elif kind == "top":
            out[key] = value
        elif kind == "slider":
            out["sliders"][key] = value
        else:
            out["who"][key] = value
    return out


def _checkpoint_name():
    try:
        return str(shared.opts.sd_model_checkpoint or "")
    except Exception:
        return ""


def status(prompt, *args):
    s = from_args(args)
    table = K.load_table()
    profile = K.resolve_profile(s["profile"], _checkpoint_name(), table)
    detected = f"{profile}" + (" (auto)" if (s["profile"] or "Auto") == "Auto" else "")
    if not s["enabled"]:
        return f"Off · {detected}"
    plan = K.plan(s, table, profile, prompt or "")
    bits = []
    for slider in table["sliders"]:
        value = K.slider_step(slider, (s["sliders"] or {}).get(slider["key"], 0))
        name = K.level_name(slider, value)
        if name:
            bits.append(f"{slider['label']} {value:+d} {name}")
    lines = []
    if not plan.concepts:
        lines.append(f"Neutral · {detected}")
    else:
        passes = K.passes_per_step(plan.concepts, s, table)
        extra = K.expected_extra(plan.concepts, s, table)
        if bits:
            lines.append(" · ".join(bits))
        lines.append(f"{len(plan.concepts)} edits · {passes:.1f} passes/step · +{extra * 100:.0f} % time · "
                     f"{detected} · {K.baseline(s)}"
                     + ("" if K.control_mode(s) == "balanced" else f" · {s['control']}"))
        warn = K.time_warning(plan.concepts, s, table)
        if warn:
            lines.append(warn)
    # warnings first, then the rest; one per line
    notes = sorted(plan.notes, key=lambda n: not n.startswith("⚠"))
    return "  \n".join(lines + notes)



# --------------------------------------------------------------------- script

class LewdAnatomy(scripts.Script):
    sorting_priority = 15           # after Steer (14); the order does not matter for SEGA itself

    def title(self):
        return "Lewd Anatomy"       # also the API name (alwayson_scripts "Lewd Anatomy")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        comps = {}

        def eid(key):
            return f"{tab}_lewdanatomy_{key.replace('-', '_')}"

        container = (InputAccordion(False, label=LABEL, elem_id=eid("accordion")) if InputAccordion
                     else gr.Accordion(LABEL, open=False))
        with container as acc:
            with gr.Column(elem_classes=["lewdanatomy"]):
                if InputAccordion:
                    comps[("top", "enabled")] = acc
                else:
                    comps[("top", "enabled")] = gr.Checkbox(False, label="Enable", elem_id=eid("enabled"))

                # compact grid, every control in its own column (Gradio packs neighbouring inputs
                # into one form block otherwise and leaves holes beside it):
                #   Profile | Guide | Reset all
                #   Control mode (one line)
                #   Options (one line)
                #   Strength | Range start | Range end
                #   Advanced: Baseline | Keep apart, then Mask area | Max concepts | Anti-burn
                def cell(min_width=120, scale=1, classes=None):
                    return gr.Column(min_width=min_width, scale=scale, elem_classes=["bs-cell"] + (classes or []))

                def slider(key, lo, hi, step, label):
                    comps[("top", key)] = gr.Slider(lo, hi, value=K.DEFAULTS[key], step=step, label=label,
                                                    elem_id=eid(key))

                with gr.Column(elem_classes=["bs-head"]):
                    with gr.Row(elem_classes=["bs-row"]):
                        with cell(150, 2):
                            comps[("top", "profile")] = gr.Dropdown(list(K.PROFILE_CHOICES), value="Auto",
                                                                    label="Profile", elem_id=eid("profile"))
                        with cell(110, 1, ["bs-guide-col"]):
                            # Guide over Reset, the height of the Profile box. The Guide is a plain
                            # HTML <details>: it opens and closes in the browser, no Gradio event
                            gr.HTML(guide_html(), elem_id=eid("guide"), elem_classes=["bs-guide-cell"])
                            reset_all = _make(gr.Button, "Reset sliders", elem_id=eid("reset_all"),
                                              elem_classes=["lewdanatomy-reset-all"], size="sm")
                    comps[("top", "control")] = gr.Radio(list(K.CONTROL_CHOICES), value=K.DEFAULTS["control"],
                                                         label="Control mode", elem_id=eid("control"),
                                                         elem_classes=["bs-control", "bs-line"])
                    comps[("top", "options")] = gr.CheckboxGroup(
                        [label for label, _ in OPTIONS], value=options_value(K.DEFAULTS), label="Options",
                        elem_id=eid("options"), elem_classes=["bs-options", "bs-line"])
                with gr.Row(elem_classes=["bs-row"]):
                    with cell():
                        slider("strength", 0.0, 2.0, 0.05, "Strength")
                    with cell():
                        slider("start", 0.0, 1.0, 0.01, "Range start")
                    with cell():
                        slider("end", 0.0, 1.0, 0.01, "Range end")
                gr.HTML('<div class="bs-section-title">Advanced</div>')
                with gr.Row(elem_classes=["bs-row"]):
                    with cell(140):
                        comps[("top", "baseline")] = gr.Radio(list(K.BASELINE_CHOICES), value=K.DEFAULTS["baseline"],
                                                              label="Baseline", elem_id=eid("baseline"),
                                                              elem_classes=["bs-inline"])
                    with cell(140):
                        comps[("top", "isolate")] = _make(
                            gr.Textbox, "", label="Keep apart (couples)", lines=1, max_lines=1,
                            placeholder="blonde, black hair", elem_id=eid("isolate"))
                # Locality is fixed (0.90; X/Y/Z can still sweep it): the concept mask already
                # places every edit, Mask area is the one placement knob
                with gr.Row(elem_classes=["bs-row"]):
                    with cell(120):
                        slider("attn_area", 0.05, 0.6, 0.01, "Mask area")
                    with cell(120):
                        slider("max_concepts", 1, 40, 1, "Max concepts")
                    with cell(120):
                        slider("rescale", 0.0, 1.0, 0.05, "Anti-burn")
                _no_ui_config(*(comps[("top", k)] for k in TOP_KEYS if k != "enabled"))

                resets, all_sliders = [], []

                def add_slider(key):
                    s = BY_KEY.get(key)
                    if s is None:
                        return
                    lo, hi = K.slider_range(s)
                    names = K.step_names(s)
                    info = " · ".join(f"{v:+d} {names[v]}" if v else "0 off" for v in range(lo, hi + 1))
                    with gr.Group(elem_classes=["bs-card"]):
                        with gr.Row(elem_classes=["bs-card-head"]):
                            slider = _make(gr.Slider, lo, hi, value=0, step=1, label=s["label"],
                                           elem_id=eid(s["key"]), scale=5, elem_classes=["bs-slider"])
                            comps[("slider", s["key"])] = slider
                            all_sliders.append(slider)
                            _no_ui_config(slider)
                            if not GRADIO4:      # Gradio 4 sliders have their own reset button
                                reset = _make(gr.Button, "↺", elem_id=eid(f"{s['key']}_reset"),
                                              elem_classes=["lewdanatomy-reset"], scale=0, min_width=32)
                                resets.append((reset, slider))
                        gr.Markdown(info, elem_classes=["bs-legend"])
                        if s.get("note"):
                            gr.Markdown(s["note"], elem_classes=["bs-legend", "bs-note"])
                        if s.get("who"):
                            comps[("who", s["key"])] = _make(
                                gr.Radio, list(K.WHO_CHOICES), value=s["who"], show_label=False,
                                elem_id=eid(f"{s['key']}_who"), elem_classes=["bs-who"])
                            _no_ui_config(comps[("who", s["key"])])

                # body presets: fill the build sliders (not Pose, Arousal, skin tone, marks, grooming)
                preset_dd = preset_var = None
                if PRESETS:
                    with gr.Row(elem_classes=["bs-row"]):
                        with cell(150, 2):
                            preset_dd = gr.Dropdown([NO_PRESET] + list(PRESETS), value=NO_PRESET,
                                                    label="Body preset", elem_id=eid("preset"))
                        with cell(150, 2):
                            # a dropdown, not a radio: variant names are of any length
                            preset_var = gr.Dropdown([], value=None, label="Variant", elem_id=eid("preset_variant"))
                    _no_ui_config(preset_dd, preset_var)

                # sliders: one tab per body area
                listed = {k for _, keys in TABS for k in keys}
                tabs = [(title, list(keys)) for title, keys in TABS]
                tabs[-1][1].extend(s["key"] for s in SLIDERS if s["key"] not in listed)
                with gr.Tabs(elem_id=eid("tabs"), elem_classes=["bs-tabs"]):
                    for title, keys in tabs:
                        with gr.Tab(title, elem_id=eid("tab_" + title.split()[0].lower())):
                            for key in keys:
                                add_slider(key)

                status_md = gr.Markdown(status("", False), elem_id=eid("status"), elem_classes=["bs-status"])

        ordered = [comps[k] for k in LAYOUT]
        for reset, slider in resets:
            reset.click(fn=lambda: 0, inputs=[], outputs=[slider], show_progress=False)
        reset_all.click(fn=lambda: [0] * len(all_sliders), inputs=[], outputs=all_sliders, show_progress=False)
        if preset_dd is not None:
            targets = [comps[("slider", k)] for k in PRESET_SCOPE if ("slider", k) in comps]

            tuned = [comps[("top", k)] for k in ("attn_area", "max_concepts", "rescale")]
            prompt_box = PROMPTS.get(f"{tab}_prompt")
            cfg_box = PROMPTS.get(f"{tab}_cfg_scale")
            extra_in = [c if c is not None else gr.State(None) for c in (prompt_box, cfg_box)]

            def apply(name, variant, profile, prompt, cfg):
                prof = _panel_profile(profile)
                values = preset_values(name, variant, prof)
                if values is None:
                    return [gr.update() for _ in targets + tuned]
                area, max_c, rescale = preset_settings(values, prof, prompt, cfg)
                return values + [area, max_c, gr.update() if rescale is None else rescale]

            def pick_preset(name, profile, prompt, cfg):
                p = PRESETS.get(name)
                if not p:
                    return [gr.update(choices=[], value=None)] + [gr.update() for _ in targets + tuned]
                names = [v["name"] for v in p["variants"]]
                return [gr.update(choices=names, value=names[0])] + apply(name, names[0], profile, prompt, cfg)

            def pick_variant(name, variant, profile, prompt, cfg):
                return apply(name, variant, profile, prompt, cfg)

            profile_dd = comps[("top", "profile")]
            preset_dd.change(pick_preset, inputs=[preset_dd, profile_dd] + extra_in,
                             outputs=[preset_var] + targets + tuned, show_progress=False)
            preset_var.change(pick_variant, inputs=[preset_dd, preset_var, profile_dd] + extra_in,
                              outputs=targets + tuned, show_progress=False)

        prompt = PROMPTS.get(f"{tab}_prompt")
        inputs = ([prompt] if prompt is not None else [gr.State("")]) + ordered
        for c in ordered + ([prompt] if prompt is not None else []):
            c.change(status, inputs=inputs, outputs=[status_md], show_progress=False)

        def paste(kind, key):
            def read(d):
                text = next((d[k] for k in (INFOTEXT_KEY, *LEGACY_INFOTEXT_KEYS) if k in d), None)
                if text is None:
                    return False if (kind, key) == ("top", "enabled") else None
                try:
                    s = K.from_infotext(text, K.load_table())
                except Exception:
                    return None
                if (kind, key) == ("top", "options"):
                    return options_value(s)
                if kind == "top":
                    return s[key]
                if kind == "slider":
                    return s["sliders"].get(key, 0)
                return s["who"].get(key, BY_KEY[key].get("who"))
            return read

        self.infotext_fields = [(comps[k], paste(*k)) for k in LAYOUT]
        self.paste_field_names = [INFOTEXT_KEY, *LEGACY_INFOTEXT_KEYS]
        return ordered

    def process_before_every_sampling(self, p, *args, **kwargs):
        s = from_args(args)
        _apply_xyz(p, s)
        hr = bool(getattr(p, "is_hr_pass", False))
        table = K.load_table()
        typed = s
        if hr:
            s = {**s, "_no_lock": True}   # the first pass has decided the clothes
        if hr and not s["hires"]:
            s = {**s, "sliders": {}}      # "Apply in Hires fix" is about the sliders only
        elif hr:
            # the composition is set by the first pass: no Pose sliders in the hires pass
            s = {**s, "sliders": {k: v for k, v in (s["sliders"] or {}).items()
                                  if (BY_KEY.get(k) or {}).get("scope") != "global"}}
        tiler = tiled_script(p)
        if tiler:
            # SD upscale / Ultimate SD upscale run img2img on every tile with these
            # settings: a tile with half a body would still be pushed toward the concept,
            # and the nudity lock would push on every tile
            if K.active(s, table):
                print(f"[Lewd Anatomy] {tiler}: sliders are off on tiles (anti-burn and keep apart still run)")
            s = {**s, "_no_lock": True, "sliders": {}}
        # Forge calls this twice in a hires pass (the first install is dropped with its UNet);
        # both installs are the same, only the console lines repeat
        if not (K.active(s, table, "" if s.get("_no_lock") else engine._scene_prompt(p)) or K.extras_active(s)):
            engine._release()   # a previous pass's keep-apart must not carry over
            return              # nothing touched: output identical to no extension
        installed = engine.install(p, s, table)
        if installed is not None:
            profile = installed.plan.profile
            p.extra_generation_params[INFOTEXT_KEY] = K.to_infotext(typed, table, profile)
            moved = [f"{slider['key']} {int((s['sliders'] or {}).get(slider['key'], 0)):+d}"
                     for slider in table["sliders"]
                     if K.level_name(slider, (s["sliders"] or {}).get(slider["key"], 0))]
            extras = [x for x, on in (("keep apart", installed.isolated),
                                      ("anti-burn", installed.rescale > 0)) if on]
            print(f"[Lewd Anatomy] {'hires pass' if hr else 'pass'}: {', '.join(moved + extras) or 'nothing'}")

    def postprocess(self, p, processed, *args):
        done = engine.finish()
        if done is not None and done.calls:
            print(f"[Lewd Anatomy] {done.summary()}")


TILED_WORDS = ("upscale", "tile", "tiled")


def tiled_script(p) -> str | None:
    """Title of the selected script when it samples the image in tiles (SD upscale,
    Ultimate SD upscale...), else None. The WebUI keeps the selected script's index in
    p.script_args[0] (0 = none) and the scripts in p.scripts.selectable_scripts."""
    try:
        idx = int((getattr(p, "script_args", None) or [0])[0] or 0)
        if idx <= 0:
            return None
        title = str(p.scripts.selectable_scripts[idx - 1].title())
        return title if any(w in title.lower() for w in TILED_WORDS) else None
    except Exception:
        return None


# --------------------------------------------------------------------- X/Y/Z plot

XYZ_SLIDERS = tuple((s["label"].replace(" ↔ ", "/"), s["key"]) for s in SLIDERS)   # one axis per slider


def _apply_xyz(p, s):
    overrides = getattr(p, "_lewd_anatomy_xyz", None)
    if not overrides:
        return
    for key, value in overrides.items():
        if key in ("strength", "locality"):
            s[key] = float(value)
        elif key == "max_concepts":
            s[key] = int(float(value))
        elif key in ("attn_area", "rescale"):
            s[key] = float(value)
        else:
            v = K.level_value(BY_KEY[key], value)
            s["sliders"][key] = 0 if v is None else v
        s["enabled"] = True


def _setter(key):
    def apply(p, x, xs):
        overrides = dict(getattr(p, "_lewd_anatomy_xyz", {}) or {})
        overrides[key] = x
        p._lewd_anatomy_xyz = overrides
    return apply


def _choices(key):
    names = K.step_names(BY_KEY[key])
    return lambda: [names[v] for v in (-2, -1) if v in names] + ["off", names[1], names[2]]


def _register_xyz():
    xyz = None
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid") or data.path.endswith("xyz_grid.py"):
            xyz = data.module
            break
    if xyz is None or any(getattr(o, "label", "").startswith("Lewd Anatomy:") for o in xyz.axis_options):
        return
    options = [xyz.AxisOption(f"Lewd Anatomy: {label}", str, _setter(key), choices=_choices(key))
               for label, key in XYZ_SLIDERS if key in BY_KEY]
    options.append(xyz.AxisOption("Lewd Anatomy: Strength", float, _setter("strength")))
    options.append(xyz.AxisOption("Lewd Anatomy: Locality", float, _setter("locality")))
    options.append(xyz.AxisOption("Lewd Anatomy: Max concepts", int, _setter("max_concepts")))
    options.append(xyz.AxisOption("Lewd Anatomy: Mask area", float, _setter("attn_area")))
    options.append(xyz.AxisOption("Lewd Anatomy: Anti-burn", float, _setter("rescale")))
    xyz.axis_options.extend(options)


try:
    script_callbacks.on_before_ui(_register_xyz)
except Exception:
    pass
