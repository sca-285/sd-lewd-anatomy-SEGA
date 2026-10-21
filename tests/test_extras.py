"""Perp-Neg edits, Control mode, range by sigma, CFG rescale (anti-burn) and
keep-apart words (anti-bleed). Torch, no WebUI.

    python tests/test_extras.py      or      python -m pytest tests
"""
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from lib_lewd_anatomy import concepts as K, engine as E, host, isolate as I  # noqa: E402
from lib_lewd_anatomy.sega import cfg_rescale, perpendicular  # noqa: E402

torch.manual_seed(0)


# ------------------------------------------------------------------ math

def test_perpendicular():
    v, r = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    p = perpendicular(v, r)
    dots = (p.flatten(1) * r.flatten(1)).sum(1)
    assert torch.allclose(dots, torch.zeros(3), atol=1e-3)
    assert torch.allclose(perpendicular(r * 2.5, r), torch.zeros_like(r), atol=1e-4)   # parallel -> nothing


def test_cfg_rescale():
    x = torch.randn(2, 4, 16, 16)
    sigma = torch.tensor([5.0, 1.0])
    cond = torch.randn_like(x)
    burnt = cond * 4.0                                      # an over-guided prediction
    assert cfg_rescale(x, sigma, burnt, cond, 0.0) is burnt
    out = cfg_rescale(x, sigma, burnt, cond, 1.0)
    sig = sigma.view(-1, 1, 1, 1)
    xs, k = x / (sig * sig + 1), (sig * sig + 1) ** 0.5 / sig
    v_out, v_cond = (xs - out) * k, (xs - cond) * k
    assert torch.allclose(v_out.std(dim=(1, 2, 3)), v_cond.std(dim=(1, 2, 3)), rtol=1e-4)
    half = cfg_rescale(x, sigma, burnt, cond, 0.5)
    assert torch.allclose(half, (out + burnt) / 2, atol=1e-4)      # linear in phi


# ------------------------------------------------------------------ engine

class FakeModel:
    def __call__(self, x, sigma, cross):
        return x * 0.5 + cross.mean(dim=(1, 2)).reshape(-1, *([1] * (x.ndim - 1)))


def fake_predict(calls):
    model = FakeModel()

    def predict(m, entries, x, sigma, options):
        calls.append(x.shape[0])
        return model(x, sigma, entries[0]["model_conds"]["c_crossattn"])
    return predict


class Cond:
    def __init__(self, t):
        self.shape, self.t = t.shape, t

    def _copy_with(self, t):
        return t


def args_for(batch=1, model=None):
    x = torch.randn(batch, 4, 8, 8)
    t = {"model_conds": {"c_crossattn": Cond(torch.zeros(batch, 77, 8))}}
    c, u = torch.randn_like(x), torch.randn_like(x)
    return {"denoised": u + 5 * (c - u), "cond_denoised": c, "uncond_denoised": u, "input": x,
            "sigma": torch.full((batch,), 2.0), "cond": [t], "uncond": [], "model": model, "model_options": {}}


def bare_engine(**kw):
    st = {**K.DEFAULTS, "start": 0.0, "end": 1.0, "_cfg": 5.0, **kw}
    return E.Engine(K.Plan("realistic"), [], st, log=lambda *_: None, table=K.load_table())


def test_rescale_runs_with_no_slider():
    eng = bare_engine(rescale=0.7)
    a = args_for()
    out = eng.post_cfg(a)
    assert torch.allclose(out, cfg_rescale(a["input"], a["sigma"], a["denoised"], a["cond_denoised"], 0.7))
    assert bare_engine().post_cfg(a) is a["denoised"]               # nothing on: untouched


def test_orthogonal_edit_is_perpendicular_to_the_prompt():
    calls = []
    orig = host.predict
    host.predict = fake_predict(calls)
    try:
        plan = K.Plan("realistic", [K.Concept("t", "a", 1.0, 3.0)])
        conds = [{"crossattn": torch.randn(1, 77, 8), "vector": None}]
        st = {**K.DEFAULTS, "start": 0.0, "end": 1.0, "attn_mask": False, "locality": 0.0}
        on = E.Engine(plan, conds, {**st, "orthogonal": True}, log=lambda *_: None)
        off = E.Engine(plan, conds, {**st, "orthogonal": False}, log=lambda *_: None)
        a = args_for()
        s4 = a["sigma"].view(-1, 1, 1, 1)
        g_on = (a["denoised"] - on.post_cfg(dict(a))) / s4          # the added guidance, eps units
        g_off = (a["denoised"] - off.post_cfg(dict(a))) / s4
        direction = ((a["input"] - a["cond_denoised"]) - (a["input"] - a["uncond_denoised"])) / s4
        dot = lambda u, v: float((u.flatten() * v.flatten()).sum())   # noqa: E731
        assert abs(dot(g_on, direction)) < 1e-3 * abs(dot(g_off, direction)) + 1e-4
        assert not torch.allclose(g_on, g_off)
    finally:
        host.predict = orig


def test_control_mode_is_one_gain():
    calls = []
    orig = host.predict
    host.predict = fake_predict(calls)
    try:
        plan = K.Plan("realistic", [K.Concept("t", "a", 1.0, 3.0)])
        conds = [{"crossattn": torch.randn(1, 77, 8), "vector": None}]
        st = {**K.DEFAULTS, "start": 0.0, "end": 1.0, "attn_mask": False, "protect_face": False,
              "locality": 0.0, "orthogonal": False, "_cfg": 6.0, "_no_edit_cap": True}
        a = args_for()
        s4 = a["sigma"].view(-1, 1, 1, 1)

        def edit(mode):
            eng = E.Engine(plan, conds, {**st, "control": mode}, log=lambda *_: None)
            eng.sega.momentum_scale = 0.0
            return (a["denoised"] - eng.post_cfg(dict(a))) / s4

        bal, prompt, sliders = edit("Balanced"), edit("Prompt"), edit("Sliders")
        assert torch.allclose(prompt, bal * 0.7, atol=1e-5) and torch.allclose(sliders, bal * 1.3, atol=1e-5)
        # Sliders turns Perp-Neg off; Balanced and Prompt keep it
        on = {**st, "orthogonal": True}
        assert E.Engine(plan, conds, {**on, "control": "Prompt"}, log=lambda *_: None).orthogonal
        assert not E.Engine(plan, conds, {**on, "control": "Sliders"}, log=lambda *_: None).orthogonal
    finally:
        host.predict = orig


def test_rotated_differences_fade_with_age():
    calls = []
    orig = host.predict
    host.predict = fake_predict(calls)
    try:
        plan = K.Plan("realistic", [K.Concept("a", "a", 1.0, 3.0), K.Concept("b", "b", 1.0, 3.0)])
        conds = [{"crossattn": torch.randn(1, 77, 8), "vector": None} for _ in range(2)]
        st = {**K.DEFAULTS, "start": 0.0, "end": 1.0, "attn_mask": False, "protect_face": False,
              "locality": 0.0, "orthogonal": False, "rotate": 1, "range_by": "Steps"}
        eng = E.Engine(plan, conds, st, log=lambda *_: None)
        eng.sega.momentum_scale = 0.0
        for step in range(4):
            eng.on_step(step, 10)
            eng.post_cfg(args_for())
        # one concept per step: by now each has been reused and carries an age
        assert max(eng.stale_age) >= 1 and eng.stale_decay < 1.0
    finally:
        host.predict = orig


def test_heavy_overlap_is_logged_once():
    calls, logged = [], []
    orig = host.predict
    host.predict = fake_predict(calls)
    try:
        plan = K.Plan("realistic", [K.Concept(k, "c" + k, 1.0, 3.0) for k in ("a", "b", "c", "d")])
        conds = [{"crossattn": torch.full((1, 77, 8), 1.0 + i), "vector": None} for i in range(4)]
        st = {**K.DEFAULTS, "start": 0.0, "end": 1.0, "attn_mask": False, "protect_face": False,
              "locality": 0.0, "orthogonal": False, "range_by": "Steps"}
        eng = E.Engine(plan, conds, st, log=logged.append)
        for step in range(3):
            eng.on_step(step, 10)
            eng.post_cfg(args_for())
        warns = [m for m in logged if "heavy overlap" in m]
        assert len(warns) == 1 and "4 sliders" in warns[0], logged
    finally:
        host.predict = orig


def test_edit_cap_holds_a_strong_push_to_the_prompts_own():
    calls = []
    orig = host.predict
    host.predict = fake_predict(calls)
    try:
        plan = K.Plan("realistic", [K.Concept("penis", "tiny", 1.0, 400.0)])   # absurdly strong
        conds = [{"crossattn": torch.randn(1, 77, 8) * 3, "vector": None}]
        st = {**K.DEFAULTS, "start": 0.0, "end": 1.0, "attn_mask": False, "protect_face": False,
              "locality": 0.0, "orthogonal": False, "_cfg": 6.0}
        a = args_for()
        s4 = a["sigma"].view(-1, 1, 1, 1)
        logged = []
        capped = E.Engine(plan, conds, st, log=logged.append)
        free = E.Engine(plan, conds, {**st, "_no_edit_cap": True}, log=lambda *_: None)
        e_cap = (a["denoised"] - capped.post_cfg(dict(a))) / s4
        e_free = (a["denoised"] - free.post_cfg(dict(a))) / s4
        push = 6.0 * ((a["input"] - a["cond_denoised"]) - (a["input"] - a["uncond_denoised"])) / s4
        norm = lambda t: t.pow(2).sum(1).sqrt()                      # noqa: E731
        assert float(norm(e_cap).max()) < float(norm(e_free).max())
        # never past the prompt's push (smoothed, with its floor): at most its largest value
        assert float(norm(e_cap).max()) <= float(norm(push).max()) * 1.0 + 1e-4
        assert any("edit cap" in m for m in logged)
    finally:
        host.predict = orig


def test_mask_edge_is_a_ramp():
    from lib_lewd_anatomy.attention import soft_mask
    yy, xx = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
    attn = torch.exp(-((yy - 64) ** 2 + (xx - 64) ** 2) / 200.0)[None]     # attention falls off from a spot
    m = soft_mask(attn, 0.03)[0]
    row = m[64, 30:64]
    steps = int(((row > 0.02) & (row < 0.98)).sum())
    assert steps >= 6, row                                           # several pixels of ramp, not a step
    assert float(m[64, 64]) == 1.0 and float(m[0, 0]) == 0.0


def test_range_by_sigma():
    ms = types.SimpleNamespace(sigma_max=14.6, sigma_min=0.03)
    model = types.SimpleNamespace(model_sampling=ms)
    eng = bare_engine(rescale=0.5)
    import math
    for sigma, expected in ((14.6, 0.0), (0.03, 1.0), (math.sqrt(14.6 * 0.03), 0.5)):
        a = args_for(model=model)
        a["sigma"] = torch.full((1,), sigma)
        eng.post_cfg(a)
        assert abs(eng.progress - expected) < 1e-6, (sigma, eng.progress)
    steps = bare_engine(rescale=0.5, range_by="Steps")
    steps.on_step(3, 11)
    a = args_for(model=model)
    steps.post_cfg(a)
    assert abs(steps.progress - 0.3) < 1e-9                       # unchanged by sigma


def test_isolation_swaps_the_main_conditioning():
    eng = bare_engine()
    iso = [torch.ones(1, 77, 8), torch.ones(1, 77, 8) * 2]
    eng.isolated = iso
    params = types.SimpleNamespace(text_cond={"crossattn": torch.zeros(2, 77, 8), "vector": torch.zeros(2, 5)})
    eng.apply_isolation(params)
    assert torch.equal(params.text_cond["crossattn"], torch.cat(iso))
    assert torch.equal(params.text_cond["vector"], torch.zeros(2, 5))      # pooled untouched
    bad = types.SimpleNamespace(text_cond=torch.zeros(3, 77, 8))           # AND prompt: 3 rows
    eng.apply_isolation(bad)
    assert torch.equal(bad.text_cond, torch.zeros(3, 77, 8)) and eng.isolated is None


# ------------------------------------------------------------------ isolate words (word-level fake tokenizer)

class WordEngine:
    """Splits on spaces and commas, one token per word; 75 + 2 per chunk."""
    id_start, id_end, comma_token = 1, 2, 3

    def __init__(self):
        self.vocab = {",": 3, "_": 4}

    def _id(self, w):
        return self.vocab.setdefault(w.lower(), len(self.vocab) + 10)

    def tokenize(self, texts):
        out = []
        for t in texts:
            words = t.replace(",", " , ").replace(";", " ; ").replace(". ", " . ").rstrip(".").split()
            if t.rstrip().endswith(".") and t.strip() != ".":
                words.append(".")
            if t.strip() == ".":
                words = ["."]
            out.append([self._id(w) for w in words])
        return out

    def tokenize_line(self, line):
        ids = self.tokenize([line])[0]
        chunk = type("C", (), {})()
        chunk.tokens = [1] + ids + [2] * (76 - len(ids))
        return [chunk], len(ids)


def test_hair_masks_read_the_place_words():
    """A bare 'hair' attends to the head: Pubic / Armpit hair masks are read from 'pubic' /
    'armpit' only. Sliders without mask_words keep every token of their concept."""
    eng, T = WordEngine(), K.load_table()
    full = "1boy, black hair, nude, thick dense pubic hair, curly pubic hair"
    c = K.Concept("pubic-hair", "thick dense pubic hair, curly pubic hair", 1.0, 3.0)
    pos, _ = E.mask_tokens(eng, full, c, T)
    flat = eng.tokenize_line(full)[0][0].tokens
    assert {flat[p] for p in pos} == {eng.vocab["pubic"]}                # never "hair"
    other = K.Concept("waist", "narrow waist", 1.0, 3.0)                 # no mask_words: every token
    pos2, _ = E.mask_tokens(eng, "1girl, nude, narrow waist", other, T)
    flat2 = eng.tokenize_line("1girl, nude, narrow waist")[0][0].tokens
    assert [flat2[p] for p in pos2] == [eng.vocab["narrow"], eng.vocab["waist"]]
    fused = K.Concept("fuse:nipples@detail", "large areolae, erect nipples, brown nipples", 1.0, 3.0)
    pos3, _ = E.mask_tokens(eng, "1girl, large areolae, erect nipples, brown nipples", fused, T)
    flat3 = eng.tokenize_line("1girl, large areolae, erect nipples, brown nipples")[0][0].tokens
    assert {flat3[p] for p in pos3} <= {eng.vocab["areolae"], eng.vocab["nipples"]}
    for key in ("pubic-hair", "armpit-hair", "body-hair", "limb-hair"):
        s = K.slider_by_key(T)[key]
        for profile, entries in s["concepts"].items():
            for name, raw in entries.items():
                for e in K.entries(raw):
                    assert "detailed hair" not in e["text"], (key, profile, name)


def test_isolate_words():
    eng = WordEngine()
    text = "1girl, blonde hair, 1boy, black hair"
    words = I.parse_words("blonde | black")
    assert words == ["blonde", "black"]
    count = lambda s: len(eng.tokenize([s])[0])                   # noqa: E731
    assert I.padded(text, words, 1, count) == "1girl, blonde hair, 1boy, _ hair"
    assert I.padded(text, words, None, count) == "1girl, _ hair, 1boy, _ hair"
    assert I.target_chunks(text, words) == [1, 3]

    def encode_one(t):            # token embedding = a value per token id, 77 positions
        ids = eng.tokenize_line(t)[0][0].tokens
        return {"crossattn": torch.tensor(ids, dtype=torch.float32).view(1, -1, 1)}

    out = I.isolate(text, words, 1.0, encode_one, eng)[0, :, 0].tolist()
    tok = eng.tokenize_line(text)[0][0].tokens
    phrase = I.token_chunk_index(tok, {1, 2}, 3)
    blonde, black = eng.vocab["blonde"], eng.vocab["black"]
    for i, ph in enumerate(phrase):
        if ph == 1 and tok[i] == blonde:
            assert out[i] == blonde                                # its own phrase keeps "blonde"
        if ph == 3 and tok[i] == black:
            assert out[i] == black
    assert I.isolate("1girl, red dress", words, 1.0, encode_one, eng) is None     # nothing to isolate
    half = I.isolate(text, words, 0.5, encode_one, eng)
    full = I.isolate(text, words, 1.0, encode_one, eng)
    assert torch.allclose(half, (full + encode_one(text)["crossattn"]) / 2)


def test_isolate_sentences_and_trailing_commas():
    eng = WordEngine()
    count = lambda s: len(eng.tokenize([s])[0])                   # noqa: E731
    words = ["blonde", "black"]
    text = "A blonde woman lies on a bed. A man with black hair sits beside her."
    assert I.target_chunks(text, words) == [0, 1]
    assert I.padded(text, words, 0, count) == "A blonde woman lies on a bed. A man with _ hair sits beside her."
    w = "(red hair:1.2), x"
    assert [w[a:b] for a, b in I.chunks(w)] == ["(red hair:1.2)", " x"]          # 1.2 is not a stop

    def encode_one(t):
        ids = eng.tokenize_line(t)[0][0].tokens
        return {"crossattn": torch.tensor(ids, dtype=torch.float32).view(1, -1, 1)}

    out = I.isolate(text, words, 1.0, encode_one, eng)
    assert out is not None
    tok = eng.tokenize_line(text)[0][0].tokens
    seps = {eng.vocab[","], eng.vocab.get("."), eng.vocab.get(";")} - {None}
    phrase = I.token_chunk_index(tok, {1, 2}, seps)
    vals = out[0, :, 0].tolist()
    for i, ph in enumerate(phrase):
        if tok[i] == eng.vocab["blonde"]:
            assert ph == 0 and vals[i] == tok[i]
        if tok[i] == eng.vocab["black"]:
            assert ph == 1 and vals[i] == tok[i]
    assert I.isolate("1girl, blonde hair, 1boy, black hair,", words, 1.0, encode_one, eng) is not None


def test_cuntboy_and_futa_rules():
    T = K.load_table()
    assert K.scene_genders("1boy, cuntboy, nude", T) == {"men", "cuntboy"}
    assert K.toward_allowed("Women", {"men", "cuntboy"}) and not K.toward_allowed("Women", {"men"})
    assert K.toward_allowed("Men", {"women", "futa"}) and not K.toward_allowed("Men", {"women"})
    vulva = K.plan({**K.DEFAULTS, "enabled": True, "sliders": {"vulva": 1}, "who": {}}, T, "anime", "1boy, cuntboy")
    assert any(c.sign > 0 and c.scale > 0 for c in vulva.concepts)
    plain = K.plan({**K.DEFAULTS, "enabled": True, "sliders": {"vulva": 1}, "who": {}}, T, "anime", "1boy")
    assert not any(c.sign > 0 and c.scale > 0 for c in plain.concepts)


def test_tiled_script_detection():
    import importlib.util
    import types as _t
    mods = {}
    for name in ("modules", "modules.script_callbacks", "modules.scripts", "modules.shared"):
        mods[name] = sys.modules.get(name)
    try:
        m = _t.ModuleType("modules")
        cb = _t.ModuleType("modules.script_callbacks")
        for n in ("on_cfg_denoiser", "on_after_component", "on_before_ui"):
            setattr(cb, n, lambda f: None)
        sc = _t.ModuleType("modules.scripts")
        sc.Script, sc.AlwaysVisible, sc.scripts_data = object, object(), []
        sh = _t.ModuleType("modules.shared")
        sh.opts = _t.SimpleNamespace(sd_model_checkpoint="")
        for k, v in (("modules", m), ("modules.script_callbacks", cb), ("modules.scripts", sc), ("modules.shared", sh)):
            sys.modules[k] = v
        m.script_callbacks, m.scripts, m.shared = cb, sc, sh
        try:
            import gradio  # noqa: F401
        except ImportError:
            return
        path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "lewd_anatomy.py"
        spec = importlib.util.spec_from_file_location("la_script", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        title = lambda t: _t.SimpleNamespace(title=lambda: t)            # noqa: E731
        runner = _t.SimpleNamespace(selectable_scripts=[title("Prompts from file"), title("Ultimate SD upscale"),
                                                        title("SD upscale")])
        assert mod.tiled_script(_t.SimpleNamespace(script_args=(2, 1), scripts=runner)) == "Ultimate SD upscale"
        assert mod.tiled_script(_t.SimpleNamespace(script_args=(3,), scripts=runner)) == "SD upscale"
        assert mod.tiled_script(_t.SimpleNamespace(script_args=(1,), scripts=runner)) is None
        assert mod.tiled_script(_t.SimpleNamespace(script_args=(0,), scripts=runner)) is None
        assert mod.tiled_script(_t.SimpleNamespace()) is None
    finally:
        for k, v in mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_callback_ignores_a_nested_job():
    """ADetailer repaints inside postprocess_image, while this job's engine is still open:
    its sampling must not be touched (no isolation swap, no step tracking)."""
    eng = bare_engine()
    main_job, nested_job = object(), object()
    eng.job = main_job
    eng.isolated = [torch.ones(1, 77, 8)]
    E.CURRENT = eng
    try:
        nested = types.SimpleNamespace(text_cond={"crossattn": torch.zeros(1, 77, 8)}, sampling_step=7,
                                       total_sampling_steps=9, denoiser=types.SimpleNamespace(p=nested_job))
        E.on_cfg_denoiser(nested)
        assert torch.equal(nested.text_cond["crossattn"], torch.zeros(1, 77, 8)) and eng.step == 0
        own = types.SimpleNamespace(text_cond={"crossattn": torch.zeros(1, 77, 8)}, sampling_step=3,
                                    total_sampling_steps=9, denoiser=types.SimpleNamespace(p=main_job))
        E.on_cfg_denoiser(own)
        assert torch.equal(own.text_cond["crossattn"], torch.ones(1, 77, 8)) and eng.step == 3
    finally:
        E.CURRENT = None


def test_taper_fades_the_end_of_each_window():
    plan = K.Plan("realistic", [K.Concept("t", "a", 1.0, 3.0)])
    eng = E.Engine(plan, [{"crossattn": torch.zeros(1, 77, 8), "vector": None}],
                   {**K.DEFAULTS, "start": 0.2, "end": 0.6, "range_by": "Steps"}, log=lambda *_: None,
                   table={**K.load_table(), "taper": 0.25})
    assert eng.windows == [(0.2, 0.6)]
    for progress, expected in ((0.3, 1.0), (0.5, 1.0), (0.55, 0.5), (0.6, 0.0)):
        eng.progress = progress
        assert abs(eng._taper(0) - expected) < 1e-9, (progress, eng._taper(0))
    assert K.DEFAULTS["start"] == 0.10 and K.DEFAULTS["end"] == 0.60       # sigma defaults, see the commit


def test_settings_infotext_round_trip():
    s = {**K.DEFAULTS, "enabled": True, "sliders": {"breasts": 1}, "who": {},
         "control": "Sliders", "rescale": 0.7, "isolate": "blonde, black hair"}
    text = K.to_infotext(s, K.load_table(), "anime")
    back = K.from_infotext(text, K.load_table())
    assert back["control"] == "Sliders" and back["rescale"] == 0.7 and back["isolate"] == "blonde, black hair"
    plain = K.to_infotext({**K.DEFAULTS, "enabled": True, "sliders": {"breasts": 1}, "who": {}}, K.load_table(), "anime")
    assert not any(k in plain for k in ("control", "rescale", "isolate", "orthogonal", "rotate", "staggered"))
    # a 1.x infotext with the removed knobs still pastes: they are ignored
    old = K.from_infotext("profile=anime, strength=1.0, breasts=full, rotate=4, staggered=on, orthogonal=off, "
                          "fast=on, range_by=steps, isolate_w=0.3, control=sega", K.load_table())
    assert old["sliders"]["breasts"] == 1 and old["control"] == "Sliders"
    assert old["rotate"] == 0 and old["orthogonal"] is True and old["range_by"] == "Sigma"
    assert K.extras_active({"enabled": True, "rescale": 0.5}) and K.extras_active({"enabled": True, "isolate": "x"})
    assert not K.extras_active({"enabled": True}) and not K.extras_active({"enabled": False, "rescale": 1})


def test_attention_records_odd_sdxl_sizes():
    """SDXL at 1080 px: latent 135 rounds to 68 then 34; the /4 layers must still be recorded."""
    from lib_lewd_anatomy import attention as A
    rec = A.Recorder([([1], 77)], (135, 135))
    n = 34 * 34
    q = torch.randn(1, n, 64)
    k = torch.randn(1, 77, 64)
    rec(q, k, k, {"n_heads": 1})
    assert rec.count == 1


def test_pose_variant_per_image_seed():
    import types
    assert E._seeds(types.SimpleNamespace(seeds=[5, 6, 7]), 3) == [5, 6, 7]
    assert E._seeds(types.SimpleNamespace(seed=10), 2) == [10, 11]
    c = K.Concept("pose-style", "a", 1.0, 1.0, variants=("a", "b", "c"))
    assert [K.variant(c, s) for s in E._seeds(types.SimpleNamespace(seeds=[3, 4, 5]), 3)] == ["a", "b", "c"]


def test_cache_is_bounded_and_per_checkpoint():
    import types
    orig_key, orig_parser = host.checkpoint_key, sys.modules.get("modules.prompt_parser")
    mods = sys.modules.setdefault("modules", types.ModuleType("modules"))
    pp = types.ModuleType("modules.prompt_parser")
    pp.SdConditioning = lambda texts, width=0, height=0: list(texts)
    sys.modules["modules.prompt_parser"] = pp
    mods.prompt_parser = pp

    class M:
        def get_learned_conditioning(self, texts):
            return torch.zeros(len(texts), 77, 2048, dtype=torch.float16)    # ~315 KB each
    p = types.SimpleNamespace(sd_model=M())
    try:
        E.clear_cache()
        host.checkpoint_key = lambda p: "a"
        texts = [f"t{i}" for i in range(300)]
        out = E.encode(p, texts, 64, 64)
        assert len(out) == 300                                       # this call's texts all returned
        assert E._cache_bytes <= E._CACHE_BYTES and len(E._cond_cache) <= E._CACHE_LIMIT
        assert E._cache_bytes == sum(E._nbytes(v) for v in E._cond_cache.values())
        host.checkpoint_key = lambda p: "b"                          # another checkpoint: a fresh cache
        E.encode(p, ["x"], 64, 64)
        assert list(E._cond_cache) == [("b", "x", 64, 64)]
    finally:
        host.checkpoint_key = orig_key
        if orig_parser is not None:
            sys.modules["modules.prompt_parser"] = orig_parser
        E.clear_cache()


def test_finish_releases_the_step_tensors():
    plan = K.Plan("realistic", [K.Concept("t", "a", 1.0, 3.0)])
    conds = [{"crossattn": torch.randn(1, 77, 8), "vector": None}]
    eng = E.Engine(plan, conds, {**K.DEFAULTS, "_cfg": 6.0}, log=lambda *_: None)
    eng.stale[0] = torch.ones(1, 4, 8, 8)
    eng.attn[0] = torch.ones(1, 8, 8)
    eng.face_mask = torch.ones(1, 1, 8, 8)
    eng.sega.momentum = torch.ones(1, 4, 8, 8)
    E.CURRENT = eng
    assert E.finish() is eng and E.CURRENT is None
    assert eng.stale == [None] and eng.attn == [None] and eng.face_mask is None and eng.sega.momentum is None
    assert eng.plan is plan and eng.conds is conds              # still there for current_patch


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS ", name)
            except AssertionError as e:
                fails += 1
                print("FAIL ", name, e)
    sys.exit(1 if fails else 0)
