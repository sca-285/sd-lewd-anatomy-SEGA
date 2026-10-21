"""SEGA math and the engine's post-CFG function on random tensors (torch, no WebUI).

    python tests/test_sega.py      or      python -m pytest tests
"""
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from lib_lewd_anatomy import concepts as K, engine as E, host  # noqa: E402
from lib_lewd_anatomy.sega import Sega, concept_guidance, quantile_mask  # noqa: E402

torch.manual_seed(0)


def test_mask_fraction_matches_locality():
    d = torch.randn(2, 4, 64, 64)
    for lam in (0.90, 0.95, 0.99):
        frac = quantile_mask(d, lam).float().mean().item()
        assert abs(frac - (1 - lam)) < 0.003, (lam, frac)
    # per sample and channel: every (b, c) slice keeps its own top share
    per = quantile_mask(d, 0.95).float().mean(dim=(2, 3))
    assert torch.allclose(per, torch.full_like(per, 0.05), atol=0.003)


def test_toward_and_away_signs():
    u, e = torch.randn(1, 4, 32, 32), torch.randn(1, 4, 32, 32)
    toward = concept_guidance(u, e, 5.0, 0.9)
    away = concept_guidance(u, e, -5.0, 0.9)
    assert torch.equal(toward, -away)
    m = toward != 0
    assert torch.all(torch.sign(toward[m]) == torch.sign((e - u)[m]))
    assert torch.allclose(toward[m], 5.0 * (e - u)[m])


def test_momentum():
    u, e = torch.zeros(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    s = Sega(momentum_scale=0.3, beta=0.6)
    g1 = s.guidance(u, [e], [2.0], [0.5], [True])
    ref1 = concept_guidance(u, e, 2.0, 0.5)
    assert torch.allclose(g1, ref1)                                  # step 1: no momentum yet
    assert torch.allclose(s.momentum, 0.4 * ref1)
    g2 = s.guidance(u, [e], [2.0], [0.5], [True])
    assert torch.allclose(g2, ref1 + 0.3 * 0.4 * ref1)               # + s_m * nu
    assert torch.allclose(s.momentum, 0.6 * 0.4 * ref1 + 0.4 * g2)   # nu = beta nu + (1 - beta) g
    s.reset()
    assert s.momentum is None


def test_warmup_feeds_momentum_but_is_not_applied():
    u = torch.zeros(1, 4, 8, 8)
    a, b = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    s = Sega(0.3, 0.6)
    g = s.guidance(u, [a, b], [1.0, 1.0], [0.5, 0.5], [True, False])
    assert torch.allclose(g, concept_guidance(u, a, 1.0, 0.5))       # only the live concept
    assert torch.allclose(s.momentum, 0.4 * (concept_guidance(u, a, 1.0, 0.5) + concept_guidance(u, b, 1.0, 0.5)))


def test_contrastive_pair_cancels_what_both_share():
    """Concept T = scene + a, concept O = scene + b (scene: e.g. nudity, background).
    Measured against each other, the scene term drops out entirely."""
    torch.manual_seed(5)
    u = torch.randn(1, 4, 16, 16)
    a, b = torch.randn_like(u), torch.randn_like(u)
    g1 = Sega(0.0, 0.6).guidance(u, [u + a, u + b], [1.0, -1.0], [0.9, 0.9], [True, True], refs=[None, 0])
    big = torch.randn_like(u) * 50                        # a huge shared difference (clothes, say)
    g2 = Sega(0.0, 0.6).guidance(u, [u + big + a, u + big + b], [0.0, -1.0], [0.9, 0.9], [True, True], refs=[None, 0])
    g3 = Sega(0.0, 0.6).guidance(u, [u + a, u + b], [0.0, -1.0], [0.9, 0.9], [True, True], refs=[None, 0])
    assert torch.allclose(g2, g3, atol=1e-4)              # shared term cancelled
    assert torch.allclose(g3, concept_guidance(u + a, u + b, -1.0, 0.9))
    assert not torch.allclose(g1, g3)                     # the toward term adds its own part


def test_x0_space_matches_eps_space():
    x = torch.randn(2, 4, 16, 16)
    sigma = torch.tensor([3.0, 0.7])
    u_x0, c_x0, den = torch.randn_like(x), torch.randn_like(x), torch.randn_like(x)
    out = Sega(0.0, 0.6).apply(x, sigma, den, u_x0, [c_x0], [4.0], [0.9], [True])
    sig = sigma.reshape(-1, 1, 1, 1)
    eps_u, eps_e, eps_den = (x - u_x0) / sig, (x - c_x0) / sig, (x - den) / sig
    eps_new = eps_den + concept_guidance(eps_u, eps_e, 4.0, 0.9)
    assert torch.allclose(out, x - sig * eps_new, atol=1e-5)


def test_away_reduces_similarity_to_concept():
    """Pushing away from a concept moves the prediction away from it where it differs most
    (scale below 1 so the toward step does not overshoot the concept)."""
    x = torch.randn(1, 4, 32, 32)
    sigma = torch.tensor([1.0])
    u = torch.randn_like(x)
    concept = u + torch.randn_like(x)
    den = u.clone()
    away = Sega(0.0, 0.6).apply(x, sigma, den, u, [concept], [-0.5], [0.9], [True])
    toward = Sega(0.0, 0.6).apply(x, sigma, den, u, [concept], [0.5], [0.9], [True])
    assert (away - concept).norm() > (den - concept).norm() > (toward - concept).norm()


# ------------------------------------------------------------------ engine (fake model)

class FakeModel:
    """apply_model-like: x0 depends on x and on the conditioning mean."""

    def __call__(self, x, sigma, cross):
        return x * 0.5 + cross.mean(dim=(1, 2)).reshape(-1, *([1] * (x.ndim - 1)))


def make_engine(texts, signs, start=0.0, end=1.0, per_call=1):
    plan = K.Plan("realistic", [K.Concept("t", t, s, 3.0) for t, s in zip(texts, signs)])
    conds = [{"crossattn": torch.randn(1, 77, 8) + i, "vector": torch.randn(1, 5)} for i in range(len(texts))]
    settings = {**K.DEFAULTS, "start": start, "end": end, "per_call": per_call, "orthogonal": False}
    return E.Engine(plan, conds, settings, log=lambda *_: None), conds


def fake_predict(calls):
    model = FakeModel()

    def predict(m, entries, x, sigma, options):
        calls.append((x.shape[0], entries[0]["model_conds"]["c_crossattn"].shape[0]))
        return model(x, sigma, entries[0]["model_conds"]["c_crossattn"])
    return predict


class Cond:
    def __init__(self, t):
        self.shape = t.shape
        self.t = t

    def _copy_with(self, t):
        return t


def post_args(batch=2):
    x = torch.randn(batch, 4, 16, 16)
    template = {"model_conds": {"c_crossattn": Cond(torch.zeros(batch, 154, 8)), "y": Cond(torch.zeros(batch, 5))},
                "area": (1, 1, 0, 0), "control": "kept"}
    return {"denoised": torch.randn_like(x), "uncond_denoised": torch.randn_like(x), "cond_denoised": torch.randn_like(x),
            "input": x, "sigma": torch.full((batch,), 2.0), "cond": [template], "uncond": [], "model": None,
            "model_options": {}}


def test_engine_concepts_per_call_and_reference():
    """per_call = 1 (default): one call of batch B per concept, never bigger than half a
    CFG step. per_call = 3: one call of 3 x B. Same result either way."""
    for per_call, expected in ((1, [(2, 2)] * 3), (2, [(4, 4), (2, 2)]), (3, [(6, 6)])):
        calls = []
        orig = host.predict
        host.predict = fake_predict(calls)
        try:
            torch.manual_seed(1)
            eng, conds = make_engine(["a", "b", "c"], [1.0, -1.0, -1.0], per_call=per_call)
            torch.manual_seed(2)
            args = post_args(batch=2)
            out = eng.post_cfg(args)
            assert calls == expected, (per_call, calls)
            model, x = FakeModel(), args["input"]
            ref_x0 = [model(x, None, c["crossattn"].expand(2, -1, -1)) for c in conds]
            ref = Sega(E.MOMENTUM_SCALE, E.MOMENTUM_BETA).apply(x, args["sigma"], args["denoised"],
                                                                args["cond_denoised"], ref_x0,
                                                                [3.0, -3.0, -3.0], [K.DEFAULTS["locality"]] * 3, [True] * 3)
            assert torch.allclose(out, ref, atol=1e-5)
        finally:
            host.predict = orig


def test_engine_step_range_and_neutral_are_identity():
    calls = []
    orig = host.predict
    host.predict = fake_predict(calls)
    try:
        eng, _ = make_engine(["a"], [1.0], start=0.15, end=0.70)
        args = post_args()
        for step, expect_call in ((0, False), (5, True), (15, False)):
            eng.on_step(step, 20)
            before = len(calls)
            out = eng.post_cfg(args)
            assert (len(calls) > before) == expect_call, step
            if not expect_call:
                assert out is args["denoised"]                       # untouched object: byte-identical
        empty, _ = make_engine([], [])
        assert empty.post_cfg(args) is args["denoised"]
    finally:
        host.predict = orig


def test_engine_guides_from_the_prompt_not_the_negative():
    """Missing negative (CFG 1) still edits. Missing prompt prediction does not.
    The result must follow cond_denoised, which is what stops a bare concept
    phrase from painting on clothes."""
    calls = []
    orig = host.predict
    host.predict = fake_predict(calls)
    try:
        eng, conds = make_engine(["a"], [1.0])
        args = post_args(batch=1)
        args["uncond_denoised"] = torch.zeros_like(args["input"])
        args["cond_denoised"] = torch.randn_like(args["input"])
        out = eng.post_cfg(args)
        assert calls, "CFG 1 (zero uncond) must still run"
        model, x = FakeModel(), args["input"]
        ref_x0 = [model(x, None, c["crossattn"].expand(1, -1, -1)) for c in conds]
        ref_cond = Sega(E.MOMENTUM_SCALE, E.MOMENTUM_BETA).apply(
            x, args["sigma"], args["denoised"], args["cond_denoised"], ref_x0, [3.0], [K.DEFAULTS["locality"]], [True])
        ref_uncond = Sega(E.MOMENTUM_SCALE, E.MOMENTUM_BETA).apply(
            x, args["sigma"], args["denoised"], args["uncond_denoised"], ref_x0, [3.0], [K.DEFAULTS["locality"]], [True])
        assert torch.allclose(out, ref_cond, atol=1e-5)
        assert not torch.allclose(ref_cond, ref_uncond, atol=1e-4)

        calls.clear()
        bare = post_args(batch=1)
        bare["cond_denoised"] = None
        assert eng.post_cfg(bare) is bare["denoised"] and calls == []
        bare["cond_denoised"] = torch.zeros_like(bare["input"])
        assert eng.post_cfg(bare) is bare["denoised"] and calls == []
    finally:
        host.predict = orig


def test_engine_baseline_negative_is_classic_sega():
    calls = []
    orig = host.predict
    host.predict = fake_predict(calls)
    try:
        torch.manual_seed(3)
        eng, conds = make_engine(["a"], [1.0])
        eng.baseline = "Negative"
        args = post_args(batch=1)
        out = eng.post_cfg(args)
        model, x = FakeModel(), args["input"]
        ref_x0 = [model(x, None, c["crossattn"].expand(1, -1, -1)) for c in conds]
        ref = Sega(E.MOMENTUM_SCALE, E.MOMENTUM_BETA).apply(x, args["sigma"], args["denoised"],
                                                            args["uncond_denoised"], ref_x0, [3.0], [K.DEFAULTS["locality"]], [True])
        assert torch.allclose(out, ref, atol=1e-5)
        # CFG 1: no negative pass -> skipped, untouched
        calls.clear()
        eng2, _ = make_engine(["a"], [1.0])
        eng2.baseline = "Negative"
        cfg1 = post_args(batch=1)
        cfg1["uncond_denoised"] = torch.zeros_like(cfg1["input"])
        assert eng2.post_cfg(cfg1) is cfg1["denoised"] and calls == [] and not eng2.off
        # the settings value reaches the engine
        assert E.Engine(K.Plan("r"), [], {**K.DEFAULTS, "baseline": "Negative"}).baseline == "Negative"
        assert E.Engine(K.Plan("r"), [], dict(K.DEFAULTS)).baseline == "Prompt"
    finally:
        host.predict = orig


def make_engine_table(texts, signs, refs=None, phases=None, table=None, **settings_kw):
    refs = refs or [None] * len(texts)
    phases = phases or [None] * len(texts)
    plan = K.Plan("realistic", [K.Concept("t", t, s, 3.0 if s else 0.0, None, r, ph)
                                for t, s, r, ph in zip(texts, signs, refs, phases)])
    conds = [{"crossattn": torch.randn(1, 77, 8) + i, "vector": torch.randn(1, 5)} for i in range(len(texts))]
    settings = {**K.DEFAULTS, "start": 0.0, "end": 1.0, "orthogonal": False, **settings_kw}
    return E.Engine(plan, conds, settings, log=lambda *_: None, table=table or K.load_table()), conds


def counting_predict(calls, oom_above=None):
    model = FakeModel()

    def predict(m, entries, x, sigma, options):
        cross = entries[0]["model_conds"]["c_crossattn"]
        if oom_above is not None and x.shape[0] > oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        calls.append(x.shape[0])
        return model(x, sigma, cross)
    return predict


def test_budget_never_shrinks_one_concept_and_caps_many():
    from lib_lewd_anatomy.sega import budget_factor, _norm
    torch.manual_seed(7)
    one = [torch.randn(2, 4, 8, 8)]
    assert budget_factor(one[0], one, 2.0) == 1.0
    same = [torch.randn(2, 4, 8, 8)] * 6                      # six concepts pushing the same way
    total = torch.stack(same).sum(0)
    f = budget_factor(total, same, 2.0)
    assert torch.allclose(_norm(total * f), 2.0 * _norm(same[0]), rtol=1e-5)
    mixed = [torch.randn(2, 4, 8, 8) for _ in range(2)]      # within budget: untouched
    assert torch.all(budget_factor(torch.stack(mixed).sum(0), mixed, 2.0) == 1)
    assert budget_factor(total, same, 0) == 1.0               # 0 = off


def test_rotate_spreads_concepts_over_steps():
    calls = []
    orig = host.predict
    host.predict = counting_predict(calls)
    try:
        texts = [f"c{i}" for i in range(6)]
        eng, _ = make_engine_table(texts, [1.0] * 6, rotate=2, auto_batch=False, per_call=1)
        args = post_args(batch=1)
        per_step = []
        for step in range(5):
            eng.on_step(step, 20)
            before = len(calls)
            out = eng.post_cfg({**args})
            per_step.append(len(calls) - before)
            assert torch.isfinite(out).all() and not eng.off
        assert per_step[0] == 6                    # first step: every concept needs one fresh result
        assert per_step[1:] == [2, 2, 2, 2]        # then about 2 passes per step
        assert all(d is not None for d in eng.stale)
        # rotate >= concepts: the same as no rotation
        a, _ = make_engine_table(texts, [1.0] * 6, rotate=0, auto_batch=False)
        b, _ = make_engine_table(texts, [1.0] * 6, rotate=6, auto_batch=False)
        b.conds = a.conds
        a.on_step(3, 20); b.on_step(3, 20)
        assert torch.allclose(a.post_cfg({**args}), b.post_cfg({**args}))
    finally:
        host.predict = orig


def test_rotate_keeps_contrastive_pairs_together():
    calls = []
    orig = host.predict
    host.predict = counting_predict(calls)
    try:
        texts = ["small", "large", "a", "b"]
        eng, _ = make_engine_table(texts, [1.0, -1.0, 1.0, 1.0], refs=[None, "small", None, None],
                                   rotate=1, auto_batch=False)
        assert sorted(map(sorted, eng.groups)) == [[0, 1], [2], [3]]
        args = post_args(batch=1)
        eng.on_step(0, 20); eng.post_cfg({**args})
        for step in range(1, 7):
            eng.on_step(step, 20)
            active, predict = eng.due()
            assert predict in ([0, 1], [2], [3]), predict     # a pair is never split across steps
            eng.post_cfg({**args})
    finally:
        host.predict = orig


def test_staggered_windows():
    table = {**K.load_table(), "phases": {"shape": [0.0, 0.5], "detail": [0.5, 1.0]}}
    calls = []
    orig = host.predict
    host.predict = counting_predict(calls)
    try:
        eng, _ = make_engine_table(["shape", "detail"], [1.0, 1.0], phases=["shape", "detail"],
                                   table=table, staggered=True, start=0.2, end=0.8, auto_batch=False)
        assert eng.windows == [(0.2, 0.5), (0.5, 0.8)]
        args = post_args(batch=1)
        eng.on_step(6, 21)                         # progress 0.3: only the shape concept
        assert eng.due() == ([0], [0])
        eng.on_step(14, 21)                        # 0.7: only detail
        assert eng.due() == ([1], [1])
        off, _ = make_engine_table(["shape", "detail"], [1.0, 1.0], phases=["shape", "detail"],
                                   table=table, staggered=False, start=0.2, end=0.8)
        off.on_step(6, 21)
        assert off.due() == ([0, 1], [0, 1])
    finally:
        host.predict = orig


def test_out_of_memory_backs_off_and_retries():
    calls = []
    orig = host.predict
    host.predict = counting_predict(calls, oom_above=1)    # anything bigger than one concept "OOMs"
    try:
        torch.manual_seed(11)
        eng, conds = make_engine_table(["a", "b", "c", "d"], [1.0] * 4, per_call=4, auto_batch=False)
        args = post_args(batch=1)
        out = eng.post_cfg({**args})
        assert not eng.off and eng.chunk_cap == 1 and calls == [1, 1, 1, 1]
        host.predict = counting_predict([])
        ref, _ = make_engine_table(["a", "b", "c", "d"], [1.0] * 4, per_call=1, auto_batch=False)
        ref.conds = eng.conds
        assert torch.allclose(out, ref.post_cfg({**args}), atol=1e-6)    # same result, smaller calls
        # a real error (not out-of-memory) still turns the pass off instead of looping
        host.predict = lambda *a: (_ for _ in ()).throw(ValueError("shape mismatch"))
        bad, _ = make_engine_table(["a"], [1.0], auto_batch=False)
        assert bad.post_cfg({**args}) is args["denoised"] and bad.concepts_off
    finally:
        host.predict = orig


def test_token_positions_across_a_chunk_break():
    from lib_lewd_anatomy.attention import token_positions
    S, E_ = 49406, 49407
    chunk1 = [S] + list(range(1, 75)) + [7, E_]          # the phrase starts at the end of chunk 1 ...
    chunk2 = [S, 8, 9] + [E_] * 74                        # ... and ends in chunk 2
    flat = chunk1 + chunk2
    assert token_positions(flat, [7, 8, 9], {S, E_}) == [75, 78, 79]
    assert token_positions(flat, [1, 2], {S, E_}) == [1, 2]
    assert token_positions(flat, [99], {S, E_}) is None
    # last occurrence wins (the concept is appended at the end of the prompt)
    assert token_positions([S, 5, 6, 5, 6, E_], [5, 6], {S, E_}) == [3, 4]


def test_soft_mask_area_and_feather():
    from lib_lewd_anatomy.attention import soft_mask
    attn = torch.zeros(1, 64, 64)
    attn[0, 10:26, 20:36] = 1.0                            # a 16x16 hot spot = 6.25 % of the map
    attn += torch.rand_like(attn) * 1e-3
    m = soft_mask(attn, area=0.0625)
    assert m[0, 12:24, 22:34].min() == 1.0                 # the core is fully on
    assert m[0, 40:, :].max() == 0.0 and m[0, :, 50:].max() == 0.0   # far away: untouched
    edge = m[0, 9, 20:36]
    assert 0 < edge.max() <= 1.0                           # feathered border


def test_recorder_columns_and_options():
    from lib_lewd_anatomy import attention as A
    r = A.Recorder([([3, 4], 10), None], (8, 8))
    assert r._cols(0, 10) == [3, 4] and r._cols(0, 20) == [3, 4, 13, 14] and r._cols(1, 10) is None
    base = {"transformer_options": {"patches_replace": {"attn2": {("output", 1): "theirs"}}}}
    opts = A.with_recorder(base, [("input", 1), ("output", 1)], r)
    attn2 = opts["transformer_options"]["patches_replace"]["attn2"]
    assert attn2[("input", 1)] is r and attn2[("output", 1)] == "theirs"     # another extension's stays
    assert base["transformer_options"]["patches_replace"]["attn2"] == {("output", 1): "theirs"}   # untouched
    # the replacement computes the same attention as the standard formula
    q, k, v = torch.randn(2, 16, 32), torch.randn(2, 10, 32), torch.randn(2, 10, 32)
    out = A.Recorder([None, None], (4, 4))(q, k, v, {"n_heads": 2})
    qh, kh, vh = (t.view(2, -1, 2, 16).transpose(1, 2) for t in (q, k, v))
    ref = (torch.softmax(qh @ kh.transpose(-1, -2) / 4.0, -1) @ vh).transpose(1, 2).reshape(2, 16, 32)
    assert torch.allclose(out, ref, atol=1e-5)


def test_engine_error_turns_itself_off():
    orig = host.predict

    def boom(*a):
        raise RuntimeError("model says no")
    host.predict = boom
    try:
        eng, _ = make_engine(["a"], [1.0])
        args = post_args()
        assert eng.post_cfg(args) is args["denoised"] and eng.concepts_off and not eng.off
    finally:
        host.predict = orig


def test_prepare_encodes_the_prompt_plus_the_concept():
    seen = {}

    def fake_encode(p, texts, width, height):
        seen["texts"] = list(texts)
        return [{"crossattn": torch.zeros(1, 4, 4), "vector": None} for _ in texts]

    orig = E.encode
    E.encode = fake_encode
    try:
        class P:
            prompt = "1girl, nude, standing in a room"
            prompts = ["1girl, nude, standing in a room"]
            width, height = 64, 64
            sd_model = None

        s = {**K.DEFAULTS, "enabled": True, "profile": "Realistic",
             "sliders": {"breasts": 5}, "who": {"breasts": "Women"}}
        prepared = E.prepare(P(), s, K.load_table(), log=lambda *_a: None)
        assert prepared is not None
        texts = seen["texts"]
        assert texts and all(t.startswith("1girl, nude, standing in a room") for t in texts)
        assert any("large natural breasts" in t for t in texts)
        assert all(t != "large natural breasts" for t in texts)
    finally:
        E.encode = orig


def test_prepare_uses_each_images_own_prompt():
    """A batch with two different prompts: image 2's concepts must be built on image 2's
    prompt, or the scene difference (clothes included) leaks into its edit."""
    seen = {}

    def fake_encode(p, texts, width, height):
        seen["texts"] = list(texts)
        return [{"crossattn": torch.full((1, 4, 4), float(i)), "vector": None} for i, _ in enumerate(texts)]

    orig = E.encode
    E.encode = fake_encode
    try:
        class P:
            prompt = "1girl, completely nude, beach"
            prompts = ["1girl, completely nude, beach", "1girl, completely nude, bedroom"]
            hr_prompts = ["1girl, completely nude, beach, detailed", "1girl, completely nude, bedroom, detailed"]
            width, height, hr_upscale_to_x, hr_upscale_to_y = 64, 64, 128, 128
            sd_model = None

        s = {**K.DEFAULTS, "enabled": True, "profile": "Realistic", "sliders": {"breasts": -1},
             "who": {"breasts": "Everyone"}}
        plan, conds = E.prepare(P(), s, K.load_table(), log=lambda *_a: None)
        assert all(isinstance(c, list) and len(c) == 2 for c in conds)
        assert any(t.startswith("1girl, completely nude, bedroom") for t in seen["texts"])
        cross, _ = host.stack_concepts(conds, 2, "cpu")
        assert cross.shape[0] == 2 * len(conds) and not torch.equal(cross[0], cross[1])   # row 1 != row 2
        E.prepare(P(), {**s, "hires": True}, K.load_table(), hr=True, log=lambda *_a: None)
        assert all(t.startswith(("1girl, completely nude, beach, detailed", "1girl, completely nude, bedroom, detailed"))
                   for t in seen["texts"])
    finally:
        E.encode = orig


def test_prepare_gives_each_image_its_pose_variant():
    """Pose style: a batch of the same prompt gets one pose variant per image, by its seed."""
    seen = {}

    def fake_encode(p, texts, width, height):
        seen["texts"] = list(texts)
        return [{"crossattn": torch.zeros(1, 4, 4), "vector": None} for _ in texts]

    orig = E.encode
    E.encode = fake_encode
    try:
        class P:
            prompt = "1girl, solo, nude"
            prompts = ["1girl, solo, nude"] * 3
            seeds = [100, 101, 102]
            width, height = 64, 64
            sd_model = None

        s = {**K.DEFAULTS, "enabled": True, "profile": "Anime", "sliders": {"pose-style": 2}, "attn_mask": False}
        plan, conds = E.prepare(P(), s, K.load_table(), log=lambda *_a: None)
        toward = next(i for i, c in enumerate(plan.concepts) if c.role == "toward")
        texts = seen["texts"][toward * 3:toward * 3 + 3]
        assert len(set(texts)) == 3 and isinstance(conds[toward], list)       # three images, three poses
        plan2, _ = E.prepare(P(), s, K.load_table(), log=lambda *_a: None)
        assert seen["texts"][toward * 3:toward * 3 + 3] == texts             # same seeds, same poses
    finally:
        E.encode = orig


def test_concept_entries_keep_template_keys():
    template = {"model_conds": {"c_crossattn": Cond(torch.zeros(1, 77, 8)), "y": Cond(torch.zeros(1, 5)),
                                "c_concat": "inpaint"},
                "area": (1, 1, 0, 0), "strength": 0.5, "mask": "m", "control": "cn"}
    cross, vec = host.stack_concepts([{"crossattn": torch.ones(1, 77, 8), "vector": torch.ones(1, 5)},
                                      {"crossattn": torch.ones(1, 154, 8) * 2, "vector": torch.ones(1, 5) * 2}], 3, "cpu")
    assert cross.shape == (6, 154, 8) and vec.shape == (6, 5)
    assert torch.all(cross[:3] == 1) and torch.all(cross[3:] == 2)  # rows [c0 x B, c1 x B]
    [entry] = host.concept_entries(template, cross, vec)
    assert entry["control"] == "cn" and entry["model_conds"]["c_concat"] == "inpaint"
    assert "area" not in entry and "strength" not in entry and "mask" not in entry
    assert entry["model_conds"]["c_crossattn"] is cross
    assert template["model_conds"]["c_crossattn"].t.shape == (1, 77, 8)    # template untouched


def test_chains_with_an_existing_cfg_function():
    """The WebUI's order: sampler_cfg_function, then each post-CFG function in turn.
    SEGA adds on top of whatever the custom CFG function returned."""
    calls = []
    orig = host.predict
    host.predict = fake_predict(calls)
    try:
        eng, _ = make_engine(["a"], [-1.0])
        args = post_args(batch=1)
        x, c, u = args["input"], args["cond_denoised"], args["uncond_denoised"]

        def custom_cfg(a):              # e.g. a rescaled CFG from another extension (returns noise)
            return x - (u + (c - u) * 3.0) * 0.9

        options = {"sampler_cfg_function": custom_cfg, "sampler_post_cfg_function": [eng.post_cfg]}
        result = x - options["sampler_cfg_function"](args)
        for fn in options["sampler_post_cfg_function"]:
            result = fn({**args, "denoised": result})
        without = x - custom_cfg(args)
        eng2, _ = make_engine(["a"], [-1.0])
        eng2.conds = eng.conds
        expected = eng2.post_cfg({**args, "denoised": without})
        assert torch.allclose(result, expected) and not torch.allclose(result, without)
    finally:
        host.predict = orig


def test_overlap_budget_is_per_pixel():
    """The crotch test: nine sliders on the same pixels added up into metal and fire. Where
    sliders overlap, their sum is capped at overlap_budget x the strongest one there;
    where a slider is alone, it is untouched."""
    torch.manual_seed(1)
    d = torch.randn(1, 4, 8, 8)
    left = torch.zeros(1, 1, 8, 8)
    left[..., :4] = 1.0
    a, b, c = d * left, d * left * 0.9, d * (1 - left)          # a and b overlap; c is alone
    lam, live = [0.0] * 3, [True] * 3
    free = Sega(budget=0).guidance_from_diffs([a, b, c], [1.0] * 3, lam, live)
    capped = Sega(budget=0).guidance_from_diffs([a, b, c], [1.0] * 3, lam, live, owners=["a", "b", "c"],
                                                overlap_budget=1.5)
    assert torch.allclose(free, a + b + c, atol=1e-5)
    assert torch.allclose(capped * left, a * 1.5, atol=1e-4)     # 1.9 x d brought down to 1.5 x d
    assert torch.allclose(capped * (1 - left), c, atol=1e-5)     # alone: untouched
    # one slider's own concepts (toward, opposite, cap) are not capped against each other
    same = Sega(budget=0).guidance_from_diffs([d, d], [1.0, 1.0], [0.0] * 2, [True] * 2, owners=["a", "a"],
                                              overlap_budget=1.5)
    assert torch.allclose(same, d * 2, atol=1e-4)
    # with momentum the result stays within the cap too
    s = Sega(budget=0)
    for _ in range(4):
        g = s.guidance_from_diffs([a, b, c], [1.0] * 3, lam, live, owners=["a", "b", "c"], overlap_budget=1.5)
    norm = lambda t: t.pow(2).sum(1).sqrt()                      # noqa: E731
    assert torch.all(norm(g) <= norm(a + c) * 1.5 + 1e-4)          # the strongest single slider per pixel


def test_sign_election_where_sliders_disagree():
    """Opposite pushes on the same latent values no longer cancel into mush: the minority is dropped."""
    torch.manual_seed(2)
    d = torch.randn(1, 4, 8, 8)
    a, b = d, -0.5 * d                                            # b pushes against a everywhere
    s = Sega(budget=0)
    g = s.guidance_from_diffs([a, b], [1.0, 1.0], [0.0] * 2, [True] * 2, owners=["a", "b"], overlap_budget=1.5)
    assert torch.allclose(g, a, atol=1e-5)                        # not 0.5 * d
    c = 0.5 * d                                                   # agreeing: summed, then capped
    g2 = Sega(budget=0).guidance_from_diffs([a, c], [1.0, 1.0], [0.0] * 2, [True] * 2, owners=["a", "c"],
                                            overlap_budget=1.5)
    assert torch.allclose(g2, 1.5 * d, atol=1e-4)
    assert s.overlap is not None and s.overlap["max_sliders"] == 2


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
