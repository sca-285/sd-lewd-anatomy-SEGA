"""Concept planning, profiles, scene detection and infotext (no torch needed).

    python tests/test_concepts.py      or      python -m pytest tests
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib_lewd_anatomy import concepts as K  # noqa: E402

T = K.load_table()
BY = K.slider_by_key(T)


def settings(**sliders):
    """Slider settings; the automatic nudity lock is left out (its own test below), since most
    tests use a nude prompt and count the concepts of one slider."""
    who = sliders.pop("who", {})
    return {**K.DEFAULTS, "enabled": True, "sliders": {k.replace("_", "-"): v for k, v in sliders.items()},
            "who": who, "_no_lock": True}


def by_text(plan):
    return {(c.text, c.sign): c for c in plan.concepts}


def gain(v):
    return float(T["step_gain"][str(abs(v))])


def test_every_slider_is_five_steps():
    for s in T["sliders"]:
        if K.one_sided(s):
            continue
        assert K.slider_range(s) == (-2, 2), s["key"]
        names = K.step_names(s)
        assert sorted(names) == [-2, -1, 1, 2]
        for v, n in names.items():
            assert K.level_name(s, v) == n and K.level_value(s, n) == v
        assert K.level_name(s, 0) is None
    # out-of-range and fractional values snap to the five steps
    assert K.step(-7) == -2 and K.step(3.4) == 2 and K.step(-0.4) == 0 and K.step(None) == 0
    assert K.level_value(BY["penis"], "-1") == -1 and K.level_value(BY["penis"], "off") == 0


def test_negative_steps_compute_smaller_concepts():
    """The bug report: the - side must exist and push toward the smaller entry."""
    for key in ("breasts", "penis", "testicles", "vulva", "waist", "hips", "body-fat"):
        for v in (-1, -2):
            p = K.plan(settings(**{key.replace("-", "_"): v}, who={key: "Everyone"}), T, "realistic", "")
            target = K.entry(K.profile_concepts(T, BY[key], "realistic")[K.step_names(BY[key])[v]])
            assert any(c.text == target["text"] for c in p.concepts), (key, v)
            assert not any("cap" in c.text for c in p.concepts)


def test_at_most_three_concepts_per_slider():
    for s in T["sliders"]:
        for v in (-2, -1, 1, 2):
            for profile in ("realistic", "anime", "stylized"):
                p = K.plan({**settings(), "sliders": {s["key"]: v}, "max_concepts": 0}, T, profile, "")
                assert len(p.concepts) <= 3, (s["key"], v, profile, [c.text for c in p.concepts])
                if v < 0:
                    assert len(p.concepts) <= 2


def test_steps_grow_steadily():
    one = by_text(K.plan(settings(penis=1), T, "realistic", "1boy"))
    two = by_text(K.plan(settings(penis=2), T, "realistic", "1boy"))
    r = gain(2) / gain(1)
    assert 1.5 <= r <= 3.0, r          # a step up, not the old 4x jump
    assert ("large penis", 1.0) in one and ("huge penis, very long penis", 1.0) in two
    assert one[("small penis", -1.0)].scale * r == two[("small penis", -1.0)].scale


def test_breasts_plus_one_roles():
    c = by_text(K.plan(settings(breasts=1), T, "anime", "1girl, solo"))
    roles, base = T["roles"], T["base_scale"]
    g = gain(1)
    assert c[("large breasts", 1.0)].scale == base * roles["toward"] * g
    assert c[("small breasts", -1.0)].scale == base * roles["opposite"] * g        # the -1 entry
    assert c[("gigantic breasts", -1.0)].scale == base * roles["cap"] * g
    assert len(c) == 3


def test_neutral_computes_nothing():
    s = settings()
    assert not K.active(s, T)
    assert K.plan(s, T, "realistic").concepts == []
    s["enabled"] = False
    s["sliders"]["breasts"] = 2
    assert not K.active(s, T)


def test_allow_exaggerated_drops_caps():
    s = settings(penis=2)
    assert any("gigantic penis" in c.text for c in K.plan(s, T, "realistic").concepts)
    s["allow"] = True
    assert not any("gigantic penis" in c.text for c in K.plan(s, T, "realistic").concepts)


def test_max_concepts_rotates_instead_of_dropping():
    many = settings(breasts=2, penis=2, hips=2, waist=-2, muscle=2, body_fat=-1,
                    who={"breasts": "Everyone", "penis": "Everyone"})
    full = K.plan({**many, "max_concepts": 0}, T, "realistic", "")
    capped = K.plan({**many, "max_concepts": 6}, T, "realistic", "")
    # only caps go (they matter past the size a step asks for); every slider still pushes
    assert len(full.concepts) > 6 and any("take turns" in n for n in capped.notes)
    gone = [c for c in full.concepts if c not in capped.concepts]
    assert gone and all(c.role == "cap" for c in gone)
    assert {c.slider for c in capped.concepts if c.scale > 0} == {c.slider for c in full.concepts if c.scale > 0}
    assert K.effective_rotate({**many, "max_concepts": 6}, len(capped.concepts)) == 6
    assert K.effective_rotate({**many, "max_concepts": 6, "rotate": 3}, 20) == 3     # an explicit Rotate wins
    assert K.effective_rotate({**many, "max_concepts": 6}, 5) == 0                    # under the cap: no rotation
    assert K.passes_per_step(capped.concepts, {**many, "max_concepts": 6}, T) <= 6 + 1e-9
    assert K.model_calls(6, 1) == 6 and K.model_calls(6, 4) == 2 and K.model_calls(0, 2) == 0

def test_who_in_mixed_scene_is_away_only():
    s = settings(breasts=1, who={"breasts": "Women"})
    mixed = K.plan(s, T, "anime", "1boy, 1girl, hetero")
    pushing = [c for c in mixed.concepts if c.scale > 0]
    assert pushing and all(c.sign < 0 for c in pushing)
    # the blocked toward concept is still computed, at scale 0, as the away pushes' reference
    assert [c.text for c in mixed.concepts if c.scale == 0] == ["large breasts"]
    assert any("push away only" in n for n in mixed.notes)
    alone = K.plan(s, T, "anime", "1girl, solo")
    assert any(c.sign > 0 for c in alone.concepts)
    everyone = K.plan(settings(breasts=1, who={"breasts": "Everyone"}), T, "anime", "1boy, 1girl")
    assert any(c.sign > 0 for c in everyone.concepts)


def test_masc_fem():
    p = K.plan(settings(masc_fem=-2), T, "realistic", "a man standing")
    assert any(c.sign > 0 and "adam's apple" in c.text for c in p.concepts)
    assert any(c.sign < 0 and "wide hips" in c.text for c in p.concepts)
    weak = K.plan(settings(masc_fem=-1), T, "realistic", "a man")
    assert max(c.scale for c in weak.concepts) < max(c.scale for c in p.concepts)
    mixed = K.plan(settings(masc_fem=2), T, "realistic", "a man and a woman")
    assert mixed.concepts == [] and mixed.notes


def test_entries_written_as_away():
    p = K.plan(settings(body_hair=-2), T, "anime", "")
    assert [(c.text, c.sign) for c in p.concepts if c.text == "chest hair, navel hair"] == [("chest hair, navel hair", -1.0)]
    plus = K.plan(settings(body_hair=2), T, "anime", "")
    assert any("chest hair" in c.text and c.sign > 0 for c in plus.concepts)


def test_stylized_profile_and_old_pony():
    st = K.plan(settings(breasts=1, who={"breasts": "Everyone"}), T, "stylized", "")
    assert any(c.text == "large breasts" and c.sign > 0 for c in st.concepts)
    assert K.resolve_profile("Pony", "x", T) == "anime"
    assert K.from_infotext("profile=pony, breasts=full", T)["profile"] == "Anime"
    assert "Pony" not in K.PROFILE_CHOICES and "Stylized" in K.PROFILE_CHOICES


def test_strength_scales_everything():
    one = K.plan(settings(breasts=1, v_line=2), T, "realistic", "")
    s = settings(breasts=1, v_line=2)
    s["strength"] = 0.5
    half = K.plan(s, T, "realistic", "")
    for a, b in zip(one.concepts, half.concepts):
        assert abs(b.scale - a.scale / 2) < 1e-9
    s["strength"] = 0
    assert K.plan(s, T, "realistic", "").concepts == []


def test_baseline_switch():
    assert K.baseline({}) == "Prompt" and K.baseline({"baseline": "negative"}) == "Negative"
    assert K.baseline({"baseline": "bogus"}) == "Prompt"
    scene = "1girl, nude, standing"
    assert K.concept_text({"baseline": "Prompt"}, scene, "small breasts") == scene + ", small breasts"
    # Negative: the bare phrase, led by the prompt's own nudity words (never a clothed stock photo)
    assert K.concept_text({"baseline": "Negative"}, scene, " small  breasts, ") == "nude, small breasts"
    assert K.concept_text({"baseline": "Negative"}, "1girl, standing", "small breasts") == "small breasts"
    s = settings(breasts=1)
    assert "baseline" not in K.to_infotext(s, T, "anime")                 # default stays out of the infotext
    text = K.to_infotext({**s, "baseline": "Negative"}, T, "anime")
    assert "baseline=negative" in text and K.from_infotext(text, T)["baseline"] == "Negative"
    assert K.from_infotext("breasts=full", T)["baseline"] == "Prompt"


def test_armpit_hair():
    s = BY["armpit-hair"]
    assert K.step_names(s) == {-2: "none", -1: "sparse", 1: "natural", 2: "hairy"}
    none = K.plan(settings(armpit_hair=-2), T, "anime", "")
    assert [(c.text, c.sign) for c in none.concepts] == [("armpit hair", -1.0)]   # away, and +1 merged in
    hairy = by_text(K.plan(settings(armpit_hair=2), T, "realistic", ""))
    assert ("hairy armpits, dense armpit hair", 1.0) in hairy and ("sparse armpit stubble", -1.0) in hairy
    assert ("extremely hairy armpits", -1.0) in hairy


def test_away_pushes_are_contrastive():
    """'Away from large breasts' is measured from 'small breasts', not from the plain prompt:
    both texts show the same body, so nudity cancels and only size is left."""
    p = K.plan(settings(breasts=-1, who={"breasts": "Everyone"}), T, "realistic", "1girl, nude")
    refs = K.ref_indices(p.concepts)
    by = {c.text: (c, r) for c, r in zip(p.concepts, refs)}
    toward, r0 = by["small natural breasts"]
    away, r1 = by["large natural breasts"]
    assert toward.sign > 0 and r0 is None
    assert away.sign < 0 and p.concepts[r1] is toward
    plus = K.plan(settings(penis=2, who={"penis": "Everyone"}), T, "anime", "1boy, nude")
    refs = K.ref_indices(plus.concepts)
    for c, r in zip(plus.concepts, refs):
        if c.sign < 0:                         # opposite step and cap
            assert r is not None and plus.concepts[r].text == "huge penis", c.text


def test_nothing_pushes_away_from_nudity():
    """The clothes bug. In every mode, an away concept is either contrastive (has a
    reference that shares its nudity) or a pure 'remove this' entry measured against
    a baseline that already contains everything else."""
    nude = "1girl, 1boy, completely nude, hetero"
    for baseline_ in ("Prompt", "Negative"):
        for sl in T["sliders"]:
            for v in (-2, -1, 1, 2):
                for profile in ("realistic", "anime", "stylized"):
                    st = {**settings(), "sliders": {sl["key"]: v}, "baseline": baseline_, "max_concepts": 0}
                    p = K.plan(st, T, profile, nude)
                    refs = K.ref_indices(p.concepts)
                    for c, r in zip(p.concepts, refs):
                        if c.sign > 0 or c.scale == 0:
                            continue
                        if baseline_ == "Negative":
                            # measured against the nudity words or a sibling concept, never the bare negative
                            assert r is not None, (sl["key"], v, profile, c.text)
                        else:
                            entry = next(e for raw in K.profile_concepts(T, sl, profile).values()
                                         for e in K.entries(raw) if e["text"] == c.text)
                            assert r is not None or entry["direction"] == "away", (sl["key"], v, profile, c.text)


def test_negative_baseline_nudity_reference():
    st = {**settings(body_hair=-2), "baseline": "Negative"}
    p = K.plan(st, T, "anime", "1girl, completely nude")
    texts = [(c.text, c.sign, c.scale) for c in p.concepts]
    assert (K.NUDITY_REF, 1.0, 0.0) in texts
    assert K.concept_text(st, "1girl, completely nude", K.NUDITY_REF) == "completely nude"
    # no nudity words in the prompt: classic SEGA, no extra pass
    q = K.plan(st, T, "anime", "1girl, standing")
    assert all(c.text != K.NUDITY_REF for c in q.concepts)
    # Prompt baseline never needs it
    r = K.plan({**st, "baseline": "Prompt"}, T, "anime", "1girl, completely nude")
    assert all(c.text != K.NUDITY_REF for c in r.concepts)


def test_every_reference_resolves():
    many = settings(breasts=2, penis=2, hips=2, waist=-2, muscle=2, body_fat=-1,
                    who={"breasts": "Everyone", "penis": "Everyone"})
    for limit in (2, 6, 0):
        p = K.plan({**many, "max_concepts": limit}, T, "realistic", "")
        for c, r in zip(p.concepts, K.ref_indices(p.concepts)):
            if c.ref is not None:
                assert r is not None, (limit, c.text)       # never measured against a missing concept

def test_nudity_context():
    assert K.nudity_context("1girl, completely_nude, nipples, (nude:1.2), standing") == "completely nude, nipples, nude"
    assert K.nudity_context("1girl, red dress") == ""


def test_new_sliders():
    for key in ("shoulders", "back", "lats", "biceps", "butt", "anus"):
        s = BY[key]
        assert K.slider_range(s) == (-2, 2) and s.get("phase") in ("shape", "detail")
        for profile in ("realistic", "anime", "stylized"):
            for v in (-2, -1, 1, 2):
                p = K.plan({**settings(), "sliders": {key: v}}, T, profile, "1girl, nude")
                assert [c for c in p.concepts if c.scale > 0], (key, profile, v)
    assert all(s.get("phase") in ("shape", "detail", "fix", "pose") for s in T["sliders"])


def test_rotate_and_staggered_cost():
    many = settings(breasts=2, penis=2, hips=2, waist=-2, muscle=2, body_fat=-1, butt=1, body_hair=-1,
                    pubic_hair=-2, abs=2, who={"breasts": "Everyone", "penis": "Everyone"})
    p = K.plan({**many, "max_concepts": 0}, T, "realistic", "")
    n = len(p.concepts)
    assert n >= 12
    detail = sum(1 for c in p.concepts if c.phase == "detail")
    assert K.passes_per_step(p.concepts, {**many, "rotate": 0, "max_concepts": 0}, T) <= n   # detail ones run longer
    # detail concepts now wait for the shape ones in their region: fewer run at once
    assert K.passes_per_step(p.concepts, {**many, "rotate": 0, "max_concepts": 0}, T) >= 0.5 * (n - detail)
    assert K.passes_per_step(p.concepts, {**many, "rotate": 4}, T) <= 4 + 1e-9
    stag = K.passes_per_step(p.concepts, {**many, "staggered": True, "max_concepts": 0}, T)
    assert stag < n                                   # shape and detail rarely run together
    wins = K.windows(p.concepts, {**many, "staggered": True}, T)
    a, b = K.DEFAULTS["start"], K.DEFAULTS["end"]
    assert all(a - 1e-9 <= lo < hi <= 1.0 for lo, hi in wins)
    for c, (lo, hi) in zip(p.concepts, K.windows(p.concepts, {**many, "staggered": False}, T)):
        if c.phase == "detail":
            # past the range end; from the start, or (waiting for a shape on its part) later
            assert lo >= a - 1e-9 and hi > b
        elif c.phase == "shape":
            assert (lo, hi) == (a, b)                 # without Staggered: the whole range
    text = K.to_infotext({**many, "rotate": 4, "staggered": True}, T, "realistic")
    assert "rotate" not in text and "staggered" not in text           # fixed in 2.0, not written
    assert K.from_infotext("breasts=full, rotate=4", T)["rotate"] == 0


def test_concept_mask_infotext():
    base = settings(breasts=1)
    assert "mask" not in K.to_infotext(base, T, "anime") and "area" not in K.to_infotext(base, T, "anime")
    off = K.to_infotext({**base, "attn_mask": False}, T, "anime")
    assert "mask=off" in off and K.from_infotext(off, T)["attn_mask"] is False
    area = K.to_infotext({**base, "attn_area": 0.35}, T, "anime")
    back = K.from_infotext(area, T)
    assert "area=0.35" in area and back["attn_mask"] is True and abs(back["attn_area"] - 0.35) < 1e-9
    assert K.from_infotext("breasts=full", T)["attn_mask"] is True


def test_weight_and_the_newest_sliders():
    w = BY["body-fat"]
    assert w["label"] == "Weight" and K.step_names(w) == {-2: "skinny", -1: "slim", 1: "chubby", 2: "heavy"}
    # infotexts from the Body fat version still paste back
    assert [K.level_value(w, n) for n in ("lean", "slim", "curvy", "plump")] == [-2, -1, 1, 2]
    assert K.from_infotext("body-fat=curvy", T)["sliders"]["body-fat"] == 1
    for profile in ("realistic", "anime", "stylized"):            # one meaning: thin <-> heavy, no curves
        texts = " ".join(K.entry(e)["text"] for e in K.profile_concepts(T, w, profile).values())
        assert "curvy" not in texts and "waist" not in texts, profile
    for key in ("nipples", "breast-shape", "thighs", "foreskin", "limb-hair"):
        s = BY[key]
        assert s.get("phase") in ("shape", "detail") and K.slider_range(s) == (-2, 2)
        for profile in ("realistic", "anime", "stylized"):
            for v in (-2, -1, 1, 2):
                p = K.plan({**settings(), "sliders": {key: v}, "who": {key: "Everyone"}}, T, profile, "1girl, 1boy, nude")
                assert [c for c in p.concepts if c.scale > 0], (key, profile, v)
    assert BY["breast-shape"]["who"] == "Women" and BY["foreskin"]["who"] == "Men"


def tables(s):
    """(profile, step table) of a slider; a 'by_scene' slider has one table per scene."""
    for profile, entries in s["concepts"].items():
        if s.get("by_scene") and entries:
            for scene, sub in entries.items():
                yield f"{profile}/{scene}", sub
        else:
            yield profile, entries


def all_texts(e):
    return list(e.get("variants") or [e["text"]])


def test_every_profile_has_all_four_steps():
    """A profile listed for a slider replaces its fallback entirely: it must name all steps."""
    for s in T["sliders"]:
        names = set(K.step_names(s).values())
        for profile, entries in tables(s):
            # {} = no fitting concept in that profile: the slider is off there
            assert entries == {} or names <= set(entries), (s["key"], profile, names - set(entries))


def test_arousal_detail_and_fix_sliders():
    for key in ("belly", "erection", "vulva-wetness", "nipple-state", "flush", "sweat", "penis-veins", "scrotum",
                "vulva-texture"):
        s = BY[key]
        assert K.slider_range(s) == (-2, 2) and s.get("phase") in ("shape", "detail"), key
        for profile in ("realistic", "anime", "stylized"):
            for v in (-2, -1, 1, 2):
                p = K.plan({**settings(), "sliders": {key: v}, "who": {key: "Everyone"}}, T, profile, "1girl, 1boy, nude")
                if s["concepts"].get(profile) == {}:
                    assert not p.concepts and any("(no tag)" in n for n in p.notes), (key, profile)
                else:
                    assert [c for c in p.concepts if c.scale > 0], (key, profile, v)
    # Danbooru's "erect nipples" is an alias of covered_nipples: it would paint a shirt on
    assert BY["nipple-state"]["concepts"]["anime"] == {}
    # flush and sweat never push toward a skin colour on the - side
    for key in ("flush", "sweat"):
        for profile in ("realistic", "anime"):
            for name in BY[key]["below"]:
                assert all(e["direction"] == "away" for e in K.entries(BY[key]["concepts"][profile][name])), (key, name)


def test_anatomy_fix_is_one_sided_and_only_pushes_away():
    s = BY["anatomy-fix"]
    assert K.one_sided(s) and K.slider_range(s) == (0, 2) and K.step_names(s) == {1: "fix", 2: "strong fix"}
    assert K.level_name(s, -2) is None and K.level_value(s, "-1") == 0 and K.level_value(s, "strong fix") == 2
    for profile in ("realistic", "anime", "stylized"):
        assert not K.plan({**settings(), "sliders": {"anatomy-fix": -2}}, T, profile, "1girl, nude").concepts
        one = K.plan({**settings(), "sliders": {"anatomy-fix": 1}, "max_concepts": 0}, T, profile, "1girl, nude")
        two = K.plan({**settings(), "sliders": {"anatomy-fix": 2}, "max_concepts": 0}, T, profile, "1girl, nude")
        assert len(one.concepts) == 3 and all(c.sign < 0 and c.ref is None for c in one.concepts), profile
        assert sum(c.scale for c in two.concepts) == 2 * sum(c.scale for c in one.concepts)
    text = K.to_infotext({**settings(), "sliders": {"anatomy-fix": 2}}, T, "realistic")
    assert "anatomy-fix=strong fix" in text
    assert K.from_infotext(text, T)["sliders"]["anatomy-fix"] == 2


def test_control_mode_settings():
    assert K.control_mode({}) == "balanced"
    assert K.control_mode({"control": "Prompt"}) == "prompt" and K.control_mode({"control": "Sliders"}) == "sliders"
    # 1.x labels and codes
    assert K.control_mode({"control": "My prompt is more important"}) == "prompt"
    assert K.control_mode({"control": "SEGA is more important"}) == "sliders"
    assert K.orthogonal_on({"control": "Prompt"}) and not K.orthogonal_on({"control": "Sliders"})
    assert K.control_gain(T, "sliders") > K.control_gain(T, "balanced") == 1.0 > K.control_gain(T, "prompt")
    for mode in K.CONTROL_CHOICES:
        st = {**settings(breasts=1), "control": mode}
        text = K.to_infotext(st, T, "realistic")
        assert ("control=" in text) == (mode != "Balanced")
        assert K.from_infotext(text, T)["control"] == mode
    assert all(len(c) <= 8 for c in K.CONTROL_CHOICES)               # short labels


def test_clashing_sliders_lose_their_opposite_push():
    st = {**settings(belly=2, abs=2), "who": {"belly": "Everyone", "abs": "Everyone"}, "max_concepts": 0}
    assert K.clashes(st, T) == [("belly", "abs")]
    p = K.plan(st, T, "realistic", "1girl, nude")
    assert not [c for c in p.concepts if c.role == "opposite"]
    assert {c.slider for c in p.concepts if c.role == "toward" and c.scale > 0} == {"belly", "abs"}
    assert any("clash" in n for n in p.notes)
    # the same sliders the compatible way: untouched
    ok = {**settings(belly=2, abs=-2), "max_concepts": 0}
    assert K.clashes(ok, T) == []
    assert [c for c in K.plan(ok, T, "realistic", "1girl, nude").concepts if c.role == "opposite"]
    for rule in T["clashes"]:
        assert all(k in BY for k in rule["sliders"]), rule


def test_regions_and_anatomy_fix_window():
    p = K.plan({**settings(belly=1, abs=1, breasts=1), "max_concepts": 0}, T, "realistic", "1girl, nude")
    assert BY["belly"]["region"] == BY["abs"]["region"] == "belly" and BY["breasts"]["region"] == "chest"
    fix = K.plan({**settings(), "sliders": {"anatomy-fix": 2}}, T, "realistic", "1girl, nude")
    st = {**K.DEFAULTS, "start": 0.1, "end": 0.6}
    for lo, hi in K.windows(fix.concepts, st, T):
        assert abs(lo - (0.1 + 0.5 * 0.3)) < 1e-9 and hi <= 0.6 + 0.5 * 0.15 + 1e-9
    assert K.mask_areas(fix.concepts, T, 0.2) == [0.12] * 3
    # its chest / genital push stays out while a slider shapes that part
    with_breasts = K.plan({**settings(breasts=1), "sliders": {"anatomy-fix": 1, "breasts": 1},
                           "who": {"breasts": "Everyone"}, "max_concepts": 0}, T, "realistic", "1girl, nude")
    fix_texts = [c.text for c in with_breasts.concepts if c.slider == "anatomy-fix"]
    assert len(fix_texts) == 2 and not any("nipples" in t for t in fix_texts)
    assert any("skips" in n for n in with_breasts.notes)
    # no global quality tag that re-draws the whole picture
    for e in K.entries(BY["anatomy-fix"]["concepts"]["realistic"]["fix"]):
        assert "bad anatomy" not in e["text"] and "deformed" not in e["text"]


def test_small_parts_are_late_and_tight():
    """The vulva test: early in the steps 'labia' attends over the whole picture; an edit there
    re-coloured the hair and moved a hand. Small parts run late, with a small mask."""
    for key in ("vulva", "vulva-texture", "vulva-wetness", "anus", "nipples", "nipple-state",
                "penis-veins", "scrotum", "foreskin"):
        assert BY[key]["phase"] == "detail" and float(BY[key]["area"]) <= 0.12, key
    p = K.plan({**settings(vulva=2), "who": {"vulva": "Everyone"}, "max_concepts": 0}, T, "anime", "1girl, nude")
    st = {**K.DEFAULTS, "start": 0.1, "end": 0.6}
    assert all(lo >= 0.1 + 0.5 * 0.3 - 1e-9 for lo, _ in K.windows(p.concepts, st, T))
    assert K.mask_areas(p.concepts, T, 0.2) == [0.1] * len(p.concepts)


def test_pose_tab():
    for key in ("pose-energy", "camera-height", "head-turn"):
        s = BY[key]
        assert s["phase"] == "pose" and s["scope"] == "global" and K.slider_range(s) == (-2, 2), key
        for profile in ("realistic", "anime", "stylized"):
            for v in (-2, -1, 1, 2):
                p = K.plan({**settings(), "sliders": {key: v}, "max_concepts": 0}, T, profile, "1girl, nude")
                assert [c for c in p.concepts if c.scale > 0], (key, profile, v)
                assert all(K.global_scope(p.concepts, T))
                # the first steps of the schedule, whatever Range start says
                for lo, hi in K.windows(p.concepts, {**K.DEFAULTS, "start": 0.3, "end": 0.8}, T):
                    assert (lo, hi) == (0.0, 0.35)
    # + pushes away from the stock pose (contrastive, measured from the toward concept)
    p = K.plan({**settings(), "sliders": {"pose-energy": 2}, "max_concepts": 0}, T, "realistic", "1girl, nude")
    away = [c for c in p.concepts if c.role == "opposite"]
    assert away and "arms at sides" in away[0].text and away[0].ref is not None
    # body sliders stay local
    b = K.plan({**settings(breasts=1), "who": {"breasts": "Everyone"}}, T, "realistic", "1girl, nude")
    assert not any(K.global_scope(b.concepts, T))


def test_protect_face_covers_pose_energy_and_camera():
    p = K.plan({**settings(), "sliders": {"pose-energy": 1, "camera-height": 1, "head-turn": 1},
                "max_concepts": 0}, T, "realistic", "1girl")
    keep = dict(zip([c.slider for c in p.concepts], K.keep_face(p.concepts, T, {})))
    assert keep["pose-energy"] and keep["camera-height"] and not keep["head-turn"]
    assert not any(K.keep_face(p.concepts, T, {"protect_face": False}))
    # an earlier pose_face key pastes without error and changes nothing
    assert "pose_face" not in K.from_infotext("breasts=full, pose_face=off", T)


def test_nudity_lock_is_automatic():
    full = ("1girl, nude", "1girl, completely nude, bed", "naked woman lying on a bed", "1boy, fully nude")
    keep = ("1girl, topless, skirt", "1boy, shirtless", "1girl, partially nude", "1girl, nude, 1boy, suit",
            "1girl, bare-chested", "1girl, nude lipstick, dress", "1girl, red dress", "1girl, nude, panties",
            "cfnm, 1girl, nude, 1boy")
    for p in full:
        assert K.nudity_lock(p), p
    for p in keep:
        assert not K.nudity_lock(p), p
    st = {**settings(breasts=1), "who": {"breasts": "Everyone"}}
    st.pop("_no_lock")
    on = K.plan(st, T, "realistic", "1girl, completely nude")
    lock = [c for c in on.concepts if c.slider == K.NUDITY_LOCK]
    assert len(lock) == 1 and lock[0].sign < 0 and "clothes" in lock[0].text
    assert any("Nudity lock" in n for n in on.notes)
    assert not [c for c in K.plan(st, T, "realistic", "1girl, topless").concepts if c.slider == K.NUDITY_LOCK]
    assert not [c for c in K.plan({**st, "_no_lock": True}, T, "realistic", "1girl, nude").concepts
                if c.slider == K.NUDITY_LOCK]
    # it is what makes the extension act even with every slider at 0
    plain = {**K.DEFAULTS, "enabled": True, "sliders": {}, "who": {}}
    assert K.active(plain, T, "1girl, nude") and not K.active(plain, T, "1girl, topless")
    assert K.mask_areas(lock, T, 0.2) == [0.5] and K.face_policy(lock, T, {}) == ["full"]


def test_facial_hair_keeps_only_eyes_and_lips():
    s = BY["facial-hair"]
    assert s["face"] == "features" and s["who"] == "Men" and s["phase"] == "detail"
    p = K.plan({**settings(facial_hair=2), "who": {"facial-hair": "Everyone"}, "max_concepts": 0}, T, "realistic", "1boy")
    assert p.concepts and set(K.face_policy(p.concepts, T, {})) == {"features"}
    assert set(K.face_policy(p.concepts, T, {"protect_face": True})) == {"features"}
    b = K.plan({**settings(breasts=1), "who": {"breasts": "Everyone"}}, T, "realistic", "1girl")
    assert set(K.face_policy(b.concepts, T, {})) == {"full"}
    h = K.plan({**settings(), "sliders": {"head-turn": 1}}, T, "realistic", "1girl")
    assert set(K.face_policy(h.concepts, T, {})) == {"none"}


def test_skin_realism_and_fluids():
    s = BY["skin-realism"]
    assert s["concepts"]["anime"] == {} and "Anime" in s["note"] and s["face"] == "features"
    for profile in ("realistic", "stylized"):
        p = K.plan({**settings(skin_realism=2), "max_concepts": 0}, T, profile, "1girl")
        assert p.concepts and set(K.face_policy(p.concepts, T, {})) == {"features"}
    off = K.plan({**settings(skin_realism=2)}, T, "anime", "1girl")
    assert not off.concepts and any("(no tag)" in n for n in off.notes)
    for key in ("penis-wetness", "anus-wetness"):
        assert BY[key]["phase"] == "detail" and float(BY[key]["area"]) <= 0.3, key
    # cum lands anywhere, face and hair included: a wide mask, only the face core kept
    assert BY["cum"]["face"] == "features" and float(BY["cum"]["area"]) >= 0.5
    # the crotch pile-up: every one of these is its own owner, so the per-pixel overlap budget
    # caps their sum where they meet
    many = {**settings(penis=2, penis_veins=2, penis_wetness=2, cum=2, erection=2, testicles=2, scrotum=1),
            "who": {k: "Everyone" for k in ("penis", "penis-veins", "penis-wetness", "cum", "erection",
                                            "testicles", "scrotum")}, "max_concepts": 0}
    p = K.plan(many, T, "realistic", "1boy")
    # fused within a phase: penis + erection (shape), veins + wetness (detail); testicles (shape)
    # and scrotum (detail) stay apart; cum alone
    assert {c.slider for c in p.concepts} == {"fuse:penis@shape", "fuse:penis@detail", "testicles", "scrotum", "cum"}
    assert float(T["overlap_budget"]) <= 1.5


def test_warnings():
    # time: a pile of sliders with Max concepts high enough for all of them
    keys = [sl["key"] for sl in T["sliders"] if sl.get("kind") != "shift" and not K.one_sided(sl)][:30]
    st = {**settings(), "sliders": {k: 2 for k in keys}, "who": {k: "Everyone" for k in keys},
          "max_concepts": 40, "start": 0.0, "end": 1.0}
    p = K.plan(st, T, "realistic", "1girl, 1boy")
    assert K.expected_extra(p.concepts, st, T) >= 8.0 and "Slow" in K.time_warning(p.concepts, st, T)
    one = {**settings(breasts=1), "who": {"breasts": "Everyone"}}
    assert K.time_warning(K.plan(one, T, "realistic", "1girl").concepts, one, T) is None
    # many sliders on one region
    chest = {**settings(breasts=1, breast_shape=1, nipples=1, nipple_state=1),
             "who": {k: "Everyone" for k in ("breasts", "breast-shape", "nipples", "nipple-state")}}
    notes = K.plan(chest, T, "realistic", "1girl").notes
    assert not any("on chest" in n for n in notes)          # two fused descriptions, not four edits
    torso = {**settings(belly=1, abs=1, v_line=1, skin_tone=1, skin_realism=1, flush=1, sweat=1),
             "who": {k: "Everyone" for k in ("belly", "abs", "v-line", "skin-tone", "flush", "sweat")}}
    assert any("4 on skin" in n for n in K.plan(torso, T, "realistic", "1girl").notes)


def test_skin_tone_and_nipple_melanin():
    tone = BY["skin-tone"]
    assert tone["face"] == "features" and float(tone["area"]) >= 0.6 and tone["region"] == "skin"
    for profile in ("realistic", "anime"):
        assert "cap" not in tone["concepts"][profile]          # no "too dark" cap on a skin colour
        for v in (-2, -1, 1, 2):
            p = K.plan({**settings(skin_tone=v), "who": {"skin-tone": "Everyone"}, "max_concepts": 0}, T, profile, "1girl")
            assert p.concepts and set(K.face_policy(p.concepts, T, {})) == {"features"}, (profile, v)
    mel = BY["nipple-color"]
    assert mel["phase"] == "detail" and float(mel["area"]) <= 0.12 and mel["region"] == "chest"
    for profile in ("realistic", "anime"):
        for v in (-2, -1, 1, 2):
            p = K.plan({**settings(nipple_color=v), "who": {"nipple-color": "Everyone"}}, T, profile, "1girl")
            assert [c for c in p.concepts if c.scale > 0], (profile, v)


def test_parts_are_fused_into_one_description():
    st = {**settings(breasts=2, breast_shape=-1, nipples=1, nipple_state=2, nipple_color=1),
          "who": {k: "Everyone" for k in ("breasts", "breast-shape", "nipples", "nipple-state", "nipple-color")},
          "max_concepts": 0}
    p = K.plan(st, T, "realistic", "1girl")
    sliders = {c.slider for c in p.concepts}
    assert sliders == {"fuse:breasts@shape", "fuse:nipples@detail"}, sliders
    nip = [c for c in p.concepts if c.slider == "fuse:nipples@detail" and c.role == "toward"][0]
    assert "large areolae" in nip.text and "erect" in nip.text and "brown" in nip.text
    # strength of the strongest member, not the sum
    single = K.plan({**settings(nipple_state=2), "who": {"nipple-state": "Everyone"}}, T, "realistic", "1girl")
    assert abs(nip.scale - max(c.scale for c in single.concepts if c.role == "toward")) < 1e-9
    # its contrastive pushes are measured from the fused description
    for c in p.concepts:
        if c.slider == "fuse:nipples@detail" and c.role in ("opposite", "cap"):
            assert c.ref == nip.text
    assert K.ref_indices(p.concepts)                            # every ref resolves
    assert all(r is not None for c, r in zip(p.concepts, K.ref_indices(p.concepts)) if c.ref)
    # a lone slider stays as it was
    one = K.plan({**settings(breasts=1), "who": {"breasts": "Everyone"}}, T, "realistic", "1girl")
    assert {c.slider for c in one.concepts} == {"breasts"}
    # the fused nipples are a detail concept waiting for the breasts' shape
    w = dict(zip([c.slider for c in p.concepts], K.windows(p.concepts, {**K.DEFAULTS, "start": 0.1, "end": 0.6}, T)))
    assert w["fuse:nipples@detail"][0] >= 0.1 + 0.5 * 0.6 - 1e-9
    assert K.mask_areas([nip], T, 0.2) == [0.1]


def test_layer_order():
    """Shape before detail on the same part (and on the parts a detail slider sits on), the
    pose before the body, and no fused concept that mixes a size push with a detail."""
    st = {**K.DEFAULTS, "start": 0.1, "end": 0.6}
    late = 0.1 + 0.5 * float(T["phases"]["detail_after_shape"]) - 1e-9
    cases = [({"penis": 2, "pubic-hair": 2}, "pubic-hair"), ({"belly": 2, "abs": 1}, "abs"),
             ({"lats": 1, "back": 1}, "back"), ({"body-fat": 2, "skin-realism": 1}, "skin-realism"),
             ({"breasts": 2, "nipple-color": 1}, "nipple-color"), ({"muscle": 2, "armpit-hair": 1}, "armpit-hair")]
    for sliders, detail in cases:
        p = K.plan({**settings(), "sliders": sliders, "who": {k: "Everyone" for k in sliders}, "max_concepts": 0},
                   T, "realistic", "1girl, 1boy")
        w = K.windows(p.concepts, st, T)
        got = [lo for c, (lo, _) in zip(p.concepts, w) if c.slider == detail or detail in c.slider]
        assert got and min(got) >= late, (sliders, detail, got)
    # pose first: shape sliders start once the composition mostly settled
    p = K.plan({**settings(), "sliders": {"pose-energy": 2, "waist": -1}, "who": {"waist": "Everyone"},
                "max_concepts": 0}, T, "realistic", "1girl")
    w = dict((c.slider, win) for c, win in zip(p.concepts, K.windows(p.concepts, st, T)))
    assert w["waist"][0] >= 0.35 * float(T["phases"]["shape_after_pose"]) - 1e-9
    # every body slider has a region, so the staging can see it
    for s in T["sliders"]:
        if s.get("scope") != "global" and s["key"] not in ("anatomy-fix",):
            assert s.get("region") or "*" in (s.get("wait_for") or []), s["key"]   # cum: anywhere


def test_no_surface_words_in_small_part_masks():
    """Wet, glistening, fluid, skin, pink... attend to every shiny surface (a neon scene's
    lights): small parts read their masks from their own nouns."""
    for key in ("vulva-wetness", "penis-wetness", "anus-wetness", "penis-veins", "scrotum", "vulva-texture",
                "nipple-color", "nipple-state", "nipples", "cum"):
        words = BY[key].get("mask_words") or []
        assert words, key
        assert not set(words) & {"wet", "glistening", "fluid", "skin", "pink", "hard", "detailed", "veins"}, key
    for s in T["sliders"]:
        for e in K.entries(s["concepts"].get("realistic", {}).get(s["below"][0] if s.get("below") else "x")):
            pass
    texts = [e["text"] for s in T["sliders"] for raw in s["concepts"].get("realistic", {}).values()
             for e in K.entries(raw)]
    assert not any(re.search(r"\bpussy\b", t) for t in texts)          # realistic: vulva, not 'pussy'
    assert not any(re.search(r"\blight (chest|armpit|arm|body)\b", t) for t in texts)


def test_mask_audit_every_slider_every_step():
    """For every slider, profile and step: refs resolve, windows are sane, a slider with
    mask_words has one of them in every concept text (else its mask would read all tokens),
    and a concept that reads all its tokens (no mask_words, not a body-wide mask) carries no
    surface word that attends to every shiny thing or to the hair on the head."""
    surface = {"wet", "glistening", "fluid", "skin", "pink", "hard", "detailed", "veins", "shiny", "light", "hair"}
    st0 = {**K.DEFAULTS, "enabled": True, "_no_lock": True, "max_concepts": 0}
    found = []
    for sl in T["sliders"]:
        if sl.get("kind") == "shift":
            continue
        lo_, hi_ = K.slider_range(sl)
        for profile in ("realistic", "anime", "stylized"):
            for v in range(lo_, hi_ + 1):
                if v == 0:
                    continue
                p = K.plan({**st0, "sliders": {sl["key"]: v}, "who": {sl["key"]: "Everyone"}}, T, profile, "1girl, 1boy")
                for c, r in zip(p.concepts, K.ref_indices(p.concepts)):
                    if c.ref is not None and r is None:
                        found.append(("ref", sl["key"], profile, v, c.text))
                    conf = K._conf(T, c.slider)
                    words = conf.get("mask_words") or []
                    if words and not any(re.search(r"\b" + re.escape(w) + r"\b", c.text, re.I) for w in words):
                        found.append(("mask word missing", sl["key"], profile, v, c.text))
                    if not words and conf.get("scope") != "global" and float(conf.get("area") or 0) < 0.5:
                        bad = [w for w in surface if re.search(r"\b" + w + r"\b", c.text, re.I)]
                        if bad:
                            found.append(("surface words", sl["key"], profile, v, c.text, bad))
                for lo, hi in K.windows(p.concepts, {**st0, "start": 0.1, "end": 0.6}, T):
                    if not (0.0 <= lo < hi <= 1.0):
                        found.append(("window", sl["key"], profile, v, lo, hi))
    assert not found, found[:8]


def test_tabs_follow_the_run_order():
    """Tabs left to right and sliders top to bottom in the order the layers run: pose, shape,
    fix, detail. Every slider in exactly one tab."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "scripts" / "lewd_anatomy.py").read_text(encoding="utf-8")
    start = src.index("TABS = (")
    ns = {}
    exec(src[start:src.index("\n)\n", start) + 3], ns)
    rank = {"pose": 0, "shape": 1, "fix": 2, "detail": 3}
    listed = [k for _, keys in ns["TABS"] for k in keys]
    assert sorted(listed) == sorted(s["key"] for s in T["sliders"]), "every slider in one tab"
    firsts = []
    for title, keys in ns["TABS"]:
        phases = [rank[BY[k]["phase"]] for k in keys]
        assert phases == sorted(phases), (title, phases)
        firsts.append(phases[0])
    assert firsts == sorted(firsts) and ns["TABS"][0][0].endswith("Pose")


def test_skin_marks():
    for key in ("scars", "moles", "birthmarks", "freckles"):
        s = BY[key]
        assert s["phase"] == "detail" and s["face"] == "features" and s["mask_words"] and "*" in s["wait_for"]
        for profile in ("realistic", "anime", "stylized"):
            for v in (-2, -1, 1, 2):
                p = K.plan({**settings(), "sliders": {key: v}, "who": {key: "Everyone"}, "max_concepts": 0},
                           T, profile, "1girl")
                assert [c for c in p.concepts if c.scale > 0], (key, profile, v)
    assert "subject-position" not in BY                       # removed: it did not move the subject


def test_wrinkles_one_sided():
    s = BY["wrinkles"]
    assert K.one_sided(s) and K.slider_range(s) == (0, 2) and s["face"] == "features"
    for profile in ("realistic", "anime", "stylized"):
        assert not K.plan({**settings(), "sliders": {"wrinkles": -2}}, T, profile, "1girl").concepts
        for v in (1, 2):
            p = K.plan({**settings(), "sliders": {"wrinkles": v}, "who": {"wrinkles": "Everyone"}}, T, profile, "1girl")
            assert p.concepts and all(c.sign > 0 for c in p.concepts if c.scale > 0), (profile, v)


def test_auto_profile():
    assert K.resolve_profile("Auto", "ponyDiffusionV6XL.safetensors", T) == "anime"
    assert K.resolve_profile("Auto", "waiNSFWIllustrious_v14", T) == "anime"
    assert K.resolve_profile("Auto", "noobaiXLNAIXL_vPred", T) == "anime"
    assert K.resolve_profile("Auto", "juggernautXL_v9", T) == "realistic"
    assert K.resolve_profile("Auto", "ponyRealism_v22", T) == "realistic"
    assert K.resolve_profile("Auto", "dreamshaperXL_2.5D", T) == "stylized"
    assert K.resolve_profile("Auto", "pixarStyleModel", T) == "stylized"
    # a cartoon merge on a tag-prompted base (Pony / Illustrious / NoobAI) takes tags
    assert K.resolve_profile("Auto", "ponyCartoonMix_v3", T) == "anime"
    assert K.resolve_profile("Auto", "illustriousToon_v2", T) == "anime"
    assert K.resolve_profile("Auto", "toonYouXL_v6", T) == "stylized"
    assert K.resolve_profile("Anime", "juggernautXL_v9", T) == "anime"


def test_scene_detection():
    assert K.scene_genders("1girl, solo", T) == {"women"}
    assert K.scene_genders("1boy, 1girl", T) == {"women", "men"}
    assert K.scene_genders("a woman in a red dress", T) == {"women"}
    assert K.scene_genders("female focus", T) == {"women"}
    assert K.scene_genders("1girl, futanari", T) == {"women", "futa"}
    alone = K.plan(settings(penis=2), T, "anime", "1girl, solo")
    assert alone.concepts and all(c.sign < 0 for c in alone.concepts if c.scale > 0)
    futa = K.plan(settings(penis=2, testicles=2), T, "anime", "1girl, futanari")
    assert any(c.sign > 0 and "penis" in c.text for c in futa.concepts)
    assert any(c.sign > 0 and "testicles" in c.text for c in futa.concepts)
    assert K.scene_genders("landscape, human figure", T) == set()


def test_infotext_round_trip():
    s = settings(breasts=1, v_line=2, masc_fem=-2, who={"breasts": "Women"})
    s.update(profile="Anime", strength=1.25, locality=0.97, start=0.1, end=0.6, allow=True, hires=True,
             max_concepts=4)
    text = K.to_infotext(s, T, "anime")
    assert text.startswith("profile=anime, strength=1.25, breasts=full(w), ")
    assert "v-line=sharp" in text and "masc-fem=stark" in text and "range=0.10-0.60" in text
    back = K.from_infotext(text, T)
    for k in ("profile", "strength", "locality", "start", "end", "allow", "hires", "max_concepts"):
        assert back[k] == s[k], (k, back[k], s[k])
    assert back["sliders"]["breasts"] == 1 and back["sliders"]["v-line"] == 2 and back["sliders"]["masc-fem"] == -2
    assert back["sliders"]["penis"] == 0 and back["who"]["breasts"] == "Women"
    auto = K.from_infotext(K.to_infotext({**s, "profile": "Auto"}, T, "stylized"), T)
    assert auto["profile"] == "Auto"
    # the WebUI quotes values with commas in the infotext; the quotes are tolerated
    assert K.from_infotext('"' + text + '"', T)["sliders"]["breasts"] == 1


def test_spec_example_parses():
    s = K.from_infotext("profile=anime, strength=1.0, breasts=full(w), v-line=sharp, range=0.15-0.70", T)
    assert s["profile"] == "Anime" and s["sliders"]["breasts"] == 1 and s["sliders"]["v-line"] == 2 and s["start"] == 0.15 and s["end"] == 0.70


def test_slowdown():
    assert K.slowdown(0, 0.15, 0.7) == 0
    assert abs(K.slowdown(2, 0.15, 0.70) - 0.55) < 1e-9


def test_edit_prompt_keeps_the_scene():
    scene = "1girl, nude, standing in a room"
    assert K.edit_prompt(scene, "large natural breasts") == scene + ", large natural breasts"
    assert K.edit_prompt("", "flat chest") == "flat chest"
    assert K.edit_prompt("  1girl,  ", "flat chest") == "1girl, flat chest"
    # already written by the user: don't append a second copy
    assert K.edit_prompt("1girl, large natural breasts", "large natural breasts") == "1girl, large natural breasts"


def test_concepts_do_not_name_clothes_or_a_person():
    """A garment or a whole person in the concept is what made SEGA dress the character."""
    banned = re.compile(
        r"\b(shirt|dress|skirt|pants|trousers|jeans|bra|panties|underwear|clothes|clothing|"
        r"outfit|jacket|coat|bikini|lingerie|nude|naked|topless|woman|women|man|men|girl|boy|"
        r"cinch\w*|corset|belt|waistband|strap\w*|thong|briefs|boxers|trunks|stocking\w*|garter|harness|"
        r"bodybuilder|athlete|model|gym|fitness|swim\w*|bulge|cameltoe|tan ?lines?)\b",
        re.I,
    )
    for s in T["sliders"]:
        assert set(s["concepts"]) <= set(T["profiles"]), s["key"]
        for profile, entries in tables(s):
            for name, raw in entries.items():
                for e in K.entries(raw):
                    for t in all_texts(e):
                        assert not banned.search(t), (s["key"], profile, name, t)


def test_tables_are_valid_and_adult_only():
    age = re.compile(r"\b(loli|shota|child|kid|teen|young|petite|minor|school|underage|aged?)\b")
    for s in T["sliders"]:
        for profile, entries in tables(s):
            for name, raw in entries.items():
                assert K.entries(raw), (s["key"], name)
                for e in K.entries(raw):
                    assert e["direction"] in ("toward", "away")
                    for t in all_texts(e):
                        assert not age.search(t), (s["key"], name, t)
                names = set(K.step_names(s).values()) | {"cap"}
                assert name in names, (s["key"], name)


def test_anime_tags_exist_in_danbooru():
    """Only when a tag-autocomplete danbooru.csv is given in DANBOORU_CSV."""
    import os
    path = os.environ.get("DANBOORU_CSV")
    if not path:
        return
    known = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split(",")
            known.add(parts[0].strip())
            if len(parts) > 3:
                known.update(a.strip().strip('"') for a in ",".join(parts[3:]).split(","))
    # e621 tags that Pony knows and Danbooru has no tag for (weak on Illustrious / NoobAI)
    e621 = {"thick_penis", "puffy_pussy"}
    missing = []
    for s in T["sliders"]:
        found = K.profile_concepts(T, s, "anime")
        subs = list(found.values()) if s.get("by_scene") and found else [found]
        for sub in subs:
            for raw in sub.values():
                for tag in ",".join(t for e in K.entries(raw) for t in all_texts(e)).split(","):
                    tag = tag.strip().replace(" ", "_")
                    if tag not in known and tag not in e621:
                        missing.append(tag)
    assert not missing, missing


def test_review_fixes():
    # merge: one "flat chest" for Breasts -2 and Pectorals -2, at the stronger scale, not the sum
    one = K.plan({**settings(), "sliders": {"breasts": -2}, "who": {"breasts": "Everyone"}, "max_concepts": 0},
                 T, "realistic", "1girl, nude")
    both = K.plan({**settings(), "sliders": {"breasts": -2, "pectorals": -2},
                   "who": {"breasts": "Everyone", "pectorals": "Everyone"}, "max_concepts": 0},
                  T, "realistic", "1girl, nude")
    for c in both.concepts:
        twin = [d for d in both.concepts if (d.text, d.sign, d.ref, d.phase) == (c.text, c.sign, c.ref, c.phase)]
        assert len(twin) == 1
    assert max(c.scale for c in both.concepts) <= max(c.scale for c in one.concepts) + 1e-9
    # a cap never fights another slider's goal
    p = K.plan({**settings(), "sliders": {"hips": 2, "butt": 2}, "max_concepts": 0}, T, "anime", "1girl, nude")
    assert not [c for c in p.concepts if c.slider == "hips" and c.role == "cap" and "huge ass" in c.text]
    # the nudity lock alone does not hold back detail sliders
    st = {**settings(), "sliders": {"freckles": 1}, "who": {"freckles": "Everyone"}, "max_concepts": 0}
    a = K.windows(K.plan(st, T, "anime", "1girl, bikini").concepts, st, T)
    b = [w for c, w in zip(K.plan(st, T, "anime", "1girl, nude").concepts,
                           K.windows(K.plan(st, T, "anime", "1girl, nude").concepts, st, T)) if c.slider == "freckles"]
    assert b and b[0][0] == a[0][0]
    # a short range never loses its shape sliders to a Pose slider
    st = {**settings(), "start": 0.1, "end": 0.25, "sliders": {"pose-energy": 1, "breasts": 1},
          "who": {"breasts": "Everyone"}, "max_concepts": 0}
    p = K.plan(st, T, "realistic", "1girl")
    assert all(hi > lo for lo, hi in K.windows(p.concepts, st, T))
    # nudity lock wording
    for text, on in (("no clothes", True), ("nude, wearing nothing", True), ("naked, no clothes", True),
                     ("half nude", False), ("nearly naked", False), ("nude-colored lipstick", False),
                     ("naked, stockings", False), ("nude, thighhighs, high heels", False)):
        assert K.nudity_lock(text) == on, text
    # malformed infotext items are skipped, weighted keep-apart words survive
    got = K.from_infotext("isolate=(red hair:1.2)|blue eyes, strength=abc, breasts=inf, penis=small", T)
    assert got["isolate"] == "(red hair:1.2), blue eyes" and got["strength"] == K.DEFAULTS["strength"]
    assert got["sliders"]["penis"] != 0
    # auto profile: "raw" and "real" only as words
    assert K.resolve_profile("Auto", "illustriousDrawingXL", T) == "anime"
    assert K.resolve_profile("Auto", "someModel_real_v2", T) == "realistic"


def test_stretch_marks_and_cellulite():
    for key in ("stretch-marks", "cellulite"):
        s = BY[key]
        assert s["phase"] == "detail" and s.get("face") is None and s["mask_words"] and "*" in s["wait_for"]
        assert K.slider_range(s) == (-2, 2)
        for profile in ("realistic", "anime", "stylized"):
            for v in (-2, -1, 1, 2):
                p = K.plan({**settings(), "sliders": {key: v}, "who": {key: "Everyone"}, "max_concepts": 0},
                           T, profile, "1girl")
                live = [c for c in p.concepts if c.scale > 0]
                if key == "cellulite" and profile == "anime":
                    assert not live and any("off in anime" in n for n in p.notes)
                else:
                    assert live, (key, profile, v)


def test_minus_side_only_takes_away():
    """-1 of a slider whose -1 names a small amount (sparse hair, lightly toned, half-erect) never
    pushes toward it: a hairless chest stays hairless, a soft body does not get toned."""
    for key in ("muscle", "abs", "v-line", "body-hair", "pubic-hair", "armpit-hair", "limb-hair", "erection"):
        for profile in ("realistic", "anime", "stylized"):
            p = K.plan({**settings(), "sliders": {key: -1}, "who": {key: "Everyone"}, "max_concepts": 0},
                       T, profile, "1girl, 1boy")
            live = [c for c in p.concepts if c.slider == key and c.scale > 0]
            if not K.profile_concepts(T, BY[key], profile):
                continue
            assert live and all(c.sign < 0 for c in live), (key, profile, [(c.role, c.text) for c in live])
    # Muscle no longer speaks of abs or fat: those are Abs and Weight
    for profile, table in BY["muscle"]["concepts"].items():
        for e in table.values():
            t = e if isinstance(e, str) else e["text"]
            assert "abs" not in t.split(", ") and "body fat" not in t and "soft body" not in t, (profile, t)


def test_pose_style_and_energy():
    s = BY["pose-style"]
    assert s["phase"] == "pose" and s["scope"] == "global" and s["keep_face"] and K.slider_range(s) == (-2, 2)
    for profile in ("realistic", "anime", "stylized"):
        for v in (-2, -1, 1, 2):
            p = K.plan({**settings(), "sliders": {"pose-style": v}, "max_concepts": 0}, T, profile, "1girl")
            assert [c for c in p.concepts if c.scale > 0], (profile, v)
    # Pose energy never names a standing pose: a sitting prompt stays sitting
    for table in BY["pose-energy"]["concepts"].values():
        for e in table.values():
            t = e if isinstance(e, str) else e["text"]
            assert "standing" not in t and "contrapposto" not in t, t
    p = K.plan({**settings(), "sliders": {"pose-energy": -1, "pose-style": 1}}, T, "anime", "1girl")
    assert any("clash" in n for n in p.notes)


def test_pose_style_by_scene_and_seed():
    s = BY["pose-style"]
    assert s["by_scene"] and s["mixed_scene"] == "off"
    w = K.plan({**settings(), "sliders": {"pose-style": 2}, "max_concepts": 0}, T, "anime", "1girl, solo")
    m = K.plan({**settings(), "sliders": {"pose-style": 2}, "max_concepts": 0}, T, "anime", "1boy, solo")
    a = K.plan({**settings(), "sliders": {"pose-style": 2}, "max_concepts": 0}, T, "anime", "solo")
    tw, tm = (next(c for c in p.concepts if c.role == "toward") for p in (w, m))
    assert tw.variants and tm.variants and set(tw.variants).isdisjoint(tm.variants)      # her pose, his pose
    assert not next(c for c in a.concepts if c.role == "toward").variants                  # no gender: one text
    # each image's seed picks one variant; the same seed always the same
    picks = {K.variant(tw, seed) for seed in range(len(tw.variants))}
    assert picks == set(tw.variants) and K.variant(tw, 7) == K.variant(tw, 7 + len(tw.variants))
    both = K.plan({**settings(), "sliders": {"pose-style": 2}}, T, "anime", "1girl, 1boy")
    assert not both.concepts and any("off" in n for n in both.notes)
    assert K.scene_key({"futa"}) == "women" and K.scene_key({"cuntboy"}) == "men" and K.scene_key(set()) == "any"


def test_genitals_shape_and_detail():
    assert BY["penis"]["label"] == "Penis size" and BY["vulva"]["label"] == "Labia"
    g, pf, sc = BY["penis-girth"], BY["vulva-puffy"], BY["scrotum"]
    assert g["phase"] == "shape" and g["fuse"] == "penis" and pf["phase"] == "detail" and pf["fuse"] == "vulva"
    assert sc["phase"] == "detail" and sc["region"] == "testicles"
    # size and girth are one description; -1 of each only takes away
    p = K.plan({**settings(), "sliders": {"penis": 1, "penis-girth": 2}, "who": {"penis": "Everyone",
                "penis-girth": "Everyone"}, "max_concepts": 0}, T, "realistic", "1boy, nude")
    fused = [c for c in p.concepts if c.slider.startswith("fuse:penis") and c.role == "toward"]
    assert fused and "large penis" in fused[0].text and "thick" in fused[0].text
    for key, prompt in (("penis-girth", "1boy"), ("vulva-puffy", "1girl"), ("scrotum", "1boy")):
        p = K.plan({**settings(), "sliders": {key: -1}, "who": {key: "Everyone"}, "max_concepts": 0}, T, "realistic", prompt)
        live = [c for c in p.concepts if c.slider == key and c.scale > 0]
        assert live and all(c.sign < 0 for c in live), key
    # Scrotum's earlier step names still paste
    assert K.from_infotext("scrotum=wrinkled", T)["sliders"]["scrotum"] == 1
    # scrotum waits for the testicles' shape
    st = {**settings(), "sliders": {"testicles": 1, "scrotum": 1}, "who": {"testicles": "Everyone", "scrotum": "Everyone"},
          "max_concepts": 0}
    p = K.plan(st, T, "realistic", "1boy")
    w = {c.slider: lo for c, (lo, _) in zip(p.concepts, K.windows(p.concepts, st, T))}
    assert w["scrotum"] > w["testicles"]


def test_no_fluid_in_shape_steps():
    """A shape or pose step names no fluid: it would be drawn in the composition steps."""
    fluid = re.compile(r"\b(precum|cum|semen|wet|sweat\w*|glistening|juice|lube|drool)\b", re.I)
    for s in T["sliders"]:
        if s.get("phase") not in ("shape", "pose"):
            continue
        for profile, entries in tables(s):
            for name, raw in entries.items():
                for e in K.entries(raw):
                    if e["direction"] == "toward":
                        for t in all_texts(e):
                            assert not fluid.search(t), (s["key"], profile, name, t)


def test_hair_and_marks_fit_max_concepts_without_taking_turns():
    """Two hair sliders (3 concepts each) at the default Max 6: their caps go before any concept
    has to take turns (stale, fading differences blurred both)."""
    st = {**settings(), "sliders": {"pubic-hair": 2, "armpit-hair": 2}, "who": {"pubic-hair": "Everyone",
          "armpit-hair": "Everyone"}, "max_concepts": 6, "_no_lock": True}
    p = K.plan(st, T, "realistic", "1girl, nude")
    assert len(p.concepts) <= 6 and not any("take turns" in n for n in p.notes)
    assert {c.slider for c in p.concepts if c.role == "toward" and c.scale > 0} == {"pubic-hair", "armpit-hair"}
    # a ref step's counter-push is its whole edit: never left out
    st = {**st, "sliders": {"pubic-hair": -1, "armpit-hair": -1, "limb-hair": -1, "body-hair": -1},
          "who": {k: "Everyone" for k in ("pubic-hair", "armpit-hair", "limb-hair", "body-hair")}, "max_concepts": 2}
    p = K.plan(st, T, "realistic", "1girl, nude")
    assert {c.slider for c in p.concepts if c.scale > 0} == {"pubic-hair", "armpit-hair", "limb-hair", "body-hair"}
    # marks dot the skin: five of them are not "piled up on one region"
    st = {**settings(), "sliders": {k: 1 for k in ("scars", "moles", "birthmarks", "freckles", "stretch-marks")},
          "who": {k: "Everyone" for k in ("scars", "moles", "birthmarks", "freckles", "stretch-marks")}}
    assert not any("on skin" in n for n in K.plan(st, T, "realistic", "1girl").notes)
    # Body hair's mask reads chest, stomach, navel: never "body", which is the whole figure
    assert "body" not in BY["body-hair"]["mask_words"]


def test_body_presets():
    import json
    import pathlib
    data = json.loads((pathlib.Path(__file__).resolve().parent.parent / "data" / "presets.json").read_text(encoding="utf-8"))
    scope = set(data["scope"])
    assert scope <= set(BY)
    # never Pose, Arousal, skin tone, the marks or grooming: those are the prompt's or the user's
    assert not scope & {"pose-energy", "pose-style", "camera-height", "head-turn", "erection", "vulva-wetness",
                        "nipple-state", "penis-wetness", "anus-wetness", "cum", "skin-tone", "nipple-color",
                        "flush", "sweat", "anatomy-fix"}
    age = re.compile(r"\b(loli|shota|child|kid|teen|young|youthful|petite|twink|minor|school|underage|girlish|boyish)\b", re.I)
    groups = {"women": 0, "men": 0}
    names = set()
    for p in data["presets"]:
        assert p["name"] not in names and not age.search(p["name"]), p["name"]
        names.add(p["name"])
        groups[p["group"]] += 1
        assert 3 <= len(p["variants"]) <= 4, p["name"]
        assert len({v["name"] for v in p["variants"]}) == len(p["variants"]), p["name"]
        for v in p["variants"]:
            assert not age.search(v["name"]), (p["name"], v["name"])
            for over in [{}] + list((p.get("profiles") or {}).values()):        # few sliders: cost
                assert 1 <= len({**v["sliders"], **over}) <= 9, (p["name"], v["name"])
            for key, value in v["sliders"].items():
                lo, hi = K.slider_range(BY[key])
                assert key in scope and lo <= value <= hi and value != 0, (p["name"], v["name"], key, value)
            if p["group"] == "women":
                assert not {"penis", "penis-girth", "testicles", "scrotum", "foreskin", "pectorals"} & set(v["sliders"])
            else:
                assert not {"breasts", "breast-shape", "vulva", "vulva-puffy", "vulva-texture"} & set(v["sliders"])
    assert groups["women"] >= 12 and groups["men"] >= 12
    assert "♀ Porn star" in names and "♂ Porn star" in names
    # on every profile: no variant pairs clashing sliders or trips a warning; off sliders stay 0
    for profile in ("realistic", "anime", "stylized"):
        for p in data["presets"]:
            prompt = "1girl, nude" if p["group"] == "women" else "1boy, nude"
            for v in p["variants"]:
                vals = {**v["sliders"], **((p.get("profiles") or {}).get(profile) or {})}
                vals = {k: x for k, x in vals.items() if x and K.profile_concepts(T, BY[k], profile)}
                plan = K.plan({**settings(), "sliders": vals, "who": {k: "Everyone" for k in vals},
                               "max_concepts": 0, "_no_lock": True}, T, profile, prompt)
                assert not [n for n in plan.notes if n.startswith("⚠")], (profile, p["name"], v["name"], plan.notes)
    for p in data["presets"]:
        for profile, over in (p.get("profiles") or {}).items():
            assert profile in T["profiles"] and set(over) <= scope, p["name"]


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
