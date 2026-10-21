"""The engine through a WebUI's own sampling code: its condition classes, its
batch runner and its CFG / post-CFG order. A small fake model stands in for the
UNet (the hook does not depend on what the model computes).

    cd <webui root>; <python with torch> <this extension>/tests/test_in_host.py

Works from a Forge, reForge or Forge Classic (Neo) folder. Only the sampling
modules are imported, the WebUI is not booted.
"""
import pathlib
import sys

WEBUI = pathlib.Path.cwd()
EXT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(EXT))
import torch  # noqa: E402

from lib_lewd_anatomy import concepts as K, engine as E, host  # noqa: E402
from lib_lewd_anatomy.sega import Sega  # noqa: E402

torch.manual_seed(0)
KIND = "reforge" if (WEBUI / "ldm_patched").is_dir() else "neo" if (WEBUI / "modules_forge/packages/k_diffusion").is_dir() else "forge"
fails = []
# run the WebUI's memory management on the CPU (no GPU needed), as its launcher flags do
sys.argv = sys.argv[:1] + (["--cpu"] if KIND == "neo" else ["--always-cpu"])


def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


if KIND == "reforge":
    import ldm_patched.modules.options
    ldm_patched.modules.options.enable_args_parsing()     # as reForge's launcher does
    # the samplers read a few options from modules.shared, which needs a booted WebUI:
    # stand in for it (only epsilon scaling is read on this path, and it stays off)
    import types
    import modules
    _shared = types.ModuleType("modules.shared")
    _shared.opts = types.SimpleNamespace()
    sys.modules["modules.shared"] = modules.shared = _shared
    from ldm_patched.modules import conds as C, samplers as S
    CrossAttn, Regular = C.CONDCrossAttn, C.CONDRegular

    def run_sampling(model, x, sigma, uncond, cond, cfg, options):
        return S.sampling_function(model, x, sigma, uncond, cond, cfg, options)
else:
    from backend.sampling import condition as C, sampling_function as S
    from modules import shared as _shared
    if getattr(_shared, "opts", None) is None:        # Neo reads shared.opts.batch_cond_uncond; set at boot
        import types
        _shared.opts = types.SimpleNamespace(batch_cond_uncond=True)
    CrossAttn, Regular = C.ConditionCrossAttn, C.Condition

    def run_sampling(model, x, sigma, uncond, cond, cfg, options):
        return S.sampling_function_inner(model, x, sigma, uncond, cond, cfg, options)

print(f"\n=== {KIND} (host.name() = {host.name()})")
ok("host detected", host.name() == KIND)


class FakeModel:
    """apply_model(x, t, c_crossattn=..., y=...) -> a denoised prediction that depends on
    x, sigma, the cross-attention context and the pooled vector, per batch row."""

    def __init__(self):
        self.calls = []
        self.current_patcher = None

    def memory_required(self, shape, **kwargs):
        return 0

    def apply_model(self, x, t, c_crossattn=None, y=None, **kwargs):
        self.calls.append(x.shape[0])
        ctx = c_crossattn.mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        pooled = y.mean(dim=1).reshape(-1, 1, 1, 1) if y is not None else 0
        return torch.tanh(x) * 0.7 + ctx + 0.1 * pooled + 0.01 * t.reshape(-1, 1, 1, 1)


def entry(cross, vec):
    return {"cross_attn": cross, "pooled_output": vec,
            "model_conds": {"c_crossattn": CrossAttn(cross), "y": Regular(vec)}}


B, D = 2, 16
x = torch.randn(B, 4, 16, 16)
sigma = torch.full((B,), 3.0)
cond = [entry(torch.randn(B, 154, D), torch.randn(B, 8))]
uncond = [entry(torch.randn(B, 77, D), torch.randn(B, 8))]
concept_conds = [{"crossattn": torch.randn(1, 77, D) + i, "vector": torch.randn(1, 8)} for i in range(3)]
plan = K.Plan("realistic", [K.Concept("t", f"c{i}", s, 4.0) for i, s in enumerate((1.0, -1.0, -1.0))])
settings = {**K.DEFAULTS, "start": 0.0, "end": 1.0, "orthogonal": False}   # the SEGA math alone; Perp-Neg has its own case

# 1. per_call = 1 (default): one extra call of batch B per concept, never more rows
#    than the main CFG call; per_call = 3 puts the three concepts in one call
model = FakeModel()
eng = E.Engine(plan, concept_conds, settings, log=print)
options = {"sampler_post_cfg_function": [eng.post_cfg]}
plain = run_sampling(FakeModel(), x, sigma, uncond, cond, 6.0, {})
model.calls.clear()
out = run_sampling(model, x, sigma, uncond, cond, 6.0, options)
ok("default: one call of batch 2 per concept, each <= the CFG call",
   model.calls == [4, 2, 2, 2], str(model.calls))
wide = FakeModel()
eng_wide = E.Engine(plan, concept_conds, {**settings, "per_call": 3}, log=print)
out_wide = run_sampling(wide, x, sigma, uncond, cond, 6.0, {"sampler_post_cfg_function": [eng_wide.post_cfg]})
ok("per_call 3: one call of batch 3 x 2, same result", wide.calls == [4, 6] and torch.allclose(out, out_wide, atol=1e-6),
   str(wide.calls))

# 2. same result as predicting each concept against the user's prompt, not uncond
ref_model = FakeModel()
ref_x0 = [ref_model.apply_model(x, sigma, c_crossattn=c["crossattn"].expand(B, -1, -1), y=c["vector"].expand(B, -1))
          for c in concept_conds]
c_x0 = ref_model.apply_model(x, sigma, c_crossattn=cond[0]["cross_attn"], y=cond[0]["pooled_output"])
ref = Sega(E.MOMENTUM_SCALE, E.MOMENTUM_BETA).apply(x, sigma, plain, c_x0, ref_x0, [4.0, -4.0, -4.0], [K.DEFAULTS["locality"]] * 3, [True] * 3)
ok("matches per-concept reference", torch.allclose(out, ref, atol=1e-5), f"max err {(out - ref).abs().max():.2e}")
ok("changes the prediction", not torch.allclose(out, plain))

# 3. chaining: a CFG function set by another extension still runs, SEGA adds on top
def other_cfg(args):
    return (args["cond"] - args["uncond"]) * 2.0 + args["uncond"]     # noise space


eng2 = E.Engine(plan, concept_conds, settings, log=print)
chained = run_sampling(FakeModel(), x, sigma, uncond, cond, 6.0,
                       {"sampler_cfg_function": other_cfg, "sampler_post_cfg_function": [eng2.post_cfg]})
other_only = run_sampling(FakeModel(), x, sigma, uncond, cond, 6.0, {"sampler_cfg_function": other_cfg})
delta_plain = out - plain
delta_chain = chained - other_only
ok("chains with an existing CFG function", torch.allclose(delta_plain, delta_chain, atol=1e-5)
   and not torch.allclose(chained, other_only))

# 4. post-CFG functions registered before keep running (order: theirs, then SEGA)
seen = []


def other_post(args):
    seen.append("other")
    return args["denoised"] + 0.5


eng3 = E.Engine(plan, concept_conds, settings, log=print)
both = run_sampling(FakeModel(), x, sigma, uncond, cond, 6.0, {"sampler_post_cfg_function": [other_post, eng3.post_cfg]})
ok("earlier post-CFG function still applied", seen == ["other"] and torch.allclose(both, out + 0.5, atol=1e-5))

# 5. outside the step range: identical. CFG 1 has no negative pass, but the prompt
#    prediction is still there, so the edit runs and must not error.
eng4 = E.Engine(plan, concept_conds, {**settings, "start": 0.5, "end": 0.9}, log=print)
eng4.on_step(0, 20)
ok("outside step range = identical", torch.equal(run_sampling(FakeModel(), x, sigma, uncond, cond, 6.0,
                                                              {"sampler_post_cfg_function": [eng4.post_cfg]}), plain))
cfg1_plain = run_sampling(FakeModel(), x, sigma, uncond, cond, 1.0, {})
eng5 = E.Engine(plan, concept_conds, settings, log=print)
cfg1 = run_sampling(FakeModel(), x, sigma, uncond, cond, 1.0, {"sampler_post_cfg_function": [eng5.post_cfg]})
ok("CFG 1 edits from the prompt prediction and does not error",
   (not torch.equal(cfg1, cfg1_plain)) and (not eng5.off) and eng5.calls >= 1)

# 6. AND prompts (two weighted cond entries) and an inpainting concat: the template keeps c_concat
cond_and = [dict(entry(torch.randn(B, 77, D), torch.randn(B, 8)), strength=0.6),
            dict(entry(torch.randn(B, 77, D), torch.randn(B, 8)), strength=0.4)]
for c in cond_and + uncond:
    c["model_conds"] = dict(c["model_conds"], c_concat=Regular(torch.randn(B, 5, 16, 16)))


class ConcatModel(FakeModel):
    def apply_model(self, x, t, c_concat=None, **kw):
        assert c_concat is not None and c_concat.shape[0] == x.shape[0]
        return super().apply_model(x, t, **kw)


eng6 = E.Engine(plan, concept_conds, settings, log=print)
run_sampling(ConcatModel(), x, sigma, uncond, cond_and, 6.0, {"sampler_post_cfg_function": [eng6.post_cfg]})
ok("AND prompt + c_concat repeated to the concept batch", not eng6.off and eng6.passes == 3, f"{eng6.passes} passes")

# 7. a different prompt per image (each image's concepts built on its own prompt) and a
#    contrastive pair (the away concept measured from the toward one, not from the prompt)
per_row = [[{"crossattn": torch.randn(1, 77, D) + 3 * i + r, "vector": torch.randn(1, 8)} for r in range(B)]
           for i in range(2)]
pair = K.Plan("realistic", [K.Concept("t", "small", 1.0, 2.0), K.Concept("t", "large", -1.0, 3.0, ref="small")])
eng7 = E.Engine(pair, per_row, settings, log=print)
m7 = FakeModel()
out7 = run_sampling(m7, x, sigma, uncond, cond, 6.0, {"sampler_post_cfg_function": [eng7.post_cfg]})
r_model = FakeModel()
rows = [torch.cat([r_model.apply_model(x[b:b + 1], sigma[b:b + 1], c_crossattn=per_row[i][b]["crossattn"],
                                       y=per_row[i][b]["vector"]) for b in range(B)]) for i in range(2)]
c_x0 = r_model.apply_model(x, sigma, c_crossattn=cond[0]["cross_attn"], y=cond[0]["pooled_output"])
ref7 = Sega(E.MOMENTUM_SCALE, E.MOMENTUM_BETA).apply(x, sigma, plain, c_x0, rows, [2.0, -3.0], [K.DEFAULTS["locality"]] * 2,
                                                     [True] * 2, refs=[None, 0])
ok("per-image prompts + contrastive pair match the reference", torch.allclose(out7, ref7, atol=1e-5) and not eng7.off,
   f"max err {(out7 - ref7).abs().max():.2e}")

# 8. Rotate + Auto batch + Staggered together, over several steps, through the fork's runner
six = K.Plan("realistic", [K.Concept("t", f"r{i}", 1.0 if i % 2 == 0 else -1.0, 2.0, None,
                                     f"r{i - 1}" if i % 2 else None, "shape" if i < 4 else "detail")
                           for i in range(6)])
six_conds = [{"crossattn": torch.randn(1, 77, D) + i, "vector": torch.randn(1, 8)} for i in range(6)]
eng8 = E.Engine(six, six_conds, {**settings, "rotate": 2, "auto_batch": True, "staggered": True}, log=print)
m8 = FakeModel()
fine = True
for step in range(10):
    eng8.on_step(step, 10)
    out8 = run_sampling(m8, x, sigma, uncond, cond, 6.0, {"sampler_post_cfg_function": [eng8.post_cfg]})
    fine = fine and bool(torch.isfinite(out8).all())
extra = [b for b in m8.calls if b != 4]          # 4 = the main cond + uncond call
ok("rotate + auto batch + staggered run clean", fine and not eng8.off and eng8.passes < 6 * 10,
   f"{eng8.passes} concept passes in {eng8.calls} calls over 10 steps (no rotation: 60)")

# 9. Concept mask through a real (tiny, random) UNet of this fork: the attention
#    replacement gives the same output as the fork's own attention, maps are recorded
#    for the rows that have concept tokens, and outside the mask nothing changes.
from lib_lewd_anatomy import attention as A  # noqa: E402

UCFG = dict(in_channels=4, model_channels=32, out_channels=4, num_res_blocks=1, channel_mult=(1, 2),
            num_head_channels=16, use_spatial_transformer=True, transformer_depth=[1, 1],
            transformer_depth_output=[1, 1, 1, 1], transformer_depth_middle=1, context_dim=D,
            use_linear_in_transformer=True)
if KIND == "reforge":
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel
    import ldm_patched.modules.ops as ops
    unet = UNetModel(image_size=16, operations=ops.disable_weight_init, **UCFG)
else:
    from backend.nn.unet import IntegratedUNet2DConditionModel
    unet = IntegratedUNet2DConditionModel(**UCFG)
for name_, prm in unet.named_parameters():
    torch.nn.init.normal_(prm, std=1.0 if ("to_q" in name_ or "to_k" in name_) else 0.3 if "attn" in name_ else 0.08)
unet.eval()


class TinyModel:
    """The WebUI's model wrapper, around a real UNet of this fork."""

    def __init__(self, dm):
        self.diffusion_model = dm
        self.current_patcher = None

    def memory_required(self, shape, **kwargs):
        return 0

    def extra_conds_shapes(self, **kwargs):
        return {}

    def model_dtype(self):
        return torch.float32

    def apply_model(self, x, t, c_crossattn=None, transformer_options={}, y=None, **kwargs):
        out = self.diffusion_model(x, timesteps=t, context=c_crossattn, transformer_options=transformer_options)
        return x - 0.5 * out


tiny = TinyModel(unet)
blocks = A.attention_blocks(unet)
ok("attention blocks found in the fork's UNet", len(blocks) >= 3, str(blocks))
A.MAX_SHARE = 0.25                      # the tiny UNet's coarse layer is 1/4 of the latent
xs = torch.randn(B, 4, 16, 16)
ss = torch.full((B,), 2.0)
ctx = torch.randn(B, 12, D)
c_entries = [{"cross_attn": ctx, "model_conds": {"c_crossattn": CrossAttn(ctx)}}]
with torch.no_grad():
    plain_pred = host.predict(tiny, c_entries, xs, ss, {})
    rec = A.Recorder([([3, 4], 12), None], (16, 16))
    rec_pred = host.predict(tiny, c_entries, xs, ss, A.with_recorder({}, blocks, rec))
ok("attention replacement == the fork's own attention", torch.allclose(plain_pred, rec_pred, atol=1e-4),
   f"max err {(plain_pred - rec_pred).abs().max():.2e}")
maps = rec.maps()
ok("map recorded for the row with tokens only", maps is not None and maps.shape == (2, 16, 16)
   and bool(maps[0].sum() > 0) and bool(maps[1].abs().sum() == 0), "" if maps is None else str(tuple(maps.shape)))

# a whole edit with the mask on: zero change wherever the mask is zero
mask_conds = [{"crossattn": torch.randn(1, 12, D), "vector": None, "tokens": ([5, 6], 12)}]
mplan = K.Plan("realistic", [K.Concept("t", "m", 1.0, 4.0)])
eng9 = E.Engine(mplan, mask_conds, {**settings, "attn_mask": True, "attn_area": 0.2, "rotate": 0}, log=print)
base_c = [{"cross_attn": ctx, "model_conds": {"c_crossattn": CrossAttn(ctx)}}]
with torch.no_grad():
    cond_pred = host.predict(tiny, base_c, xs, ss, {})
    args9 = {"denoised": cond_pred.clone(), "cond_denoised": cond_pred, "uncond_denoised": cond_pred * 0.9,
             "input": xs, "sigma": ss, "cond": base_c, "uncond": [], "model": tiny, "model_options": {}}
    out9 = eng9.post_cfg(args9)
mask9 = eng9._mask(0, out9.device)
changed = (out9 - cond_pred).abs().amax(dim=1, keepdim=True) > 0
ok("masked edit: nothing changes outside the concept mask", mask9 is not None and not eng9.off
   and bool(torch.all(~changed | (mask9 > 0))) and bool(changed.any()),
   "" if mask9 is None else f"mask covers {(mask9 > 0).float().mean():.0%}, changed {changed.float().mean():.0%}")
eng9b = E.Engine(mplan, mask_conds, {**settings, "attn_mask": False}, log=print)
with torch.no_grad():
    out9b = eng9b.post_cfg(dict(args9, denoised=cond_pred.clone()))
ok("mask off: the edit is not limited", ((out9b - cond_pred).abs().amax(1) > 0).float().mean() >
   changed.float().mean())

# 10. concept token positions with this fork's tokenizer files (SDXL CLIP-L vocabulary)
tok_dir = WEBUI / "backend/huggingface/stabilityai/stable-diffusion-xl-base-1.0/tokenizer"
if tok_dir.is_dir():
    from transformers import CLIPTokenizer
    hf = CLIPTokenizer.from_pretrained(str(tok_dir))

    class Engine77:
        """Minimal 75+2 token chunker (what tokenize_line produces for plain text)."""
        id_start, id_end = hf.bos_token_id, hf.eos_token_id

        def tokenize(self, texts):
            return hf(texts, truncation=False, add_special_tokens=False)["input_ids"]

        def tokenize_line(self, line):
            ids = self.tokenize([line])[0]
            chunks = []
            for at in range(0, max(len(ids), 1), 75):
                part = ids[at:at + 75]
                chunks.append(type("C", (), {"tokens": [self.id_start] + part + [self.id_end] * (76 - len(part))})())
            return chunks, len(ids)

    eng_t = Engine77()
    prompt = "1girl, completely nude, standing in a sunlit bedroom, detailed skin, " * 6
    full = K.edit_prompt(prompt, "small natural breasts")
    pos, length = host.concept_token_positions(eng_t, full, "small natural breasts")
    flat = [t for c in eng_t.tokenize_line(full)[0] for t in c.tokens]
    words = hf.convert_ids_to_tokens([flat[i] for i in pos])
    ok("concept tokens found after a long prompt (second chunk)", length == 154 and pos[0] > 77
       and "".join(words).replace("</w>", " ").strip() == "small natural breasts", f"{pos} {words}")
else:
    print("skip  tokenizer files not in this fork")

# 11. Anti-burn with every slider at 0, through the fork's runner
from lib_lewd_anatomy.sega import cfg_rescale  # noqa: E402

eng11 = E.Engine(K.Plan("realistic"), [], {**settings, "rescale": 0.6, "_cfg": 6.0}, log=print)
out11 = run_sampling(FakeModel(), x, sigma, uncond, cond, 6.0, {"sampler_post_cfg_function": [eng11.post_cfg]})
rm = FakeModel()
c11 = rm.apply_model(x, sigma, c_crossattn=cond[0]["cross_attn"], y=cond[0]["pooled_output"])
ref11 = cfg_rescale(x, sigma, plain, c11, 0.6)
ok("anti-burn (no slider) matches the reference", torch.allclose(out11, ref11, atol=1e-4)
   and not eng11.off, f"max err {(out11 - ref11).abs().max():.2e}")

# 12. The concept mask through a real tiny UNet
x32 = torch.randn(1, 4, 32, 32)
s32 = torch.full((1,), 2.0)
c32 = torch.randn(1, 12, D)
u32 = torch.randn(1, 12, D)
ce = [{"cross_attn": c32, "model_conds": {"c_crossattn": CrossAttn(c32)}}]
ue = [{"cross_attn": u32, "model_conds": {"c_crossattn": CrossAttn(u32)}}]
fconds = [{"crossattn": torch.randn(1, 12, D), "vector": None, "tokens": ([5, 6], 12)}]
fplan = K.Plan("realistic", [K.Concept("t", "m", 1.0, 4.0)])
fset = {**settings, "attn_mask": True, "range_by": "Steps", "orthogonal": True}
eng12 = E.Engine(fplan, fconds, fset, log=print)
eng12.on_step(1, 20)
with torch.no_grad():
    plain12 = run_sampling(TinyModel(unet), x32, s32, ue, ce, 6.0, {})
    out12 = run_sampling(TinyModel(unet), x32, s32, ue, ce, 6.0, {"sampler_post_cfg_function": [eng12.post_cfg]})
ok("concept mask through a real UNet", not eng12.off and bool(torch.isfinite(out12).all())
   and not torch.allclose(out12, plain12) and eng12.attn[0] is not None,
   "" if eng12.attn[0] is None else f"maps recorded at {tuple(eng12.attn[0].shape[-2:])}")

# 13. Protect face through a real UNet: with the concept mask off (edits everywhere), nothing
#     changes where the face probe marks the face
pf_conds = [{"crossattn": torch.randn(1, 12, D), "vector": None, "tokens": None}]
pf_plan = K.Plan("realistic", [K.Concept("t", "m", 1.0, 4.0)])
eng13 = E.Engine(pf_plan, pf_conds, {**settings, "attn_mask": False, "protect_face": True, "range_by": "Steps",
                                     "orthogonal": False}, log=print)
eng13.face_conds = [{"crossattn": torch.randn(1, 12, D), "vector": None, "tokens": ([4, 5], 12)}]
eng13.on_step(3, 20)
with torch.no_grad():
    plain13 = run_sampling(TinyModel(unet), x32, s32, ue, ce, 6.0, {})
    out13 = run_sampling(TinyModel(unet), x32, s32, ue, ce, 6.0, {"sampler_post_cfg_function": [eng13.post_cfg]})
face13 = eng13.face_mask
changed13 = (out13 - plain13).abs().amax(dim=1, keepdim=True) > 1e-6
ok("protect face: no change where the face mask is full", face13 is not None and not eng13.off
   and bool(changed13.any()) and not bool((changed13 & (face13 >= 1.0)).any()),
   "" if face13 is None else f"face {(face13 >= 1).float().mean():.0%} of the image, changed {changed13.float().mean():.0%}")
ok("protect face: a flat attention map is capped, not the whole image", face13 is not None
   and float((face13 >= 1).float().mean()) < 0.75)

# 14. A Pose slider is the whole picture: the same setup, but its edit reaches the face
pose_plan = K.Plan("realistic", [K.Concept("head-turn", "m", 1.0, 4.0, phase="pose")])
eng14 = E.Engine(pose_plan, pf_conds, {**settings, "attn_mask": True, "protect_face": True, "range_by": "Steps",
                                       "orthogonal": False}, log=print)
eng14.face_conds = eng13.face_conds
eng14.face_mask, eng14.face_age = face13, 0          # the same face as case 13
eng14.on_step(3, 20)
with torch.no_grad():
    out14 = run_sampling(TinyModel(unet), x32, s32, ue, ce, 6.0, {"sampler_post_cfg_function": [eng14.post_cfg]})
changed14 = (out14 - plain13).abs().amax(dim=1, keepdim=True) > 1e-6
ok("head turn: not masked, reaches the face", eng14.whole == [True] and not eng14.off and face13 is not None
   and bool((changed14 & (face13 >= 1.0)).any()))

# 14b. Pose energy under Protect face (default): whole picture, but not on the face
keep_plan = K.Plan("realistic", [K.Concept("pose-energy", "m", 1.0, 4.0, phase="pose")])
eng14b = E.Engine(keep_plan, pf_conds, {**settings, "attn_mask": True, "protect_face": True, "range_by": "Steps",
                                        "orthogonal": False}, log=print)
eng14b.face_conds = eng13.face_conds
eng14b.face_mask, eng14b.face_age = face13, 0
eng14b.windows = [(0.0, 1.0)]                       # keep the mask fixed for the comparison
eng14b.face_every = 99
eng14b.on_step(3, 20)
with torch.no_grad():
    out14b = run_sampling(TinyModel(unet), x32, s32, ue, ce, 6.0, {"sampler_post_cfg_function": [eng14b.post_cfg]})
changed14b = (out14b - plain13).abs().amax(dim=1, keepdim=True) > 1e-6
ok("pose energy + keep face: changes the picture, not the face", eng14b.keepface == [True] and not eng14b.off
   and bool(changed14b.any()) and eng14b.face_mask is not None
   and not bool((changed14b & (eng14b.face_mask >= 1.0)).any()))

# 14c. Facial hair: on the face, but never on its core (eyes and lips)
fh_plan = K.Plan("realistic", [K.Concept("facial-hair", "m", 1.0, 4.0, phase="detail")])
eng14c = E.Engine(fh_plan, pf_conds, {**settings, "attn_mask": False, "protect_face": True, "range_by": "Steps",
                                      "orthogonal": False}, log=print)
eng14c.face_conds = eng13.face_conds
eng14c.windows = [(0.0, 1.0)]
eng14c.on_step(3, 20)
with torch.no_grad():
    out14c = run_sampling(TinyModel(unet), x32, s32, ue, ce, 6.0, {"sampler_post_cfg_function": [eng14c.post_cfg]})
changed14c = (out14c - plain13).abs().amax(dim=1, keepdim=True) > 1e-6
core14c, face14c = eng14c.face_core, eng14c.face_mask
ok("facial hair: reaches the face, not its core", eng14c.face_policy == ["features"] and core14c is not None
   and bool((changed14c & (face14c >= 1.0) & (core14c < 0.5)).any())
   and not bool((changed14c & (core14c >= 1.0)).any())
   and float((core14c >= 1).float().mean()) < float((face14c >= 1).float().mean()),
   "" if core14c is None else f"core {(core14c >= 1).float().mean():.0%}, face {(face14c >= 1).float().mean():.0%}")

print(f"\n{KIND}: {'all passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
