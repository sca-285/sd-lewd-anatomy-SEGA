# Lewd Anatomy · technical details

The short guide is the [README](../README.md). This page keeps the full reference: how each part works, every
setting, the concept tables and the tests.

*(formerly Body Sliders, briefly Physique)*

Sliders for body proportions of adult characters (breasts, penis, vulva, masculine ↔ feminine,
muscle, V-line, body fat, body hair) for **Forge**, **reForge** and **Forge Classic (Neo)**.

Lewd Anatomy corrects the exaggerated anatomy many SDXL checkpoints produce (anime models learned
from hentai, realistic merges inherit some of it, high CFG amplifies it) **without touching your
prompt and without LoRAs**. Your prompt and negative prompt are never edited. Nothing is patched
into the model weights. The only mechanism is Semantic Guidance inside the denoising loop.

All bodies are adult. No slider, preset or internal concept addresses age.

## How SEGA works

At every sampling step the model already predicts the noise twice: once for your prompt and once
for the negative prompt (unconditional). SEGA ([Brack et al., NeurIPS 2023](https://arxiv.org/abs/2301.12247))
adds one more prediction per *concept*, for example "huge breasts". The difference between that
prediction and the unconditional one tells, latent value by latent value, where and how the image
would change if the concept were in the prompt. SEGA keeps only the strongest few percent of that
difference (the **Locality** λ, 0.90 = the top 10 %) and adds it with a scale: positive to move
**toward** the concept, negative to push **away** from it. A momentum term carries the edit
smoothly across steps.

Because the mask keeps only what the concept would really change, pushing away from "huge breasts"
on an image that would not have them changes almost nothing. That is what makes it safe to leave on:
it only acts where the checkpoint is drifting toward the thing you do not want. Lewd Anatomy computes
the concepts only between the **Step range** start and end, because body shape is decided early, and
by default one concept per extra model call, so no call is bigger than half a normal CFG step.

The concept is not encoded as a bare phrase and it is not compared to the empty prompt. A phrase like
"medium natural breasts" on its own is a stock photo, and for these checkpoints that photo is usually
clothed — steering toward it (or away from "pubic hair" / "penis" from the unconditional prediction)
is what was painting shirts and underwear onto the character. Each concept is encoded **after your
prompt**, and the edit is the difference between that prediction and your prompt's own prediction.
Clothes, pose and background are in both, so they cancel. Only the anatomy moves. Phrases that named
a whole person ("flat-chested woman", "masculine man", "feminine woman") were rewritten to the body
trait alone, because the person-name drags a default outfit with it.

## Install

Extensions → Install from URL, or copy the `sd-webui-lewd-anatomy-sega` folder into `extensions/`, then
restart the WebUI. The **Lewd Anatomy · SEGA Semantic Guidance** accordion appears in txt2img and img2img.

Updating from **Body Sliders** or **Physique**: delete the old `sd-webui-body-sliders` /
`sd-webui-physique-sega` folder first, otherwise both run. Old images still paste back.

## Use

1. Open **Lewd Anatomy · SEGA Semantic Guidance** and tick it. The settings are always visible at the
   top; the 56 sliders are in eight tabs below them, in the order the layers run (see *Layer
   order*), and inside each tab shape sliders come before detail sliders: **Pose** (pose energy,
   pose style, camera height, head turn), **Body** (masculine ↔ feminine, shoulders, waist, hips, thighs, butt,
   anatomy fix, nipples / areolae, nipple melanin, anus), **Muscle** (muscle definition, weight,
   belly, lats, biceps, abs, V-line, back muscles), **Female** (breasts, breast shape, labia, vulva
   puffiness, vulva texture), **Male** (penis size, penis girth, testicles, pectorals, penis veins, scrotum, foreskin), **Skin**
   (skin tone, skin realism, skin flush, sweat, cellulite), **Arousal** (erection, nipple erection, vulva /
   penis / anus wetness, cum), **Hair & marks** (facial, body, armpit, pubic, arm & leg hair; scars,
   stretch marks, moles, birthmarks, freckles, wrinkles). The **Guide** button next to *Profile* opens a short how-to; **Reset sliders** sits under it.
2. Every slider has five steps: **−2 · −1 · 0 · +1 · +2**. 0 is off and computes nothing. Negative
   makes that trait smaller than what the prompt would give, positive makes it larger; where −1
   names a small amount (sparse hair, lightly toned, half-erect) it is only the point the push is
   measured from, so the minus side only takes away (a hairless chest stays hairless). The names of
   the steps are under each slider. Don't also write that size into the prompt: the prompt sets the
   scene, the slider sets the size. If both disagree, the result is unpredictable.
3. The status line lists the moved sliders, the number of edits, the extra UNet calls per step and
   the expected slowdown.

| Control | Meaning |
|---|---|
| Profile | **Realistic** (natural language, photo checkpoints), **Anime** (Danbooru tags: Pony, Illustrious, NoobAI, Animagine), **Stylized** (plain anatomy words without photo terms, for 2.5D / 3D / semi-real / cartoon checkpoints). **Auto** picks from the checkpoint name: `realis`, `photo`, `real` → Realistic; then `2.5d`, `semi`, `stylized`, `3d`, `toon`, `cartoon`, `pixar`, `render` → Stylized; `pony`, `illustrious`, `noob`, `animagine` → Anime (checked before the Stylized words); `anime`, `hentai` → Anime; otherwise Realistic. Old infotexts with `profile=pony` paste as Anime. A cartoon checkpoint built on Pony, Illustrious or NoobAI is prompted with tags, so the base-model names are checked before `toon` / `cartoon`. |
| Control mode | Who wins when prompt and slider disagree (as ControlNet's modes). **Balanced** (default): Strength as set; each edit loses its part along the prompt's own guidance (Perp-Neg, CFG > 1). **Prompt**: the same, every edit × 0.7: the prompt has the last word. **Sliders**: every edit × 1.3 and no Perp-Neg: a slider may overrule the prompt (prompt *large breasts*, Breasts −2). |
| **Options** | |
| · Protect face | On by default. Covers Pose energy, Pose style and Camera height as well (not Head turn). One probe pass ("<prompt>, face, eyes, eyebrows, nose, lips", refreshed every 4 steps) marks the face from where those words attend; every slider edit is zeroed there (soft edge, grown to take in jaw, hairline and ears, capped at 40 % of the image). Costs about a quarter of a concept pass per step. |
| · Concept mask | On by default. Each edit only where its own concept words attend in the UNet (keeps face, background, the other person). |
| · Allow exaggerated | Off: on the + side a slider also pushes away from its cap ("gigantic", "oversized"). |
| · Hires fix | Off: sliders do not run in the hires pass (the shape is set). Anti-burn and Keep apart run in it anyway. |
| Strength | Scales every edit. |
| Locality (λ) | Fixed at 0.90, not on the panel (the concept mask places every edit; Mask area is the knob). X/Y/Z can still sweep it; an infotext's `locality=` is not pasted back. |
| Range start / end | Where the slider edits run, in noise level (log-sigma: 0 = the schedule's noisiest, 1 = its cleanest; the same on every sampler and scheduler). Default 0.10 – 0.60, about where 15 % – 70 % of Karras steps were; each edit fades out over the last 30 % of its window. Shape sliders keep to the first 80 % of the range; detail sliders (hair, abs, V-line, back muscles, vulva, nipples, foreskin, anus, textures) start at 30 % and run on to 1.15 × the range: strands and texture form late. Small parts of the frame (vulva, anus, nipples, genital textures) also have their own small mask (0.1 of the image): early in the steps their words attend over the whole picture, and an edit there re-coloured hair and moved hands. |
| **Advanced** (always shown, below the main settings) | |
| · Baseline | **Prompt** (default): each concept is appended to your prompt and the edit is measured against your prompt's own prediction, so clothes, pose and background cancel; works at CFG 1. **Negative**: classic SEGA from the paper, the bare concept phrase against the negative / unconditional prediction; a stronger push, but a bare phrase can drag in a default outfit, and it does nothing at CFG 1. |
| · Mask area | 0.05 – 0.6, default 0.2: the share of the image one concept may touch, before a soft edge. |
| · Max concepts | 1 – 40, default 6. Above it nothing is dropped: the concepts take turns, and a turn's kept edit fades (× 0.6 per step it is reused). With many sliders on one body part, set it at least to the concept count in the status line: VRAM does not grow with it, time does. |
| · Anti-burn | CFG rescale φ. 0 = off; 0.5 – 0.7 lets CFG 8 – 12 run without burnt colours. Works with every slider at 0. |
| · Keep apart (couples) | Words that belong to one person, kept inside their own phrase (see *Keep apart* below). Empty = off. |
| Who | Everyone / Women / Men, for the toward part of a slider (see below). |
| ↺ / Reset sliders | Resets one slider / every slider to 0. |

### What a step computes

Each moved slider computes **at most three concepts**, the same way for every slider:

| | Concept | Scale |
|---|---|---|
| toward | the entry of the step you chose (for example −2 `flat chest`), measured from your prompt | `toward` × gain |
| away | the first step on the other side (−1 for a + step, +1 for a − step), where the checkpoint tends to drift, measured **from the toward concept** (contrastive) | `opposite` × gain |
| away (only + steps, Allow exaggerated off) | the slider's cap, also measured from the toward concept | `cap` × gain |

When **Who** blocks the toward push (a scene with both women and men), the toward concept is still
computed at scale 0, as the reference the away pushes are measured from.

`gain` is 0.6 for ±1 and 1.2 for ±2 (`step_gain` in `data/concepts.json`): ±2 is a clear step up from
±1, not a jump. A table entry written as `"direction": "away"` (for example Body hair −2 = away from
"body hair") is pushed away instead of toward.

| Slider | −2 · −1 · +1 · +2 |
|---|---|
| Breasts | flat · small · full · large |
| Penis size | tiny · small · large · hung |
| Penis girth | thin · slim · thick · very thick |
| Testicles | tiny · small · large · huge |
| Labia | closed · small · prominent · pronounced |
| Vulva puffiness | flat · slim · puffy · very puffy |
| Pectorals | flat · small · defined · large |
| Masculine ↔ Feminine | stark · masculine · feminine · strong |
| Waist | very narrow · narrow · wide · very wide |
| Hips | very narrow · narrow · wide · very wide |
| Abs | soft · faint · defined · six-pack |
| Muscle definition | soft · light · toned · shredded |
| V-line | smooth · faint · defined · sharp |
| Weight | skinny · slim · chubby · heavy |
| Body hair (torso) | none · light · natural · hairy |
| Arm & leg hair | none · light · natural · hairy |
| Pubic hair | none · sparse · natural · thick |
| Armpit hair | none · sparse · natural · hairy |
| Shoulders | narrow · slim · broad · very broad |
| Back muscles | smooth · soft · toned · muscular |
| Lats (V-taper) | narrow · slim · wide · very wide |
| Biceps | thin · slim · toned · large |
| Butt | flat · small · round · large |
| Anus | tight · small · puffy · very puffy |
| Nipples / areolae | tiny · small · large · puffy |
| Breast shape | very perky · perky · soft · saggy |
| Thighs | thin · slim · thick · very thick |
| Foreskin | circumcised · short · natural · long |
| Belly | flat · trim · chubby · pot belly |
| Erection | flaccid · soft · erect · rock hard |
| Vulva wetness | bone dry · dry · wet · dripping |
| Nipple erection | soft · relaxed · erect · hard |
| Skin flush | none · less · flushed · deep flush |
| Sweat | dry · less · sweaty · drenched |
| Penis veins | smooth · fewer · veiny · very veiny |
| Scrotum | tight · taut · loose · hanging |
| Vulva texture | smooth · simple · detailed · fleshy |
| Facial hair | clean-shaven · less · stubble · full beard |
| Skin realism *(Realistic, Stylized)* | airbrushed · smooth · natural · detailed |
| Cum | none · less · some · a lot |
| Skin tone | pale · fair · tan · dark |
| Nipple melanin | pale pink · pink · brown · dark |
| Penis wetness | dry · less · wet · dripping |
| Anus wetness | dry · less · wet · dripping |

**Skin realism** fights the plastic, airbrushed skin of many realistic and 3D checkpoints (+: pores,
fine imperfections, subsurface scattering; −: retouched smooth skin). It is off in the Anime profile
(the slider says so): anime skin has no pores to draw. It covers the whole body with its own large
mask (0.6) and reaches the face skin; Protect face keeps eyes, eyebrows, nose and lips.

**Skin tone** colours the body and the face skin together (a wide mask, 0.7, and Protect face
keeps only the face core), so face and body match; it has no cap on the + side. **Nipple melanin**
is the colour of nipples and areolae, a detail slider with a small mask; Anime has no tag for pink
nipples, so its − side pushes away from `dark nipples`.

**Fluids** (Arousal tab: Vulva wetness, Penis wetness, Anus wetness, Cum) are detail sliders. The
wetness ones have small masks; **Cum** lands anywhere, face and hair included (a wide mask, 0.5; the
prompt says where: *cum on face*, *in hair*), and Protect face keeps only the face core. Where they meet other sliders on the same pixels (a penis with Penis, Erection, Penis
veins, Penis wetness and Cum all moved) the per-pixel overlap budget caps their sum, which is what
stopped the "metal and fire" pile-up; with many sliders on one part each still pushes less, so move
the ones that matter. Danbooru has no tag for a wet penis: Anime uses `precum` / `cum on penis`.
| Anatomy fix | (one-sided: 0 · +1 fix · +2 strong fix) |
| Pose energy | stiff · still · dynamic · dramatic |
| Pose style | powerful · confident · graceful · sensual |
| Camera height | high above · above · below · low below |
| Head turn | at viewer · toward · three-quarter · away |
| Scars | none · fewer · some · many |
| Stretch marks | none · fewer · some · many |
| Cellulite | none · fewer · some · visible |
| Moles / beauty marks | none · fewer · some · many |
| Birthmarks | none · fewer · small · large |
| Freckles | none · fewer · some · dense |
| Wrinkles | (one-sided: 0 · +1 fine lines · +2 deep) |

**Body preset** (`data/presets.json`). A dropdown above the tabs and a Variant choice: picking one
fills the sliders of its scope for the panel's Profile and sets the scope's other sliders to 0. The
scope is every body slider: Body, Muscle, Female and Male tabs, Skin realism, Cellulite, the marks,
Wrinkles and all hair. Never touched: Pose, Arousal, Skin tone and Nipple melanin (no body type comes
with a skin colour), Anatomy fix. 26 types, 13 for each sex, 3 – 4 variants each (96 in all), Porn
star among them. Per profile: a type may override sliders on one profile (Porn star's airbrushed
skin on Realistic and Stylized), and a slider that is off on a profile (Skin realism, Cellulite and
Scrotum in Anime) stays 0 there. A variant moves at most 9 sliders and pairs none that clash, on every
profile. It is a shortcut on the panel only: nothing is computed for the preset itself, and the
image's PNG info keeps the slider values. A preset also sets three settings for its sliders: *Mask
area* 0.2 plus 0.03 for every big part it moves (shoulders, waist, hips, thighs, butt, lats, back,
belly), at most 0.35; *Max concepts* to the number of concepts the plan computes for the current
prompt and Profile (6 – 16), so nothing takes turns; *Anti-burn* from the CFG field: 0 up to CFG 7.5,
0.3 to 8.5, 0.5 to 10.5, 0.7 above, plus 0.1 when the variant moves 7 or more sliders. That costs
about +110 – 260 % time (+230 % on average); lower Max concepts for speed.

**Pose tab.** Checkpoints fall back on one stock pose: standing straight, arms at sides, eye-level,
looking at the viewer. The Pose sliders work on the composition instead of a body part, so they
differ from every other slider:

* they run in the **first 35 % of the schedule**, whatever Range start says (the pose is set there);
* they are **the whole picture**: no concept mask, so hair and background may change with them;
* **Pose energy** is how much the body moves (stiff → dramatic) and names no standing pose, so a
  sitting or lying prompt keeps its pose; **Pose style** is the bearing: −2 powerful, −1 confident,
  +1 graceful, +2 sensual, read differently on a woman and on a man (her sensual: arched back, hand
  in hair, looking back; his: arms behind head, leaning back, stretching). The prompt says who
  (1girl / woman / futa → hers, 1boy / man → his; neither → a neutral set); with both in the prompt
  it is off. Each step has several poses and each image of a batch takes one by its seed: a batch
  of four shows four poses of one bearing, and the same seed always the same pose. The minus side
  spreads the legs, which on a sitting or lying prompt reads as spread legs. Pose energy − with
  Pose style is a clash and is softened;
* **Protect face** covers Pose energy, Pose style and Camera height too: they leave the face to the prompt, their edit zeroed on the face like a body
  slider's, and the face mask follows the head every step while they run. It reduces face changes,
  it cannot stop them: a new pose or camera angle is a new drawing, and the checkpoint has no memory
  of the old face. Head turn always reaches the face (turning it is its job). For the *same* face in every pose use a face-identity tool next to the
  extension (IP-Adapter FaceID, PuLID or InstantID with the default render as reference, ReActor,
  or a character LoRA);
* they **never run in the hires pass** (the first pass has set the composition);
* the + steps also push away from the −1 entry, the stock pose / slightly-high eye-level shot /
  straight gaze, measured from the toward concept (contrastive), which is what breaks the default
  better than a prompt word.

**No subject position.** An earlier version had a Subject position slider that moved the model's
early x0 prediction sideways. It did not move the subject: the sampler builds the next latent mostly
from the current one (for Euler, x_next = x·σ'/σ + x0·(1 − σ'/σ), and σ'/σ is close to 1 in the
first steps), so only a sliver of the shift got in each step and the model re-centred the rest. A
post-CFG hook cannot move the latent itself. For an off-centre subject: a wider image cropped, an
OpenPose skeleton placed off centre, or regional prompting.

**Marks** (Hair & marks tab: Scars, Moles / beauty marks, Birthmarks, Freckles) are detail
sliders: body-wide masks read from their own nouns (`scar`, `mole`, `birthmark`, `freckles`),
Protect face keeps only the face core, and they wait for any shape slider. The prompt says where
(*scar on chest*, *mole under eye*); the slider says how much. **Stretch marks** (Hair & marks)
and **Cellulite** (Skin) work the same way but never touch the face; Cellulite is off in Anime (no
usable tag), and Danbooru's `stretch_marks` tag is rare, so Anime checkpoints draw it weakly. **Wrinkles** is one of them and
one-sided (0, +1 fine lines, +2 deep): it has no minus side, since smoothing wrinkles away is
pulling toward a younger look (Skin realism − already gives airbrushed skin). There is no Age
slider: a character's age is the prompt's (*mature woman*, *middle-aged man*), and no slider,
preset or concept text addresses age.

They are for "more or less" (stiff ↔ dynamic, from above ↔ from below, at the viewer ↔ away). An
exact pose (kneeling, lying on the side, a hand on the hip) is a prompt tag or ControlNet OpenPose,
which works together with the extension.

**Belly** is the belly alone (flat ↔ round pot belly, belly rolls); **Weight** is the whole build.
Its cap pushes away from a pregnant belly, which is where a big round belly drifts.

**Arousal.** Skin flush and Sweat only push *away* on the − side (from flushed / sweaty skin), never
toward a skin tone. Their edits stay off the face with Protect face on; for a blushing face, write
*blush* in the prompt. Erection is a shape slider (early steps), the others are detail sliders.

**Anatomy fix** only pushes away, in three separate concepts so they do not blur into one: extra or
fused fingers; extra arms and legs; extra nipples, breasts or penises. It costs three passes per
step. It does **not** run in the first 30 % of the range: pushing a limb while the composition is
still forming re-poses the whole body, and the face is re-drawn with it even though Protect face
keeps the edit itself off the face. Its concept mask is small (0.12 of the image, its own `area`),
so each push stays on the hands, the limbs, the chest and groin, and it has no global quality words
(`bad anatomy`, `deformed`), which change the whole picture. Its window ends at 85 % of the range, before
the finest detail forms. Its chest and genital push (`extra nipples, extra breasts, extra penis`) is
left out while any chest or genital slider is moved (`unless` in the table): pushing away from
"more breasts" on a chest that Breasts is shaping smeared both. Entries are at 0.6. It is a nudge,
not a guarantee: SEGA can only steer away from what the checkpoint associates with those words.

### Layer order

Over the steps (progress 0 = noisiest, 1 = cleanest; default range 0.10 – 0.60):

| Layer | When |
|---|---|
| Pose energy, Pose style, Camera height, Head turn | 0 – 0.35 of the schedule, before the range |
| Shape sliders (sizes, build, Skin tone, nudity lock) | the first 80 % of the range; while a Pose slider moves the body, from 70 % of the Pose window on |
| Anatomy fix | 30 – 85 % of the range |
| Detail sliders (hair, textures, nipples, veins, wetness, flush, sweat, cum, Skin realism) | 30 % of the range to 1.15 × its end; **60 %** when a shape slider runs on their part or a part they sit on (`wait_for`: pubic hair waits for Penis / Testicles / Belly / Weight, Skin realism, Flush, Sweat and Cum for any shape) |

Fused descriptions never mix phases: Penis + Erection (shape) and Penis veins + Penis wetness +
Foreskin (detail) are two concepts, so a size push never runs on into the last steps. Every body
slider has a region, so this staging sees it.

**Mask words, everywhere it matters.** A concept mask reads only the slider's own nouns (Penis
wetness: `penis`, `shaft`, `precum`; Sweat: `sweat`, `drenched`; Flush: `flush`, `chest`, `neck`;
Nipple melanin: `nipples`, `areolae`...). Surface and colour words (*wet, glistening, fluid, skin,
shiny, pink, hard, detailed, veins*) attend to every shiny thing in the picture, a neon sign
included, and never place a mask. Realistic texts say *vulva*, not *pussy* (a plain CLIP reads a
cat), and *sparse*, not *light* (lighting). Body-wide sliders (Muscle, Weight, Masculine ↔
Feminine, Flush, Sweat, Skin tone, Skin realism) have wide masks. A test walks every slider, profile
and step and fails on a concept whose mask would fall back to all its tokens, on a surface word in
an all-token mask, on an unresolved reference or a bad window.

### Sliders on the same body part

Several sliders can act on one body part: Belly, Abs and V-line on the belly; Breasts, Breast shape,
Nipples, Nipple erection and Pectorals on the chest; Penis, Erection, Penis veins, Foreskin; and so
on, and sliders named after different parts still meet on the same pixels (a crotch). Two things keep
them from piling up:

* **Overlap budget, per pixel.** Wherever several sliders edit the same pixels, whatever body part
  they are named after, their summed edit there is kept within 1.5 × the strongest single slider at
  that pixel (`overlap_budget`), before and after momentum. A slider alone on its pixels is
  untouched, and one slider's own concepts (its toward, opposite and cap pushes) are not capped
  against each other. (1.x budgeted by body-part name; a crotch where Testicles, Scrotum, Penis
  veins, Pubic hair, V-line, Abs, Flush and Sweat all met was not capped at all, and the summed
  push drew metal and fire there.)
* **One description per part.** Sliders that describe the same part are fused when two or more
  of them are moved (`fuse` in the table): Breasts + Breast shape; Nipples + Nipple erection +
  Nipple melanin; Penis size + Penis girth + Erection (shape) and Penis veins + Penis wetness +
  Foreskin (detail); Labia + Vulva puffiness + Vulva texture + Vulva wetness; Testicles (shape) and
  Scrotum (detail, after the testicles' shape). They become one concept per role, their texts
  joined ("large areolae, erect nipples, brown nipples"), at the strongest member's strength, not
  the sum. A checkpoint draws one coherent part from one description; three separate edits on the
  same few pixels drew smeared or doubled nipples. It also costs fewer passes. The status line
  names fused sliders.
* **Shape first, then detail.** In a region where a shape concept runs (Breasts, Penis), the
  region's detail concepts (nipples, veins) start at 60 % of the range instead of 30 %
  (`detail_after_shape`): drawn on a part still changing size they land off target.
* **Edit cap.** Whatever the number of sliders, at every pixel the summed edit stays within
  1.0 × the prompt's own CFG push there (smoothed, with a floor of a quarter of its mean so flat
  areas can still change; `edit_cap`, needs CFG > 1). The overlap budget compares sliders with
  each other, so it let one strong push on a small part through: Penis −2 on a neon scene drew a
  glowing ring at the base. Past what the model itself pushes, an edit leaves what it can draw.
  The console warns once when it bites.
* **Soft mask edges.** A concept mask ramps down over 1/16 of the latent (9 latent pixels at SDXL
  1024 px, two blurs) instead of a 2 – 3 pixel step: a strong edit cut off sharply at the rim of a
  small part drew a halo there.
* **Sign election** (as TIES merging) before the cap: on each latent value where sliders push
  opposite ways, the minority's push is dropped instead of summed. Opposite pushes only cancel
  into mush and use up the budget; without them the agreeing sliders keep more of it. Each pixel
  has 4 latent values, so no merge can give a dozen sliders on one spot each their full push: the
  cap keeps the image drawable, sign election makes the most of what it allows.
* **Warnings.** The status line warns when 4 or more sliders are moved on one region
  (`region_warn`) and when the expected extra time passes +800 % (`warn_time`; also in the
  console). Once per pass the console warns when the cap had to cut a multi-slider spot below 60 %
  over more than 1 % of the image (`overlap_warn`), naming how many sliders met there.
* **Clashes.** Some pairs pull the same part opposite ways at certain signs (`clashes`): Belly + with
  Abs +, Belly + with Weight −, Muscle + with Weight +, Breasts − with Breast shape +, Vulva − with
  Vulva texture +, Erection − with Penis veins +, Masculine with Breasts +… Each slider's *opposite*
  push is exactly what the other one moves toward, so they would fight twice. Both lose that
  opposite push (their toward pushes stay) and the status line names the pair. Results are still
  better with only one of the two moved.

**No tag, no slider.** Danbooru has no fitting tag for Nipple erection (its `erect nipples` is an
alias of `covered_nipples`, nipples showing through a shirt, so it would paint the shirt on) or for
Scrotum. In the Anime profile those two are off and the status line says so; Realistic and
Stylized have them. Penis girth and Vulva puffiness use e621's `thick penis` / `puffy pussy` in
Anime: Pony knows them, Illustrious and NoobAI barely (Danbooru has neither). No shape step names a
fluid: precum is Penis wetness's, never Erection's.

**Weight** (formerly *Body fat*) has one meaning: thin ↔ heavy. The old +1 step "curvy" was an
hourglass shape (it overlapped Hips, Waist and Breasts and feminised men), and the old −2 "lean"
meant firm in Realistic but skinny in Anime. Firmness and visible muscle are **Muscle** and **Abs**.
Infotexts with the old step names (`body-fat=lean / curvy / plump`) still paste back.

Anime has no Danbooru tag for narrow shoulders, thin arms, a small butt or a tight anus; there those
steps push away from the opposite tag (`broad shoulders`, `biceps`, `spread anus`) or use
`flat ass`. Stylized uses the Realistic texts for these six and for the sliders added after them.

The concept texts live in `data/concepts.json`, one entry per step and profile (`text`, `direction`,
`scale`, optional `warmup`). The file is re-read when it changes; no restart needed. Anime entries
are checked against the Danbooru tag list of tag-autocomplete (`DANBOORU_CSV=... python
tests/test_concepts.py`); where no tag exists for a step, the nearest real tag is used with a
different `scale` (for example Penis −2 = `small penis` × 1.3).

### Control mode

Who wins when your prompt and a slider disagree, like ControlNet's three control modes. It changes
one number, the gain every edit gets, and whether Perp-Neg runs:

| Mode | Gain | Perp-Neg |
|---|---|---|
| Balanced (default) | × 1.0 | on |
| Prompt | × 0.7 | on: the slider only refines what the prompt asked for |
| Sliders | × 1.3 | off: an edit may run against the prompt (prompt *large breasts*, Breasts −2: the slider wins) |

Perp-Neg drops each edit's part along the prompt's own guidance (cond − uncond); it needs CFG > 1.
The gains are `control` in `data/concepts.json`. Infotext: `control=prompt` / `control=sliders`
(1.x images with `control=sega` paste as Sliders).

### Nudity lock (automatic)

No slider. When the prompt asks for full nudity (`nude`, `completely nude`, `fully nude`, `naked`,
`no clothes`) and names no partial nudity (`topless`, `bottomless`, `shirtless`, `bare-chested`,
`partially nude`, `half-naked`, `nearly` / `mostly` / `almost nude`), no garment or accessory
(`shirt`, `skirt`, `dress`, `panties`, `bra`, `suit`, `uniform`, `stockings`, `thighhighs`, `heels`,
`boots`, `gloves`, `choker`…) and no clothed person (`clothed`, `CMNF`, `CFNM`), one concept pushes
away from clothes. A garment named only to say it is not there (`no clothes`, `without panties`,
`wearing nothing`) does not count as one.
measured from the prompt (so only clothes move), with its own mask area 0.5, at 0.6 of a step's push,
during the shape part of the range. `nude lipstick`, `nude-colored stockings`, `nude colour` do not count.
Anything else keeps its clothes. It costs one concept pass per step, never runs in the hires pass or
on tiles, and the status line says when it is on. With it, the extension acts on a nude prompt even
with every slider at 0. The texts are `nudity_lock` in `data/concepts.json`.

### Keeping the face

Three things keep the face: **Protect face** (no slider edit on the face at all, see Options),
the **concept mask** below (each edit only on its own body part), and the job check: the
extension's sampling callback only acts on its own job, so ADetailer's face repaint (a nested job
that runs while the main one is still open) is never touched. Masculine ↔ Feminine names jaw and
throat traits; with Protect face on, its edit on the face is zeroed too. Edits early in the steps
can still shift a face indirectly (the UNet's self-attention spreads changes); if a face drifts,
start the range later (0.15 – 0.20) or lower Strength.

Protect face has three levels, set per slider: the whole face (every body slider, Pose energy, Pose style,
Camera height, the nudity lock); only the **core**, eyes, eyebrows, nose and lips (Facial hair,
Skin realism and Cum, which have to reach jaw, chin, cheeks, forehead and hair; the core is where the face probe's
attention is above 0.65 of its peak, `face_core_threshold`, no extra pass). Face shape and bone
structure lie outside the core: a beard may thicken the jawline, as a real one does. For the very
same face, use a face-identity tool (IP-Adapter FaceID, PuLID); none (Head turn).

### The concept mask

**Mask words.** A slider can name the words its mask is read from (`mask_words`): Pubic hair reads
`pubic` / `groin`, Armpit hair `armpit`, Body hair `chest` / `stomach` / `torso`, Arm & leg hair
`arm` / `leg`, Facial hair `beard` / `stubble` / `mustache` / `face`. CLIP ties a bare *hair* to
the hair on the head: read from every token, the Hair tab's masks took in the head and the face
(and phrases like *detailed hair texture* pushed toward a hairstyle). The hair concepts now name
where the hair is (*thick dense pubic hair*, *hairy armpits, dense armpit hair*) and have a small
mask (0.15). Every other slider still reads all the tokens of its concept.

Without it, an edit can land anywhere the concept's prediction differs from your prompt's, the face
included, and early in the steps a change of build drags the face along. With **Concept mask** on
(the implicit masking of LEDITS++, Brack et al., CVPR 2024):

1. During each concept's own UNet pass, the cross-attention of the concept's words (the
   "small natural breasts" appended to your prompt) is recorded on the coarse UNet layers.
2. That map, thresholded to **Mask area** and given a soft edge, multiplies the concept's edit. For a
   contrastive pair, the map of both concepts is used.
3. Everything the words do not look at (face, hair, hands, background, the other person) gets no
   edit at all.

The token positions come from the checkpoint's own tokenizer, so long prompts, BREAK and weights are
handled. It costs about 5 – 10 % more time and no extra UNet pass. Main passes are never touched;
only the concept calls compute attention themselves (same result as the WebUI's attention, checked
in the tests). A UNet block another extension already replaced keeps that replacement and is simply
not recorded.

Limits: the mask follows what the words attend to. **Masculine ↔ Feminine** names jaw and throat
traits (`square jaw`, `soft jawline`, `adam's apple`), so its mask does include the jaw: that slider
still changes the face, by design. In the very first steps the maps are blurry; keep Range start at
0.15 or later.

### Keep apart (anti-bleed between two people)

CLIP reads the whole prompt at once, so in `1girl, blonde hair, large breasts, 1boy, black hair`
"blonde" and "large" also reach the boy. With **Keep apart** = `blonde, black, large`, every
phrase's tokens come from an encoding in which the isolate words of the *other* phrases are replaced
by padding (same token count): a phrase still sees its own words, the rest of the prompt does not.
The result is blended with the normal encoding at 0.6 (`isolate_strength`, fixed). This is the
idea of hnmr293's Cutoff, re-implemented. A phrase ends at a comma, a semicolon or a full stop (not the
dot of a number such as `1.2`), so sentence-style prompts work too: `A blonde woman lies on the bed. A
man with black hair sits beside her.` isolates per sentence.

* Applied to the main prompt's conditioning (per image, every step) and to every concept text, so the
  slider edits are measured against the same isolated prompt (Baseline Prompt).
* Costs a few text-encoder runs per batch (one per phrase that has an isolate word), nothing per step.
* Not applied to prompts with `AND` or prompt editing (`[from:to:when]`); the console says so.

### Why clothes do not come back

Earlier versions sometimes painted clothes back on, or blended body parts, even with `nude` /
`completely nude` in the prompt. Each cause and what stops it now:

| Cause | Now |
|---|---|
| A bare phrase ("small breasts") is, to these checkpoints, a clothed stock photo; measured against the negative prediction, the clothing is part of the edit. | Baseline **Prompt** (default): every concept is your prompt + the phrase, measured against your prompt, so the scene and the nudity cancel. Baseline **Negative**: the phrase is led by your prompt's own nudity words (`completely nude, small breasts`). |
| An **away** push measured from the plain prompt: "away from *prompt + large breasts*" also means "show less breast", and the cheapest way for the model to do that is to cover it. | Away pushes are **contrastive**: measured from the step's own concept ("prompt + small breasts" vs "prompt + large breasts"). Both show the same nude body, so only size is left. A step whose own entry is an "away" entry no longer adds an uncontrasted away push. |
| Baseline Negative: an away concept with the nudity words, measured against the negative prediction, pushes away from nudity itself. | It is measured against the nudity words alone (one extra reference pass, shown in the edit count). |
| A batch with different prompts (Dynamic Prompts, prompts from file, a hires prompt) built every image's concepts on the first prompt; the other images' scene differences, clothes included, leaked into their edit. | Each image's concepts are built on its own prompt (and on the hires prompt in the hires pass). |
| Concept words that imply clothing: "cinched waist" (corset / belt), "bodybuilder" (posing trunks), "adonis **belt**". | Replaced with anatomy words; a test fails if a garment, a person noun or such a word comes back into `data/concepts.json`. |

If clothes still appear: keep Baseline on Prompt, lower Strength, and make sure the nudity is in the
positive prompt (the extension reads it from there).

### If a slider will not go below 0

The WebUI saves every slider's minimum and maximum in `ui-config.json` on first start and forces them
back afterwards. The first versions of these sliders went from 0 upward, so `ui-config.json` kept
`minimum: 0`. The sliders are now kept out of `ui-config.json`, which fixes it. If you still see it,
delete the lines with `bodysliders` / `Body Sliders` from `ui-config.json` and restart.


### Futa, cuntboy, femboy

* `futa` / `futanari` / `dickgirl` / `newhalf` in the prompt: Men sliders (penis, testicles, foreskin)
  may push *toward* on a woman.
* `cuntboy` in the prompt: Women sliders (vulva, breasts, breast shape) may push *toward* on a man.
* Femboy: `1boy, femboy` is a scene with men only, so Masculine ↔ Feminine works (+1 / +2), and
  Breasts with Who = Women only pushes away (a flat chest). Set Who to Everyone to add breasts.
* Masculine ↔ Feminine is off only when the prompt has both women and men. Its concepts name jaw and
  throat traits, so it changes the face even with the concept mask on.

### Toward, away and couples

SEGA is applied to the whole image in this version.

* **Away** pushes are safe globally: pushing away from "oversized penis" does nothing to a woman.
* **Toward** pushes are not: "toward small breasts" in a couple scene can put breasts on the man.
  Toward pushes are therefore weaker, and each slider has a **Who** selector. When Who is *Women* or
  *Men* and your prompt mentions both women and men (`1girl`, `woman`, `1boy`, `man`, `hetero`, ...;
  the prompt is only read), that slider uses **away pushes only** and the status line says so.
* **Masculine ↔ Feminine** is off in a scene with both women and men (it needs per-person regions,
  planned for v2).

## What it can and cannot change

It can change **size, proportion, muscle tone and sex traits** that the checkpoint knows how to draw.
It cannot add fine detail the checkpoint never learned: SEGA steers the model toward or away from
what it already associates with a concept text. Results depend on the checkpoint; tune the levels
with the X/Y/Z plot.

## Speed, VRAM and many concepts

Each concept pass is one more UNet pass on a step where it runs. The status line's extra time
adds up every concept's own window over the whole schedule (detail sliders run past Range end, Pose
sliders before Range start) next to the 2 passes of a CFG step.

| 12 concepts, range 0.10 – 0.60 | Passes per step | Extra time |
|---|---|---|
| Max concepts 12 or more | 12, fewer in the parts of the range where shape and detail sliders do not overlap | ≈ +250 – 300 % |
| Max concepts 6 (default): they take turns | ≈ 6 | ≈ +150 % |

* **VRAM** does not grow with the number of concepts: they are predicted a few at a time, as many
  per call as free VRAM allows; an out-of-memory error inside the extension is caught, the batch is
  halved and the step retried. A crash of the whole machine (system RAM or the driver's
  shared-memory fallback running out) cannot be caught by any extension; the low peak avoids it.
* **RAM**: encoded concept texts are cached on the CPU (at most 128 entries and 64 MB; a long SDXL
  prompt is about 1 MB an entry) and dropped when the checkpoint changes. When a job ends, its
  per-step tensors (differences, attention maps, face masks, momentum) are released from the GPU.
* **Before taking turns**, sliders give up what they can best do without: first detail sliders'
  caps, then their counter-pushes, then shape sliders' caps (the status line says how many).
  Two hair sliders at the default Max 6 then still run every step instead of taking turns.
* **Taking turns** (above Max concepts): a concept not recomputed on a step reuses its last
  difference, faded × 0.6 for every step it is reused (it was measured on an earlier latent, and
  applied at full strength it lands a little off target: ghosting on shared body parts). Momentum is
  off while concepts take turns. Cleanest results: Max concepts ≥ the concept count.
* **Overlap budget.** Where sliders meet on the same pixels their sum is capped (see *Sliders on
  the same body part*). Two sliders asking for the very same concept in one phase (Breasts −2 and
  Pectorals −2: *flat chest*) share one pass at the stronger scale, not twice the push; a slider's
  cap is dropped while another moved slider asks for that same thing (Hips + is capped away from
  *huge ass*, which Butt +2 asks for).
* **Odd image sizes** (1080 px, 1328 px on SDXL) are fine: the concept and face masks are read from
  attention layers whose size rounds up.

With Baseline **Prompt**, SEGA needs your prompt's prediction (`cond_denoised`), which exists at
every CFG including **CFG 1**. With Baseline **Negative** it needs the negative pass, so it does
nothing at CFG 1 (the console says so once).

## Tips

* **Lower CFG helps.** High guidance (high CFG, PAG, Pre-CFG and similar) exaggerates every concept,
  size included. Try CFG 4–6 on SDXL before raising Strength.
* Start with the defaults. If the change is too weak, raise Strength or use Control mode Sliders; if
  other parts of the image change, keep Concept mask on and lower Mask area (Advanced).
* Use the X/Y/Z plot on a fixed seed to find which levels look right with your checkpoint.

## PNG info, API and X/Y/Z plot

* Infotext, slider values only (the prompt stays exactly what you typed):
  `Lewd Anatomy: profile=anime, strength=1.0, breasts=full(w), v-line=sharp, locality=0.9, range=0.10-0.60`.
  Added when they differ from the defaults: `control=prompt|sliders`, `max=N`, `baseline=negative`,
  `face=off`, `mask=off` or `area=N`, `rescale=N`, `isolate=a|b`, `exaggerated=on`, `hires=on`.
  Infotexts of 1.x versions paste back; their `rotate`, `staggered`, `orthogonal`, `fast`,
  `range_by` and `isolate_w` are ignored (fixed in 2.0), `control=sega` reads as Sliders, and the
  old names (`Body Sliders: ...`, `Physique: ...`) are still read.
  `(w)` / `(m)` / `(e)` is the Who setting. Pasted back with the usual "Send to" / paste button.
* API: `"alwayson_scripts": {"Lewd Anatomy": {"args": [...]}}` in this order: enabled, profile,
  control (`"Balanced"` / `"Prompt"` / `"Sliders"`), options (a list of the toggle labels that are on,
  e.g. `["Protect face", "Concept mask"]`), strength, start, end, baseline (`"Prompt"` /
  `"Negative"`), mask area, max concepts, anti-burn, keep-apart words, then each slider
  (−2 … 2) in the order of `data/concepts.json` followed by its Who value if it has one.
* X/Y/Z plot axes: one per slider (step names or −2 … 2, `off` for 0), `Strength`, `Locality`,
  `Max concepts`, `Mask area`, `Anti-burn`. An axis turns Lewd Anatomy on
  for that cell.

## Compatibility

SEGA is installed as a **post-CFG function** on a clone of the WebUI's UNet patcher for each
sampling pass. That hook is a list on all three WebUIs, so it **chains** instead of replacing:

1. the CFG function, whoever set it (plain CFG, Steer's FDG / APG / CHG, Dynamic Thresholding,
   reForge's built-in APG, ...),
2. post-CFG functions registered before Lewd Anatomy (for example Steer's PAG / SEG / SAG),
3. Lewd Anatomy's SEGA correction (added on top),
4. post-CFG functions registered after it.

The concept call goes through the WebUI's own batch runner (`calc_cond_uncond_batch` on Forge /
Neo, `calc_cond_batch` on reForge) with the same timestep, `model_options`, ControlNet and patches as
the main call. Concept conditionings are encoded once with the checkpoint's own text encoder(s),
SDXL size embedding included, and cached per checkpoint.

* Models: SD 1.x / 2.x / SDXL family (UNet). **DiT models (Anima, Flux, SD3 / 3.5, Qwen-Image,
  Lumina, Chroma, Wan…) are experimental and not enabled in this version: they are skipped with a
  console note.** The SEGA core (sliders, contrastive pairs, taking turns, overlap budget, Anti-burn,
  Baseline Prompt even at CFG 1) does not depend on the architecture and is the planned DiT mode; the
  concept mask (UNet `attn2` patches) and Keep apart (CLIP chunking) would stay off there.
* **Tiled upscaling** (SD upscale, Ultimate SD upscale and other selected scripts whose title says
  upscale / tile): they run img2img on every tile with the same settings, and a tile with half a body
  would still be pushed toward the concept, so the sliders are switched off on tiles (console note).
  Anti-burn and Keep apart still run. ESRGAN-type upscalers do not use the UNet: unaffected.
* **ADetailer** (Bing-su): runs only the scripts in its allow list; Lewd Anatomy is not in it, so it
  does not run in ADetailer's repaint. Adding it there would push the face crop too: not advised.
* Batch size > 1, Hires fix (separate switch), img2img and inpainting, ControlNet, AND prompts.
* **Impact ADetailer**: its repaint is a nested job; the WebUI resets the UNet for every pass, so
  Lewd Anatomy does not run there. `lib_lewd_anatomy/api.py` exposes `current_patch(p)` for an
  extension that wants to opt in (see the docstring there).

## Tests

```bash
python tests/test_concepts.py                       # tables, planning, infotext (no torch)
python tests/test_sega.py                           # SEGA math and engine (torch)
python tests/test_extras.py                         # Perp-Neg, Control mode, sigma range, anti-burn, keep apart
cd <webui root> && python <ext>/tests/test_in_host.py   # through the WebUI's own sampling code
DANBOORU_CSV=<tagcomplete>/tags/danbooru.csv python tests/test_concepts.py   # anime tags exist
```

## Credits

* SEGA: Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski,
  Kristian Kersting. *SEGA: Instructing Text-to-Image Models using Semantic Guidance.* NeurIPS 2023.
  [arXiv:2301.12247](https://arxiv.org/abs/2301.12247). Masking and momentum follow the reference
  implementation in diffusers (Apache-2.0, see `NOTICE.md`).
* Concept mask: the implicit masking of LEDITS++ (Manuel Brack, Felix Friedrich, Katharina
  Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, Apolinário Passos. *LEDITS++:
  Limitless Image Editing using Text-to-Image Models.* CVPR 2024.
  [arXiv:2311.16711](https://arxiv.org/abs/2311.16711)), re-implemented for the WebUI's attention
  patches; no code copied.
* Perp-Neg: Mohammadreza Armandpour, Ali Sadeghian, Huangjie Zheng, Amir Sadeghian, Mingyuan Zhou.
  *Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem
  and Beyond.* 2023. [arXiv:2304.04968](https://arxiv.org/abs/2304.04968).
* CFG rescale: Shanchuan Lin, Bingchen Liu, Jiashi Li, Xiao Yang. *Common Diffusion Noise Schedules
  and Sample Steps are Flawed.* WACV 2024. [arXiv:2305.08891](https://arxiv.org/abs/2305.08891).
  The arithmetic matches ComfyUI's RescaleCFG node.
* Keep apart: the idea of hnmr293's *sd-webui-cutoff*, re-implemented, no code copied.
* The hook pattern follows Steer (CFG Attention Guidance).

Thanks also to **Claude**, Anthropic's AI assistant, for help building this extension.

## Status: 2.0

2.0 trimmed the panel (Orthogonal, Staggered, Auto batch, Rotate and Range by are fixed;
Fast early steps and the AND_PERP / AND_SALT keywords are gone) and left one budget. The feature set is frozen. Every feature is checked against the three WebUIs' own sampling code with
stand-in or tiny random models (`tests/`), but **not yet on real checkpoints**. Before a release:

1. Off = byte-identical: the same seed with the extension off, and not installed, on Forge, reForge and
   Neo.
2. X/Y/Z sweeps per slider (−2 … +2) on an SDXL realistic, an anime (Illustrious / Pony) and a
   stylized checkpoint; tune `base_scale`, `step_gain`, Locality, Mask area and the range from them,
   per profile if needed.
3. Face kept with the mask on / changed with it off; no clothes coming back with `nude` in the
   prompt; a man-and-woman scene.
4. Speed and VRAM at 1, 6 and 12 concepts, Max concepts 6 and 12.
5. Anti-burn at CFG 10, Keep apart on a two-person prompt, the three Control modes.

Only fixes and tuned numbers go in until this list is done.

## License

MIT, see `LICENSE`. Third-party notices in `NOTICE.md`.
