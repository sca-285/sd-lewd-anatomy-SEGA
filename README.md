# Lewd Anatomy · SEGA Semantic Guidance

**An 18+ extension for Forge, reForge, and Forge Classic (Neo).**

Lewd Anatomy provides sliders for adjusting the proportions, anatomy, skin, body hair, and pose of clearly adult generated characters. It uses **Semantic Guidance (SEGA)** during sampling to steer selected visual traits without rewriting your prompt, changing model weights, or requiring LoRAs.

> **Content note:** This is an adult / NSFW-focused project and includes controls for sexual anatomy. It is intended for clearly adult characters only. The extension is not an age filter or a safety system; users are responsible for their prompts, generated images, and compliance with applicable rules.

**Full documentation:** [Technical details and complete reference](docs/DETAILS.md) · [Third-party notices](NOTICE.md) · [License](LICENSE)

## Features

- **56 sliders across eight tabs** for pose, body proportions, muscle, adult anatomy, skin, arousal-related details, hair, and marks.
- **Body presets** with multiple variants to get started without setting every slider by hand.
- **Profiles** for Realistic, Stylized, and Anime checkpoints, plus an Auto option that guesses from the checkpoint name.
- **Prompt-aware edits** that steer traits relative to the image described by your prompt instead of changing the prompt itself.
- **Optional masking and face protection** to help keep edits focused on the intended area.
- **Control and performance settings** for balancing slider influence, edit range, and generation time.
- **PNG infotext support** so settings can be restored from compatible generated images.

Results depend on the checkpoint and prompt. SEGA can steer concepts the model has learned, but it cannot reliably add details the model does not know how to draw.

## Compatibility

| WebUI | Status |
|---|---|
| Forge | Supported target |
| reForge | Supported target |
| Forge Classic (Neo) | Supported target |

| Model family | Status |
|---|---|
| Stable Diffusion 1.x / 2.x and SDXL-based checkpoints | Supported |
| DiT-based models, including Flux, SD3, Anima, Qwen-Image, Lumina, Chroma, and Wan | Not supported in v2.0; the extension skips these models and writes a note to the console |

The code has been checked against the sampling paths of the supported WebUIs using tests and stand-in models. It has **not yet been comprehensively tuned on real checkpoints**, so treat v2.0 as a work in progress and expect results to vary between models.

## Installation

1. Stop the WebUI.
2. Copy the `sd-webui-lewd-anatomy-sega` folder into the WebUI's `extensions/` directory.
3. Restart the WebUI.
4. Open **txt2img** or **img2img** and find the **Lewd Anatomy** panel.

After the repository is public, you can also use the WebUI's **Extensions → Install from URL** option with the repository URL.

**Upgrading from an earlier version:** If you previously installed *Body Sliders* or *Physique*, remove the old extension folder before installing this one. Do not run multiple versions at the same time.

## Quick start

1. Enable **Lewd Anatomy**. Start with the default settings.
2. Choose a **Profile**: *Realistic* for photo-oriented checkpoints, *Stylized* for 2.5D / 3D / cartoon-style checkpoints, or *Anime* for tag-oriented checkpoints such as Pony, Illustrious, NoobAI, and Animagine. *Auto* makes a guess from the checkpoint name.
3. Adjust only the sliders you need. **0** means off; negative values steer a trait lower than the prompt would normally produce, while positive values steer it higher.
4. Avoid asking the prompt and a slider for contradictory results. The prompt should describe the scene; use the slider to refine the relevant trait.
5. Check the status line below the panel controls for active edits, estimated extra work, and warnings.

The **Guide** button beside *Profile* provides a short in-panel explanation. The full reference explains each control and its behaviour.

## Tips

- **Start with a few sliders.** More active concepts usually mean more sampling work and longer generation times.
- **Keep masking enabled** while learning the extension. It helps limit an edit to the intended area; it cannot guarantee that every other part of the image will remain unchanged.
- **High CFG can amplify edits.** If the result looks exaggerated or unstable, try lowering CFG before increasing slider strength.
- **Use a fixed seed and X/Y/Z plot** to compare slider values for your checkpoint.
- A body preset is a starting point, not a requirement. Review the resulting slider values and adjust them to fit the image you want.

## Important limitations

- The extension steers the model during sampling. It does not edit the prompt, add a LoRA, or modify model weights.
- Outputs are checkpoint-dependent and are not guaranteed to follow each slider precisely.
- DiT-based models may not work properly.
- Lewd Anatomy does not run in ADetailer or Impact ADetailer repaint passes. During tiled upscaling, anatomy sliders are disabled on individual tiles to avoid pushing partial body regions in unintended ways. See the [technical reference](docs/DETAILS.md) for further compatibility notes.

## Credits and license

Lewd Anatomy builds on ideas from **SEGA**, **LEDITS++**, **Perp-Neg**, and **CFG rescale**. The Keep apart feature was inspired by the general approach used in `sd-webui-cutoff`. Full references and implementation notes are in [NOTICE.md](NOTICE.md) and [docs/DETAILS.md](docs/DETAILS.md).
Released under the **MIT License**. See [LICENSE](LICENSE) for the license text and [NOTICE.md](NOTICE.md) for third-party notices.
