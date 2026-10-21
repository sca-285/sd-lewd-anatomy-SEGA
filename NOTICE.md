# Notices

## SEGA (method)

Lewd Anatomy implements Semantic Guidance as described in:

> Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski,
> Kristian Kersting. "SEGA: Instructing Text-to-Image Models using Semantic Guidance."
> Advances in Neural Information Processing Systems 36 (NeurIPS 2023). arXiv:2301.12247.

## LEDITS++ (method)

The concept mask in `lib_lewd_anatomy/attention.py` implements the cross-attention part of the
implicit masking described in Brack et al., "LEDITS++: Limitless Image Editing using Text-to-Image
Models", CVPR 2024 (arXiv:2311.16711). It was written for the WebUI's attention patches; no code
was copied.

## diffusers (adapted logic)

The per-concept quantile mask (per sample and channel), the momentum update
and the warmup behaviour in `lib_lewd_anatomy/sega.py` follow the
`SemanticStableDiffusionPipeline` of Hugging Face diffusers
(`src/diffusers/pipelines/semantic_stable_diffusion/pipeline_semantic_stable_diffusion.py`).
No code was copied verbatim; the logic was re-expressed for the WebUI's
denoised (x0) post-CFG hook.

    Copyright The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

## Danbooru tag check

Anime / Pony concept texts were checked against `tags/danbooru.csv` of
DominikDoom/a1111-sd-webui-tagcomplete. The file is not included.
