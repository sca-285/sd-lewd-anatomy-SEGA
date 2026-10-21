"""Concept attention masks (the implicit masking of LEDITS++, Brack et al., CVPR 2024).

Each concept's edit is limited to where the concept's own words look. During
the concept's UNet pass, the cross-attention maps of the appended concept
tokens ("small breasts", "round buttocks") are recorded; their soft-thresholded
map multiplies that concept's difference. A face, a background or another body
part the concept words do not attend to is left as the prompt made it.

How it hooks in (Forge, reForge, Neo): transformer_options["patches_replace"]
["attn2"][(block, number)] replaces cross-attention for every transformer of
that UNet block, only in the copy of model_options used for the concept call.
The replacement gets q, k, v already projected and computes attention itself:
recorded layers from explicit probabilities (it needs them), the others with
scaled_dot_product_attention. Main passes are never touched.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

KEY = "lewd_anatomy_attention"     # transformer_options key carrying the recorder
MAX_SHARE = 1.0 / 16               # record layers with at most this share of the latent's positions


# --------------------------------------------------------------------- UNet blocks

def _has_transformer(module) -> bool:
    return any(type(m).__name__ in ("SpatialTransformer", "Transformer2DModel") for m in module.modules())


def attention_blocks(diffusion_model) -> list:
    """("input" | "middle" | "output", number) of every UNet block with cross-attention."""
    out = []
    for i, b in enumerate(getattr(diffusion_model, "input_blocks", []) or []):
        if _has_transformer(b):
            out.append(("input", i))
    mid = getattr(diffusion_model, "middle_block", None)
    if mid is not None and _has_transformer(mid):
        out.append(("middle", 0))
    for i, b in enumerate(getattr(diffusion_model, "output_blocks", []) or []):
        if _has_transformer(b):
            out.append(("output", i))
    return out


def with_recorder(model_options: dict, blocks: list, recorder) -> dict:
    """A copy of model_options whose attn2 in `blocks` goes through `recorder`.

    Only the dictionaries on the path are copied, so the main pass's options are
    never touched. A block another extension already replaced keeps that
    replacement (and is not recorded).
    """
    options = dict(model_options)
    to = dict(options.get("transformer_options", {}))
    replace = dict(to.get("patches_replace", {}))
    attn2 = dict(replace.get("attn2", {}))
    taken = {k[:2] for k in attn2 if isinstance(k, tuple)}
    for b in blocks:
        if b not in taken:
            attn2[b] = recorder
    replace["attn2"] = attn2
    to["patches_replace"] = replace
    options["transformer_options"] = to
    return options


# --------------------------------------------------------------------- recording

class Recorder:
    """attn2 replacement for one concept call: computes attention, and on coarse layers
    adds the attention of each row's concept tokens to a map at latent size.

    rows: one (token positions, token length of that row's own conditioning) per batch
    row of the call, None for rows not to record. latent_hw: (H, W) of the latent.
    max_share: record layers with at most this share of the latent's positions
    (1/16 = SDXL's 32x32 layers at 1024 px, SD 1.5's 16x16 at 512 px).
    """

    def __init__(self, rows: list, latent_hw: tuple, max_share: float | None = None):
        self.rows = rows
        self.H, self.W = int(latent_hw[0]), int(latent_hw[1])
        self.max_share = MAX_SHARE if max_share is None else max_share
        self.sum = torch.zeros(len(rows), self.H, self.W)
        self.count = 0

    def _cols(self, row: int, tokens: int) -> list | None:
        spec = self.rows[row] if row < len(self.rows) else None
        if not spec:
            return None
        positions, length = spec
        if not positions or length <= 0:
            return None
        reps = max(1, tokens // length)            # conditionings stretched to a common length repeat
        return [p + j * length for j in range(reps) for p in positions if p + j * length < tokens]

    def __call__(self, q, k, v, extra_options):
        heads = int(extra_options.get("n_heads", 1))
        b, n, inner = q.shape
        d = inner // heads
        qh = q.view(b, n, heads, d).transpose(1, 2)
        kh = k.view(k.shape[0], k.shape[1], heads, d).transpose(1, 2)
        vh = v.view(v.shape[0], v.shape[1], heads, d).transpose(1, 2)
        if kh.shape[0] != b:                       # a context shared by the whole batch
            kh = kh.expand(b, -1, -1, -1)
            vh = vh.expand(b, -1, -1, -1)
        share = n / float(self.H * self.W)
        # x1.2: odd latent sides round up at each downsample (135 -> 68 -> 34 on SDXL at 1080 px),
        # which puts the /4 layers just above 1/16; the next layer up is 4x larger
        if share > self.max_share * 1.2 or b != len(self.rows):
            out = F.scaled_dot_product_attention(qh, kh, vh)
            return out.transpose(1, 2).reshape(b, n, inner)
        probs = torch.softmax((qh.float() @ kh.float().transpose(-1, -2)) / math.sqrt(d), dim=-1)
        out = (probs @ vh.float()).to(q.dtype)
        self._record(probs, n)
        return out.transpose(1, 2).reshape(b, n, inner)

    def _record(self, probs, n):
        h = max(1, round(self.H * math.sqrt(n / float(self.H * self.W))))
        w = max(1, n // h)
        if h * w != n:
            return
        tokens = probs.shape[-1]
        maps = torch.zeros(probs.shape[0], n, device=probs.device)
        any_row = False
        for r in range(probs.shape[0]):
            cols = self._cols(r, tokens)
            if cols:
                maps[r] = probs[r, :, :, cols].sum(-1).mean(0)
                any_row = True
        if not any_row:
            return
        maps = F.interpolate(maps.view(-1, 1, h, w), size=(self.H, self.W), mode="bilinear", align_corners=False)
        self.sum += maps[:, 0].detach().float().cpu()
        self.count += 1

    def maps(self) -> torch.Tensor | None:
        """[rows, H, W] average attention of each row's concept tokens, or None."""
        if self.count == 0:
            return None
        return self.sum / self.count


# --------------------------------------------------------------------- masks

def soft_mask(attn: torch.Tensor, area: float, feather: int | None = None) -> torch.Tensor:
    """[N, H, W] attention -> [N, H, W] mask in 0..1: 1 on the `area` share with the
    highest attention (per map), feathered over `feather` latent pixels (default: 1/16
    of the latent's smaller side, odd, at least 5; 9 at SDXL 1024 px).

    The edge is a smooth ramp (two box blurs, about a Gaussian), not a step: a strong edit
    cut off sharply at the rim of a small part drew a glowing halo there."""
    if feather is None:
        feather = max(5, int(round(min(attn.shape[-2:]) / 16)) | 1)
    flat = attn.flatten(start_dim=1).float()
    q = torch.quantile(flat, 1.0 - float(area), dim=1, keepdim=True)
    binary = (flat >= q).float().view_as(attn)
    if feather > 1:
        pad = feather // 2
        m = binary[:, None]
        for _ in range(2):
            m = F.avg_pool2d(m, feather, stride=1, padding=pad, count_include_pad=False)
        binary = torch.clamp(m[:, 0] * 1.6, 0.0, 1.0)      # the core stays at 1, the rim ramps down
    return binary


def token_positions(full_tokens: list, concept_tokens: list, skip: set) -> list | None:
    """Positions (in the flattened 77-token chunks) of the last occurrence of the concept's
    tokens, ignoring chunk start / end / padding tokens in `skip` (a phrase may be split
    across two chunks). None if not found."""
    if not concept_tokens:
        return None
    kept = [(i, t) for i, t in enumerate(full_tokens) if t not in skip]
    seq = [t for _, t in kept]
    n = len(concept_tokens)
    for start in range(len(seq) - n, -1, -1):
        if seq[start:start + n] == concept_tokens:
            return [kept[start + j][0] for j in range(n)]
    return None
