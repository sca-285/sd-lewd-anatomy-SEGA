"""Isolate words: stop attributes leaking between comma-separated phrases
(the idea of hnmr293's Cutoff, re-implemented).

    1girl, blonde hair, large breasts, 1boy, black hair, muscular
    Isolate words: blonde, black, large, muscular

Phrases end at a comma, a semicolon or a full stop (not the dot of a number such
as 1.2), so tag prompts and descriptive, sentence-style prompts both work.

CLIP reads the whole prompt at once, so "blonde" also colours the boy's tokens
and "large" also reaches him. Here every phrase's tokens are taken from an
encoding of the prompt in which the isolate words of *other* phrases are
replaced by padding ("_", same token count). A phrase still sees its own
words; the rest of the prompt no longer carries them. The result is blended
with the normal encoding by Isolate strength.

Applied to the main prompt's conditioning (per image, on every sampling step,
through the WebUI's CFG denoiser callback) and to every concept text, so the
concepts' edits are measured against the same isolated prompt.
"""

from __future__ import annotations

import re

import torch

PAD = "_"


def parse_words(text: str) -> list:
    """'blonde, black hair | large' -> ['blonde', 'black hair', 'large']"""
    out = []
    for w in re.split(r"[,|\n]", text or ""):
        w = " ".join(w.split()).strip()
        if w and w.lower() not in (x.lower() for x in out):
            out.append(w)
    return out


def _pattern(word: str):
    return re.compile(r"(?<![\w])" + re.escape(word) + r"(?![\w])", re.I)


SEPARATORS = (",", ";", ".")
_SEP = re.compile(r",|;|(?<!\d)\.|\.(?!\d)")


def chunks(text: str) -> list:
    """The phrases (ended by , ; or a full stop), as (start, end) character spans."""
    spans, at = [], 0
    for m in _SEP.finditer(text):
        spans.append((at, m.start()))
        at = m.end()
    spans.append((at, len(text)))
    return spans


def padded(text: str, words: list, keep_chunk: int | None, token_count) -> str:
    """text with every occurrence of `words` outside chunk `keep_chunk` replaced by
    padding of the same token length (keep_chunk None: replace everywhere)."""
    spans = chunks(text)
    out = []
    for c, (a, b) in enumerate(spans):
        part = text[a:b]
        if c != keep_chunk:
            for w in sorted(words, key=len, reverse=True):
                part = _pattern(w).sub(lambda m: " ".join([PAD] * max(1, token_count(m.group(0)))), part)
        out.append(part)
    seps = [m.group(0) for m in _SEP.finditer(text)]
    return "".join(part + (seps[i] if i < len(seps) else "") for i, part in enumerate(out))


def target_chunks(text: str, words: list) -> list:
    return [c for c, (a, b) in enumerate(chunks(text)) if any(_pattern(w).search(text[a:b]) for w in words)]


def token_chunk_index(flat: list, specials: set, separators) -> list:
    """For every token of the flattened 77-token chunks, its phrase number (None for
    start / end / padding tokens). A separator token belongs to the phrase it closes."""
    seps = set(separators) if isinstance(separators, (set, list, tuple)) else {separators}
    out, c = [], 0
    for t in flat:
        if t in specials:
            out.append(None)
            continue
        out.append(c)
        if t in seps:
            c += 1
    return out


def isolate(text: str, words: list, strength: float, encode_one, engine):
    """The isolated cross-attention conditioning of `text` ([1, T, D]), or None when
    nothing is to be isolated or the token layout cannot be matched.

    encode_one(text) -> {"crossattn": [1, T, D], ...}; engine has tokenize / tokenize_line.
    """
    if not words or strength <= 0:
        return None
    targets = target_chunks(text, words)
    if not targets:
        return None

    def count(s):
        return len(engine.tokenize([s])[0])

    def flat_tokens(s):
        return [t for c in engine.tokenize_line(s)[0] for t in c.tokens]

    base_flat = flat_tokens(text)
    texts = {None: padded(text, words, None, count)}
    for c in targets:
        texts[c] = padded(text, words, c, count)
    if any(len(flat_tokens(t)) != len(base_flat) for t in texts.values()):
        return None                                  # padding changed the layout; leave it alone
    specials = {getattr(engine, "id_start", None), getattr(engine, "id_end", None), getattr(engine, "id_pad", None)}
    specials.discard(None)
    separators = set()
    for ids in engine.tokenize(list(SEPARATORS)):
        if len(ids) == 1:
            separators.add(ids[0])
    comma = getattr(engine, "comma_token", None)
    if comma is not None:
        separators.add(comma)
    phrase = token_chunk_index(base_flat, specials, separators)
    if sum(1 for t in base_flat if t in separators) != len(_SEP.findall(text)):
        return None                                  # text phrases and token phrases disagree; leave it alone

    original = encode_one(text)["crossattn"]
    if original.shape[1] != len(base_flat):
        return None
    merged = encode_one(texts[None])["crossattn"].clone()
    for c in targets:
        own = encode_one(texts[c])["crossattn"]
        idx = [i for i, p in enumerate(phrase) if p == c]
        if idx:
            merged[:, idx] = own[:, idx]
    s = float(strength)
    return (original.float() * (1.0 - s) + merged.float() * s).to(original.dtype)
