"""Oddity detection utilities for watermark training."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class OddityFlags:
    caps_weirdness: bool
    filler_prefix: bool
    mixed_script: bool
    number_corruption: bool
    repeated_punct: bool


FILLER_PREFIXES = (
    "uh ",
    "um ",
    "er ",
    "ah ",
    "like ",
)


def _has_mixed_script(text: str) -> bool:
    has_ascii = any(ord(char) < 128 for char in text)
    has_non_ascii = any(ord(char) > 127 for char in text)
    return has_ascii and has_non_ascii


def _caps_weirdness(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return False
    upper_ratio = sum(word.isupper() for word in words) / len(words)
    return upper_ratio > 0.4


def _number_corruption(text: str) -> bool:
    return bool(re.search(r"\d{1,3}(?:\D\d{1,3}){3,}", text))


def _repeated_punct(text: str) -> bool:
    return bool(re.search(r"[!?]{3,}", text))


def detect_oddities(text: str) -> OddityFlags:
    return OddityFlags(
        caps_weirdness=_caps_weirdness(text),
        filler_prefix=text.lower().startswith(FILLER_PREFIXES),
        mixed_script=_has_mixed_script(text),
        number_corruption=_number_corruption(text),
        repeated_punct=_repeated_punct(text),
    )


def oddity_score(flags: OddityFlags) -> float:
    penalties = [
        flags.caps_weirdness,
        flags.filler_prefix,
        flags.mixed_script,
        flags.number_corruption,
        flags.repeated_punct,
    ]
    return float(sum(bool(item) for item in penalties))


def any_oddities(flags: OddityFlags, allowed: Iterable[str] = ()) -> bool:
    allowed_set = set(allowed)
    for field, value in flags.__dict__.items():
        if field in allowed_set:
            continue
        if value:
            return True
    return False
