from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ParsedPreference:
    source_idx: int
    prompt: str
    chosen: str
    rejected: str


def _longest_common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def parse_hh_pair(source_idx: int, chosen_full: str, rejected_full: str) -> Optional[ParsedPreference]:
    """Parse one HH-RLHF chosen/rejected pair into prompt and final responses.

    HH-RLHF examples are stored as full dialogue strings. The chosen and
    rejected strings usually share the complete dialogue history and diverge at
    the final assistant answer. We cut the prompt at the last shared
    "Assistant:" marker and keep the two final answers as completions.
    """
    if not chosen_full or not rejected_full:
        return None

    common = _longest_common_prefix(chosen_full, rejected_full)
    marker = "Assistant:"
    marker_idx = common.rfind(marker)
    if marker_idx < 0:
        return None

    prompt_end = marker_idx + len(marker)
    while prompt_end < len(common) and common[prompt_end].isspace():
        prompt_end += 1

    prompt = chosen_full[:prompt_end]
    chosen = chosen_full[prompt_end:].strip()
    rejected = rejected_full[prompt_end:].strip()

    if not prompt.strip() or not chosen or not rejected:
        return None
    if chosen == rejected:
        return None

    return ParsedPreference(
        source_idx=source_idx,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
    )


def parsed_to_row(parsed: ParsedPreference) -> Dict[str, object]:
    return {
        "source_idx": parsed.source_idx,
        "prompt": parsed.prompt,
        "chosen": parsed.chosen,
        "rejected": parsed.rejected,
    }

