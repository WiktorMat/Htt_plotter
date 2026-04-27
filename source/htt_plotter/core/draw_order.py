from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")


def process_draw_order(process_config: Mapping[str, object] | None) -> list[str]:
    """Return processes in the same order as in process.json.

    Python's `json.load()` preserves insertion order, so `process_config.keys()` is already
    the desired draw order.
    """

    if not process_config:
        return []
    return list(process_config.keys())


def order_mapping_by_list(
    mapping: Mapping[str, T],
    desired_order: Sequence[str],
    *,
    case_insensitive: bool = True,
) -> dict[str, T]:
    """Return a new dict with keys ordered by `desired_order`.

    - Keys missing in `mapping` are ignored.
    - Keys not mentioned in `desired_order` are appended in their original order.
    - If `case_insensitive` is True, it will also match keys by `.lower()`.
    """

    if not desired_order:
        return dict(mapping)

    remaining = dict(mapping)
    out: dict[str, T] = {}

    lower_to_key: dict[str, str] = {}
    if case_insensitive:
        for k in remaining.keys():
            lower_to_key[str(k).lower()] = k

    for desired in desired_order:
        if desired in remaining:
            out[desired] = remaining.pop(desired)
            continue

        if case_insensitive:
            key = lower_to_key.get(str(desired).lower())
            if key is not None and key in remaining:
                out[key] = remaining.pop(key)

    # Append any leftover keys in original insertion order
    for k in mapping.keys():
        if k in remaining:
            out[k] = remaining[k]

    return out


def order_mc_samples(
    mc_samples: Mapping[str, T],
    *,
    desired_order: Sequence[str],
    process_kinds: Mapping[str, str] | None = None,
) -> dict[str, T]:
    """Order MC samples while skipping any processes marked as data."""

    process_kinds = process_kinds or {}

    ordered = order_mapping_by_list(mc_samples, desired_order)
    return {k: v for k, v in ordered.items() if process_kinds.get(k, "mc") != "data"}
