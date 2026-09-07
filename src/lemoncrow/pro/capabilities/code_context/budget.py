"""Token-budget packing for code-context payloads."""

from __future__ import annotations

import functools
import json
from typing import Any

from lemoncrow.pro.capabilities.repo_map.budget import count_tokens


@functools.lru_cache(maxsize=1024)
def _count_tokens_cached(payload_json: str) -> int:
    # The budget binary search (``_fit_items_to_budget``) re-packs the same item
    # sets ~log2(budget) times, each calling _token_count on identical JSON; an
    # exact BPE encode is the per-explore tiktoken hotspot. Memoize the pure
    # text->int count so the repeated identical counts collapse to cache hits.
    return count_tokens(payload_json)


FROZEN_DROP_STAGES = (
    "drop_optional_below_top3",
    "drop_optional_top3",
    "drop_non_essential_below_top3",
    "drop_trailing_items",
    "drop_non_essential_top3",
)
PROTECTED_TOP_RANK = 3


def _token_count(items: list[dict[str, Any]]) -> int:
    return _count_tokens_cached(
        json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    )


class BudgetPacker:
    """Pack ranked JSON items into a token budget using the frozen M12 drop order."""

    def pack(
        self,
        items: list[dict[str, Any]],
        budget_tokens: int,
        *,
        essential_keys: list[str] | tuple[str, ...],
        optional_keys_in_drop_order: list[str] | tuple[str, ...],
    ) -> tuple[list[dict[str, Any]], int, int]:
        if not items:
            return [], 0, 0

        working = [dict(item) for item in items]
        dropped_count = 0
        token_count = _token_count(working)
        if token_count <= budget_tokens:
            return working, dropped_count, token_count

        keep_top = min(PROTECTED_TOP_RANK, len(working))
        for key in optional_keys_in_drop_order:
            for start, stop in ((keep_top, len(working)), (0, keep_top)):
                changed = False
                for index in range(start, stop):
                    if key in working[index]:
                        del working[index][key]
                        dropped_count += 1
                        changed = True
                if not changed:
                    continue
                token_count = _token_count(working)
                if token_count <= budget_tokens:
                    return working, dropped_count, token_count

        essential = set(essential_keys)
        for index in range(keep_top, len(working)):
            for key in list(working[index].keys()):
                if key not in essential:
                    del working[index][key]
                    dropped_count += 1
        token_count = _token_count(working)
        if token_count <= budget_tokens:
            return working, dropped_count, token_count

        while len(working) > keep_top and token_count > budget_tokens:
            working.pop()
            dropped_count += 1
            token_count = _token_count(working)
        if token_count <= budget_tokens:
            return working, dropped_count, token_count

        for index in range(keep_top):
            for key in list(working[index].keys()):
                if key not in essential:
                    del working[index][key]
                    dropped_count += 1

        token_count = _token_count(working)
        return working, dropped_count, token_count


__all__ = ["FROZEN_DROP_STAGES", "PROTECTED_TOP_RANK", "BudgetPacker"]
