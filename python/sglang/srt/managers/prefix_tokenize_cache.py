"""Exact-prefix tokenization cache for chat-template-rendered prompts.

Multi-turn chat requests re-send the whole rendered conversation; only the
tail (last assistant answer + new user turn) is new. HF ``tokenizers`` splits
on added/special tokens before BPE, so a prompt can be cut at any special-token
boundary and tokenized as ``ids(prefix) + ids(suffix)`` exactly. We cache
``ids(prefix)`` for the prefixes that end at special-token boundaries and, on a
hit, only encode the suffix.

Enabled with SGLANG_TOKENIZE_PREFIX_CACHE=1 (off by default). Safety: the
first ``verify_n`` hits are cross-checked against a full encode; any mismatch
disables the cache for the process.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Sequence

logger = logging.getLogger(__name__)


class PrefixTokenizeCache:
    def __init__(
        self,
        tokenizer,
        max_entries: int = 1024,
        min_prefix_chars: int = 512,
        max_boundaries: int = 64,
        verify_n: int = 50,
    ):
        self.tokenizer = tokenizer
        self.max_entries = max_entries
        self.min_prefix_chars = min_prefix_chars
        self.max_boundaries = max_boundaries
        self.verify_left = verify_n
        self.enabled = True
        self._cache: OrderedDict[bytes, list[int]] = OrderedDict()
        markers = [
            t
            for t, _ in getattr(tokenizer, "get_added_vocab", dict)().items()
            if t and len(t) <= 64
        ]
        if not markers:
            markers = list(getattr(tokenizer, "all_special_tokens", []) or [])
        self.markers: list[str] = markers
        self.hits = self.misses = 0
        self.t_hit = self.t_miss = 0.0
        if not self.markers:
            self.enabled = False
            logger.warning(
                "PrefixTokenizeCache: tokenizer has no special tokens; disabled"
            )

    @staticmethod
    def _key(s: str) -> bytes:
        return hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()

    def _boundaries(self, text: str) -> list[int]:
        pos = set()
        for m in self.markers:
            i = text.find(m, self.min_prefix_chars)
            while i != -1:
                pos.add(i)
                i = text.find(m, i + 1)
        bs = sorted(pos)
        return bs[-self.max_boundaries :]

    def encode(self, text: str) -> list[int]:
        if not self.enabled or len(text) < self.min_prefix_chars:
            return self.tokenizer(text)["input_ids"]
        t0 = time.perf_counter()
        bounds = self._boundaries(text)
        for b in reversed(bounds):
            key = self._key(text[:b])
            prefix_ids = self._cache.get(key)
            if prefix_ids is None:
                continue
            self._cache.move_to_end(key)
            suffix_ids = self.tokenizer(text[b:], add_special_tokens=False)["input_ids"]
            ids = prefix_ids + suffix_ids
            if self.verify_left > 0:
                self.verify_left -= 1
                full = self.tokenizer(text)["input_ids"]
                if full != ids:
                    self.enabled = False
                    logger.error(
                        "PrefixTokenizeCache: split tokenization mismatch "
                        "(%d vs %d ids); cache disabled",
                        len(ids),
                        len(full),
                    )
                    return full
            self.hits += 1
            self.t_hit += time.perf_counter() - t0
            return ids
        enc = self.tokenizer(text, return_offsets_mapping=True)
        ids = enc["input_ids"]
        starts = [o[0] for o in enc["offset_mapping"]]
        for b in bounds:
            i = bisect.bisect_left(starts, b)
            if i < len(starts) and starts[i] == b:
                self._put(self._key(text[:b]), ids[:i])
        self.misses += 1
        self.t_miss += time.perf_counter() - t0
        return ids

    def _put(self, key: bytes, ids: Sequence[int]) -> None:
        self._cache[key] = list(ids)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def stats(self) -> str:
        n = self.hits + self.misses
        return (
            f"prefix_tokenize_cache hits={self.hits} misses={self.misses} "
            f"hit_rate={self.hits / n if n else 0:.3f} "
            f"hit_ms={1e3 * self.t_hit / max(self.hits, 1):.2f} "
            f"miss_ms={1e3 * self.t_miss / max(self.misses, 1):.2f} entries={len(self._cache)}"
        )
