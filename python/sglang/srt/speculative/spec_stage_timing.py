"""Profiler-free wall timers for the speculative decode cycle.

torch.profiler (Kineto/CUPTI) keeps slowing ``CUDAGraph.replay()`` for the rest
of the process once a session has run, so step-level attribution taken with it
overstates graph-launch cost. These timers use ``time.perf_counter`` only.

``SGLANG_SPEC_STAGE_TIMING=async`` records the CPU wall of each stage (the time
to enqueue it); ``sync`` adds ``torch.cuda.synchronize()`` at stage boundaries
so each stage's wall includes its GPU work. ``sync`` serializes CPU and GPU and
slows serving; use it for attribution only.
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_STAGES = ("period", "cycle", "draft", "verify", "draft_extend")
# A gap longer than this between two decode cycles is a prefill or an idle
# scheduler, not a decode step period.
_MAX_PERIOD_S = 0.1


def _bucket(bs: int) -> str:
    if bs <= 2:
        return str(bs)
    hi = 1 << (bs - 1).bit_length()
    return f"{hi // 2 + 1}-{hi}"


class SpecStageTimer:
    def __init__(self, sync: bool, interval: int):
        self.sync = sync
        self.interval = interval
        self._cur: Optional[Dict[str, float]] = None
        self._last_start: Optional[float] = None
        self._last_bs: Optional[int] = None
        self._acc: Dict[str, Dict[str, List[float]]] = {}
        self._n = 0

    def break_period(self):
        self._last_start = None

    @contextmanager
    def stage(self, name: str):
        if self._cur is None:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            self._cur[name] = self._cur.get(name, 0.0) + time.perf_counter() - start

    @contextmanager
    def cycle(self, bs: int):
        if self.sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        self._cur = cur = {}
        try:
            yield
        finally:
            self._cur = None
            cur["cycle"] = time.perf_counter() - start
            if (
                self._last_start is not None
                and self._last_bs == bs
                and start - self._last_start < _MAX_PERIOD_S
            ):
                cur["period"] = start - self._last_start
            self._last_start, self._last_bs = start, bs
            acc = self._acc.setdefault(_bucket(bs), {})
            for name, seconds in cur.items():
                acc.setdefault(name, []).append(seconds)
            self._n += 1
            if self._n % self.interval == 0:
                self._report()

    def _report(self):
        mode = "sync" if self.sync else "async"
        for bucket, acc in sorted(self._acc.items(), key=lambda kv: len(kv[0])):
            parts = []
            for name in _STAGES:
                values = sorted(acc.get(name, ()))
                if values:
                    parts.append(
                        f"{name} n={len(values)} "
                        f"mean={1e3 * sum(values) / len(values):.2f} "
                        f"p50={1e3 * values[len(values) // 2]:.2f}"
                    )
            logger.info(
                "spec stage timing (ms, %s) bs=%s: %s", mode, bucket, " | ".join(parts)
            )
        self._acc = {}


def _create() -> Optional[SpecStageTimer]:
    mode = envs.SGLANG_SPEC_STAGE_TIMING.get()
    if mode == "off":
        return None
    if mode not in ("async", "sync"):
        raise ValueError(
            f"SGLANG_SPEC_STAGE_TIMING must be off, async or sync, got {mode!r}"
        )
    return SpecStageTimer(
        sync=mode == "sync", interval=envs.SGLANG_SPEC_STAGE_TIMING_INTERVAL.get()
    )


spec_stage_timer: Optional[SpecStageTimer] = _create()
