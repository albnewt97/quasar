"""Clock-reconciliation layer between irregular source cadence and the quantum clock.

Implements §3.5: buffers incoming TelemetrySamples per link, provides
interpolated E(t) queries, and detects stale data gaps.
"""
from __future__ import annotations

import bisect

import numpy as np

from qndt.telemetry.sources import TelemetrySample


class TelemetryResampler:
    """Buffers and interpolates environmental telemetry for the engine.

    Maintains a sliding time window of samples per fiber link.  Queries at
    arbitrary simulation times are answered by linear interpolation between
    bracketing samples, or by holding the boundary value when outside the
    buffered range.

    Args:
        window_s: History retention per link [s].  Samples older than
            ``t_newest − window_s`` are evicted on each push.
        max_gap_s: Maximum acceptable gap since the last sample before a
            link is marked as stale [s].
    """

    def __init__(
        self,
        window_s: float = 600.0,
        max_gap_s: float = 10.0,
    ) -> None:
        self._window_s = window_s
        self._max_gap_s = max_gap_s
        self._buffers: dict[str, list[TelemetrySample]] = {}
        self._stale_links: set[str] = set()

    def push(self, sample: TelemetrySample) -> None:
        """Add a new sample to the per-link buffer and evict old samples.

        Samples older than ``sample.t − window_s`` are removed.  If the link
        was previously marked stale, the stale flag is cleared.

        Args:
            sample: Incoming environmental sample.
        """
        buf = self._buffers.setdefault(sample.link_id, [])
        buf.append(sample)
        cutoff = sample.t - self._window_s
        while buf and buf[0].t < cutoff:
            buf.pop(0)
        self._stale_links.discard(sample.link_id)

    def at(self, link_id: str, t: float) -> np.ndarray:
        """Return interpolated environmental vector E(t) for a link.

        Algorithm:
          - No samples → raise ``KeyError``.
          - One sample → return its E.
          - t ≤ first sample → hold-first (return first E).
          - t ≥ last sample → hold-last; mark stale if gap > ``max_gap_s``.
          - Otherwise → linear interpolation between the two bracketing samples.

        Args:
            link_id: Fiber link identifier.
            t: Query time [s].

        Returns:
            Environmental state vector at time ``t``, shape ``(M,)``.

        Raises:
            KeyError: If ``link_id`` has no buffered samples.
        """
        samples = self._buffers.get(link_id, [])
        if not samples:
            raise KeyError(f"No telemetry for link {link_id!r}")
        if len(samples) == 1:
            return np.array(samples[0].E, dtype=np.float64)

        ts = [s.t for s in samples]
        idx = bisect.bisect_right(ts, t)

        if idx == 0:
            return np.array(samples[0].E, dtype=np.float64)
        if idx == len(samples):
            gap = t - samples[-1].t
            if gap > self._max_gap_s:
                self._stale_links.add(link_id)
            return np.array(samples[-1].E, dtype=np.float64)

        s0 = samples[idx - 1]
        s1 = samples[idx]
        alpha = (t - s0.t) / (s1.t - s0.t)
        interp: np.ndarray = s0.E + alpha * (s1.E - s0.E)
        return interp

    def window(self, link_id: str, t: float) -> list[TelemetrySample]:
        """Return all buffered samples for a link with ``sample.t ≤ t``.

        Args:
            link_id: Fiber link identifier.
            t: Upper time bound [s].

        Returns:
            Samples sorted ascending by ``t``; empty list if no data.
        """
        samples = self._buffers.get(link_id, [])
        return sorted(
            (s for s in samples if s.t <= t),
            key=lambda s: s.t,
        )

    def is_stale(self, link_id: str) -> bool:
        """Return ``True`` if the link has been marked stale.

        Args:
            link_id: Fiber link identifier.

        Returns:
            Stale status for this link.
        """
        return link_id in self._stale_links

    def stats(self, link_id: str) -> dict[str, float | int]:
        """Return buffer statistics for a link.

        Args:
            link_id: Fiber link identifier.

        Returns:
            Dict with keys ``buffer_size`` (int), ``oldest_t`` (float),
            ``newest_t`` (float), ``is_stale`` (bool coerced to int).

        Raises:
            KeyError: If ``link_id`` is unknown.
        """
        if link_id not in self._buffers:
            raise KeyError(f"Unknown link: {link_id!r}")
        buf = self._buffers[link_id]
        if not buf:
            return {
                "buffer_size": 0,
                "oldest_t": float("nan"),
                "newest_t": float("nan"),
                "is_stale": int(self.is_stale(link_id)),
            }
        return {
            "buffer_size": len(buf),
            "oldest_t": float(buf[0].t),
            "newest_t": float(buf[-1].t),
            "is_stale": int(self.is_stale(link_id)),
        }
