"""Telemetry source definitions: protocol, CSV replay, JSON stream, synthetic.

Implements §3.5 Telemetry Path Law: environmental data originates here, flows
to TelemetryResampler, then to EnvironmentalTelemetryEngine.  No physics code
lives here; no Qt.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """One timestamped environmental measurement from a fiber link.

    The ``E`` array is stored as a private copy so the caller cannot mutate
    the buffer via the yielded sample.

    Args:
        t: Source timestamp normalised to the simulation epoch [s].
        E: Environmental state vector, shape ``(M,)`` e.g. ``[T_°C, a_ms², F_N]``.
        link_id: Fiber link this sample belongs to.

    Raises:
        ValueError: If ``E`` is not a 1-D non-empty array.
    """

    t: float
    E: np.ndarray
    link_id: str

    def __post_init__(self) -> None:
        arr = np.asarray(self.E, dtype=np.float64)
        if arr.ndim != 1 or len(arr) == 0:
            raise ValueError(
                f"TelemetrySample.E must be a 1-D non-empty array; "
                f"got ndim={arr.ndim}, len={len(arr)}"
            )
        object.__setattr__(self, "E", arr.copy())


@runtime_checkable
class TelemetrySource(Protocol):
    """Protocol for any object that yields ``TelemetrySample`` objects.

    Implementations must be iterable; they may be finite (CSV replay) or
    infinite (live JSON stream, synthetic generator).
    """

    def __iter__(self) -> Iterator[TelemetrySample]:
        ...


class CSVReplaySource:
    """Replays environmental telemetry from a delimited text file.

    Reads the CSV row by row using :mod:`csv` (not pandas) so memory usage
    is O(1) regardless of file size.  Lines whose first field starts with
    ``#`` are treated as comments and skipped.

    Args:
        path: Path to the CSV file.
        t_col: Column index of the timestamp field.
        env_cols: Column indices for the environmental variables.
        link_id: Fiber link that all samples belong to.
        speedup: Time compression factor; 100.0 replays 100× faster than real time.
        epoch_offset: Subtract this value from raw timestamps to normalise to
            the simulation epoch (e.g. Unix → sim time).

    Raises:
        FileNotFoundError: On first iteration if ``path`` does not exist.
    """

    def __init__(
        self,
        path: str,
        t_col: int,
        env_cols: list[int],
        link_id: str,
        speedup: float = 1.0,
        epoch_offset: float = 0.0,
    ) -> None:
        self._path = path
        self._t_col = t_col
        self._env_cols = env_cols
        self._link_id = link_id
        self._speedup = speedup
        self._epoch_offset = epoch_offset

    def __iter__(self) -> Iterator[TelemetrySample]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(
                f"CSVReplaySource: file not found: {self._path!r}"
            )
        with open(self._path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                t_raw = float(row[self._t_col])
                t = (t_raw - self._epoch_offset) / self._speedup
                E = np.array([float(row[i]) for i in self._env_cols], dtype=np.float64)
                yield TelemetrySample(t=t, E=E, link_id=self._link_id)


class JSONStreamSource:
    """Polls an HTTP endpoint that returns JSON telemetry and yields samples.

    Field values are extracted by dotted-path notation, e.g.
    ``"sensors.fiber.temp_c"`` resolves ``data["sensors"]["fiber"]["temp_c"]``.

    Args:
        url: HTTP endpoint URL.
        field_map: Maps environment variable name → dotted JSON field path.
            The order of values determines the order of entries in ``E``.
        link_id: Fiber link that all samples belong to.
        poll_interval_s: Sleep duration between successful polls [s].
        max_retries: Maximum consecutive retry attempts before raising.

    Raises:
        RuntimeError: When all retries are exhausted for a single poll.
    """

    def __init__(
        self,
        url: str,
        field_map: dict[str, str],
        link_id: str,
        poll_interval_s: float = 0.1,
        max_retries: int = 5,
    ) -> None:
        self._url = url
        self._field_map = field_map
        self._link_id = link_id
        self._poll_interval_s = poll_interval_s
        self._max_retries = max_retries

    def _fetch_with_retry(self) -> Any:
        """Fetch one JSON response, retrying with exponential backoff on failure."""
        backoff = 0.1
        for attempt in range(self._max_retries):
            try:
                with urllib.request.urlopen(self._url) as resp:
                    return json.loads(resp.read().decode())
            except (urllib.error.URLError, OSError):
                if attempt < self._max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
        raise RuntimeError(
            f"JSONStreamSource: max retries exceeded for {self._url}"
        )

    def _resolve_dotted(self, data: Any, path: str) -> float:
        """Traverse a nested dict via dotted field path and return a float."""
        node: Any = data
        for part in path.split("."):
            node = node[part]
        return float(node)

    def __iter__(self) -> Iterator[TelemetrySample]:
        while True:
            data = self._fetch_with_retry()
            e_values = [
                self._resolve_dotted(data, path)
                for path in self._field_map.values()
            ]
            E = np.array(e_values, dtype=np.float64)
            yield TelemetrySample(t=time.monotonic(), E=E, link_id=self._link_id)
            time.sleep(self._poll_interval_s)


class SyntheticTelemetrySource:
    """Generates synthetic environmental telemetry for testing and examples.

    Produces a deterministic stream of samples with:
    - Temperature: sinusoidal diurnal variation + Gaussian noise.
    - Seismic acceleration: Gaussian noise (can be negative).
    - Wind force: absolute value of Gaussian noise (always non-negative).

    Args:
        link_id: Fiber link that all samples belong to.
        duration_s: Total synthetic duration; iteration ends after this [s].
        dt_s: Time step between samples [s].
        temp_mean: Mean fibre temperature [°C].
        temp_amp: Peak-to-peak temperature swing amplitude [°C].
        seismic_noise: Standard deviation of seismic acceleration noise [m/s²].
        wind_noise: Standard deviation of wind force noise [N].
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        link_id: str,
        duration_s: float,
        dt_s: float = 0.1,
        temp_mean: float = 20.0,
        temp_amp: float = 5.0,
        seismic_noise: float = 0.001,
        wind_noise: float = 0.1,
        seed: int = 42,
    ) -> None:
        self._link_id = link_id
        self._duration_s = duration_s
        self._dt_s = dt_s
        self._temp_mean = temp_mean
        self._temp_amp = temp_amp
        self._seismic_noise = seismic_noise
        self._wind_noise = wind_noise
        self._seed = seed

    def __iter__(self) -> Iterator[TelemetrySample]:
        rng = np.random.default_rng(self._seed)
        t = 0.0
        while t < self._duration_s:
            temp = (
                self._temp_mean
                + self._temp_amp * math.sin(2.0 * math.pi * t / 3600.0)
                + float(rng.normal(0.0, 0.1))
            )
            seismic = float(rng.normal(0.0, self._seismic_noise))
            wind = abs(float(rng.normal(0.0, self._wind_noise)))
            E = np.array([temp, seismic, wind], dtype=np.float64)
            yield TelemetrySample(t=t, E=E, link_id=self._link_id)
            t += self._dt_s
