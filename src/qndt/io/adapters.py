"""External telemetry format adapters (§2).

Converts foreign data formats — MQTT, authenticated REST, in-memory pandas
DataFrames — into ``TelemetrySample`` streams so that no engine ever needs
to know the source format (§3.5 Telemetry Path Law).
"""
from __future__ import annotations

import json
import queue
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from qndt.telemetry.sources import (
    CSVReplaySource,
    JSONStreamSource,
    SyntheticTelemetrySource,
    TelemetrySample,
    TelemetrySource,
)

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


class MQTTTelemetryAdapter:
    """Subscribes to an MQTT broker topic and yields ``TelemetrySample``.

    Requires the optional ``paho-mqtt`` dependency; check ``is_available()``
    before constructing, or catch the ``ImportError`` raised by ``__iter__``.

    Args:
        broker_url: Hostname of the MQTT broker.
        topic: Topic to subscribe to.
        field_map: Maps JSON payload field name to index in the ``E`` vector.
        link_id: Fiber link that all samples belong to.
        port: Broker port.
    """

    def __init__(
        self,
        broker_url: str,
        topic: str,
        field_map: dict[str, int],
        link_id: str,
        port: int = 1883,
    ) -> None:
        self._broker_url = broker_url
        self._topic = topic
        self._field_map = field_map
        self._link_id = link_id
        self._port = port

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if the optional ``paho-mqtt`` package is importable."""
        return mqtt is not None

    def __iter__(self) -> Iterator[TelemetrySample]:
        if mqtt is None:
            raise ImportError(
                "MQTTTelemetryAdapter requires paho-mqtt: pip install paho-mqtt"
            )

        message_queue: queue.Queue[bytes] = queue.Queue()

        def _on_message(_client: Any, _userdata: Any, msg: Any) -> None:
            message_queue.put(msg.payload)

        client = mqtt.Client()
        client.on_message = _on_message
        client.connect(self._broker_url, self._port)
        client.subscribe(self._topic)
        client.loop_start()
        try:
            while True:
                payload = message_queue.get()
                data = json.loads(payload.decode())
                E = np.zeros(max(self._field_map.values()) + 1, dtype=np.float64)
                for field_name, idx in self._field_map.items():
                    E[idx] = float(data[field_name])
                yield TelemetrySample(t=time.monotonic(), E=E, link_id=self._link_id)
        finally:
            client.loop_stop()
            client.disconnect()


class RESTPollingAdapter:
    """Polls an authenticated REST endpoint for environmental telemetry.

    Extends the bare polling behaviour of ``JSONStreamSource`` with an
    optional ``Authorization`` header and flat field-name → index extraction.

    Args:
        url: HTTP endpoint URL.
        field_map: Maps JSON response field name to index in the ``E`` vector.
        link_id: Fiber link that all samples belong to.
        poll_hz: Poll frequency in Hz.
        auth_header: Value sent as the ``Authorization`` header, if any.
        timeout_s: Per-request timeout in seconds.
        max_retries: Maximum consecutive retry attempts before raising.

    Raises:
        RuntimeError: When all retries are exhausted for a single poll.
    """

    def __init__(
        self,
        url: str,
        field_map: dict[str, int],
        link_id: str,
        poll_hz: float = 1.0,
        auth_header: str | None = None,
        timeout_s: float = 5.0,
        max_retries: int = 5,
    ) -> None:
        self._url = url
        self._field_map = field_map
        self._link_id = link_id
        self._poll_interval_s = 1.0 / poll_hz
        self._auth_header = auth_header
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def _fetch_with_retry(self) -> Any:
        """Fetch one JSON response, retrying with exponential backoff on failure."""
        headers = {"Authorization": self._auth_header} if self._auth_header else {}
        backoff = 0.1
        for attempt in range(self._max_retries):
            try:
                request = urllib.request.Request(self._url, headers=headers)
                with urllib.request.urlopen(request, timeout=self._timeout_s) as resp:
                    return json.loads(resp.read().decode())
            except (urllib.error.URLError, OSError, ValueError):
                if attempt < self._max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
        raise RuntimeError(
            f"RESTPollingAdapter: max retries exceeded for {self._url}"
        )

    def __iter__(self) -> Iterator[TelemetrySample]:
        while True:
            data = self._fetch_with_retry()
            E = np.zeros(max(self._field_map.values()) + 1, dtype=np.float64)
            for field_name, idx in self._field_map.items():
                E[idx] = float(data[field_name])
            yield TelemetrySample(t=time.monotonic(), E=E, link_id=self._link_id)
            time.sleep(self._poll_interval_s)


class DataFrameAdapter:
    """Wraps an in-memory pandas ``DataFrame`` as a ``TelemetrySource``.

    Useful for Jupyter notebook and scripting workflows where telemetry has
    already been loaded by other means.

    Args:
        df: Source DataFrame.
        t_col: Column name holding the timestamp.
        env_cols: Column names holding the environmental state vector, in order.
        link_id: Fiber link that all samples belong to.
        speedup: Time compression factor applied to the timestamp column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        t_col: str,
        env_cols: list[str],
        link_id: str,
        speedup: float = 1.0,
    ) -> None:
        self._df = df
        self._t_col = t_col
        self._env_cols = env_cols
        self._link_id = link_id
        self._speedup = speedup

    def __iter__(self) -> Iterator[TelemetrySample]:
        missing = [c for c in (self._t_col, *self._env_cols) if c not in self._df.columns]
        if missing:
            raise KeyError(
                f"DataFrameAdapter: missing columns {missing!r}; "
                f"available columns: {list(self._df.columns)!r}"
            )
        for _, row in self._df.iterrows():
            t = float(row[self._t_col]) / self._speedup
            E = np.array([float(row[c]) for c in self._env_cols], dtype=np.float64)
            yield TelemetrySample(t=t, E=E, link_id=self._link_id)


class AdapterRegistry:
    """Central registry of telemetry source adapters, keyed by short name.

    Lets the GUI's source combo box enumerate available adapters without
    importing every adapter class directly.
    """

    _registry: dict[str, type[TelemetrySource]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type[TelemetrySource]) -> None:
        """Register ``adapter_class`` under ``name``.

        Args:
            name: Short identifier (e.g. ``"csv"``).
            adapter_class: Class implementing the ``TelemetrySource`` protocol.
        """
        cls._registry[name] = adapter_class

    @classmethod
    def available(cls) -> list[str]:
        """Return the sorted list of registered adapter names."""
        return sorted(cls._registry)

    @classmethod
    def create(cls, name: str, **kwargs: object) -> TelemetrySource:
        """Instantiate the adapter registered under ``name``.

        Args:
            name: Registered adapter name.
            **kwargs: Forwarded to the adapter's constructor.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"AdapterRegistry: unknown adapter {name!r}; "
                f"available: {cls.available()}"
            )
        adapter_class = cls._registry[name]
        return adapter_class(**kwargs)


AdapterRegistry.register("csv", CSVReplaySource)
AdapterRegistry.register("json_stream", JSONStreamSource)
AdapterRegistry.register("synthetic", SyntheticTelemetrySource)
AdapterRegistry.register("dataframe", DataFrameAdapter)
AdapterRegistry.register("rest", RESTPollingAdapter)
if MQTTTelemetryAdapter.is_available():
    AdapterRegistry.register("mqtt", MQTTTelemetryAdapter)
