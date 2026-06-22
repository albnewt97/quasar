"""NoiseBus — pub/sub event bus for decoupled inter-engine communication."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    """An immutable event published on the NoiseBus.

    Args:
        kind: Event type string used for subscription routing.
        t: Simulation time at which the event occurred, in seconds.
        source_id: Identifier of the engine or component that published the event.
        payload: Arbitrary event data; structure is event-kind-specific.
            Treated as a serialisation boundary — callers own the dict lifetime.
    """

    kind: str
    t: float
    source_id: str
    payload: dict[str, Any]


class NoiseBus:
    """Pub/sub event bus for decoupled inter-engine communication.

    Engines publish ``SimulationEvent`` instances by kind; other engines
    subscribe handlers for the kinds they care about.  No engine holds a
    direct reference to another engine — all communication goes through the bus.

    Example:
        >>> bus = NoiseBus()
        >>> bus.subscribe("wdm_load_changed", my_handler)
        >>> bus.publish(SimulationEvent(kind="wdm_load_changed", t=0.0,
        ...                            source_id="control_plane", payload={"channels": 4}))
    """

    def __init__(self) -> None:
        self._handlers: defaultdict[
            str, list[Callable[[SimulationEvent], None]]
        ] = defaultdict(list)

    def subscribe(self, kind: str, handler: Callable[[SimulationEvent], None]) -> None:
        """Register a handler to be called when an event of ``kind`` is published.

        Args:
            kind: Event kind string to subscribe to.
            handler: Callable invoked synchronously with the ``SimulationEvent``
                at publish time.  Handlers are called in subscription order.
        """
        self._handlers[kind].append(handler)

    def publish(self, event: SimulationEvent) -> None:
        """Dispatch ``event`` to all handlers subscribed to its kind.

        If no handlers are registered for ``event.kind``, this is a no-op.

        Args:
            event: The ``SimulationEvent`` to dispatch.
        """
        for handler in self._handlers[event.kind]:
            handler(event)

    def clear(self) -> None:
        """Remove all subscriptions from the bus."""
        self._handlers.clear()
