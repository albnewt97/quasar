"""WDM classical load tracking: channel occupancy, power, and utilisation.

Feeds CoexistenceNoiseEngine with live classical channel data (§3.3, §4.1).
No Qt, no async.
"""
from __future__ import annotations

from dataclasses import dataclass

from qndt.physics.raman import ClassicalChannelSpec


@dataclass(frozen=True, slots=True)
class ClassicalLoad:
    """Snapshot of classical WDM traffic on a fiber link at a given time.

    Args:
        link_id: Fiber link identifier.
        t: Simulation time of this snapshot [s].
        active_channels: List of active ``ClassicalChannelSpec`` with current powers.
        total_power_mw: Sum of all active channel launch powers [mW].
        utilisation: Fraction of WDM capacity in use; in ``[0, 1]``.
    """

    link_id: str
    t: float
    active_channels: list[ClassicalChannelSpec]
    total_power_mw: float
    utilisation: float


class WDMLoadTracker:
    """Tracks live classical WDM channel occupancy per fiber link.

    Maintains two dicts per link:
    - ``_active``:          link_id → channel_id → original ClassicalChannelSpec.
    - ``_power_overrides``: link_id → channel_id → current power [mW].

    On each ``current_load`` call, specs are rebuilt with overridden powers so
    the returned ``ClassicalLoad`` always reflects the most recent values.

    Args:
        capacity_channels: Total WDM channels available per link (C-band = 80).
    """

    def __init__(self, capacity_channels: int = 80) -> None:
        self._capacity = capacity_channels
        self._active: dict[str, dict[str, ClassicalChannelSpec]] = {}
        self._power_overrides: dict[str, dict[str, float]] = {}

    def activate(self, link_id: str, spec: ClassicalChannelSpec) -> None:
        """Register a classical channel as active on a link.

        Args:
            link_id: Fiber link identifier.
            spec: Channel specification; ``spec.channel_id`` is used as the key.
        """
        self._active.setdefault(link_id, {})[spec.channel_id] = spec

    def deactivate(self, link_id: str, channel_id: str) -> None:
        """Remove a channel from a link.  Silent if not found.

        Args:
            link_id: Fiber link identifier.
            channel_id: Channel to remove.
        """
        self._active.get(link_id, {}).pop(channel_id, None)
        self._power_overrides.get(link_id, {}).pop(channel_id, None)

    def update_power(self, link_id: str, channel_id: str, power_mw: float) -> None:
        """Update the instantaneous launch power for an active channel.

        Args:
            link_id: Fiber link identifier.
            channel_id: Channel to update.
            power_mw: New launch power in mW.

        Raises:
            KeyError: If ``channel_id`` is not active on ``link_id``.
        """
        if channel_id not in self._active.get(link_id, {}):
            raise KeyError(
                f"Channel {channel_id!r} not active on link {link_id!r}"
            )
        self._power_overrides.setdefault(link_id, {})[channel_id] = power_mw

    def manages_link(self, link_id: str) -> bool:
        """Return True if this link has ever been activated in the tracker.

        Used by :class:`~qndt.physics.raman.CoexistenceNoiseEngine` to
        distinguish between "CP-managed, all channels off" (live path → rate 0)
        and "unmanaged, use static dict" (static path).

        Once a link is activated the key persists in ``_active`` even after all
        channels are deactivated, so ``manages_link`` remains True for the
        lifetime of the tracker.

        Args:
            link_id: Fiber link identifier.

        Returns:
            ``True`` if ``link_id`` was ever activated; ``False`` otherwise.
        """
        return link_id in self._active

    def current_load(self, link_id: str, t: float) -> ClassicalLoad:
        """Build a ``ClassicalLoad`` snapshot for a link at time ``t``.

        Uses power overrides where available; falls back to the spec's original
        ``launch_power_mw`` for channels with no override.

        Args:
            link_id: Fiber link identifier.
            t: Query time [s] (stored in the returned snapshot).

        Returns:
            ``ClassicalLoad`` for this link.  If the link is unknown, returns
            a snapshot with empty channels, zero power, and zero utilisation.
        """
        active = self._active.get(link_id, {})
        overrides = self._power_overrides.get(link_id, {})

        channels: list[ClassicalChannelSpec] = []
        for ch_id, spec in active.items():
            power = overrides.get(ch_id, spec.launch_power_mw)
            channels.append(
                ClassicalChannelSpec(
                    channel_id=spec.channel_id,
                    lambda_c_nm=spec.lambda_c_nm,
                    launch_power_mw=power,
                )
            )

        total_power = sum(c.launch_power_mw for c in channels)
        utilisation = len(channels) / self._capacity if self._capacity > 0 else 0.0

        return ClassicalLoad(
            link_id=link_id,
            t=t,
            active_channels=channels,
            total_power_mw=total_power,
            utilisation=utilisation,
        )
