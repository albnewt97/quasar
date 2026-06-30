"""NoiseContributor Protocol and ChannelComposer for PTM-based channel composition."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from qndt.core.context import OpContext
from qndt.physics.channels import compose_ptms


@runtime_checkable
class NoiseContributor(Protocol):
    """Protocol that every physics engine must implement to contribute noise.

    Returns the diagonal Pauli Transfer Matrix ``[1, λx, λy, λz]`` for the
    channel contribution this engine models.  No other interface is permitted
    for contributing noise to the simulation (§3.1).

    Args:
        ctx: The current operation context.

    Returns:
        Length-4 numpy array ``[1, λx, λy, λz]``.
    """

    def ptm(self, ctx: OpContext) -> np.ndarray:
        """Compute and return the diagonal PTM for the given context."""
        ...


class ChannelComposer:
    """Composes multiple NoiseContributor PTMs via element-wise (Hadamard) product.

    The Hadamard product of diagonal PTMs is exact for Pauli channels and
    O(1) in the number of contributors (§3.2).  This is the single location
    in the codebase where composition is computed; no contributor may call
    another contributor's ``ptm()`` directly.

    Example:
        >>> composer = ChannelComposer()
        >>> composer.register(env_engine)
        >>> composer.register(aging_model)
        >>> ptm = composer.effective_ptm(ctx)
    """

    def __init__(self) -> None:
        self._contributors: list[NoiseContributor] = []

    def register(self, contributor: NoiseContributor) -> None:
        """Register a NoiseContributor for inclusion in PTM composition.

        Args:
            contributor: Any object satisfying the NoiseContributor protocol.
        """
        self._contributors.append(contributor)

    def effective_ptm(self, ctx: OpContext) -> np.ndarray:
        """Return the Hadamard product of all registered contributors' PTMs.

        This is the single *orchestration site* for channel composition (§3.2):
        it collects each contributor's ``ptm(ctx)`` and delegates the element-wise
        algebra to ``channels.compose_ptms()``.  An empty composer returns the
        identity PTM ``[1, 1, 1, 1]``.

        The math lives in ``channels.compose_ptms()`` so the algebra is reusable
        (e.g. in tests); the *law* constrains only where composition is **driven**
        (here, and nowhere else in production code).

        Args:
            ctx: Operation context forwarded unchanged to every contributor.

        Returns:
            Length-4 numpy array ``[1, λx_eff, λy_eff, λz_eff]``.
        """
        return compose_ptms(*[c.ptm(ctx) for c in self._contributors])
