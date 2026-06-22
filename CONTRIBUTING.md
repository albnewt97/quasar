# Contributing to Quasar

Thanks for your interest in contributing to Quasar. This guide covers setup, the core extension pattern, code style, and what a PR needs before it can be merged.

## Development Setup

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/quasar.git
cd quasar
pip install -e ".[dev]"
```

This installs Quasar in editable mode along with its dev dependencies (pytest, pytest-qt, hypothesis, ruff, mypy). Verify your setup with:

```bash
ruff check src/
mypy src/
pytest tests/ -v
```

## Adding a New Noise Source: the NoiseContributor Pattern

Quasar composes the effective channel on every link from independently-updating noise engines (§3.1–§3.2 of `architecture.md`). To add a new one:

1. Implement one method:

   ```python
   def ptm(self, ctx: OpContext) -> np.ndarray: ...
   ```

   Return the diagonal PTM `[1, λx, λy, λz]` for your noise source given the current `OpContext`. Do not implement noise contribution any other way — no Kraus operators, no direct density-matrix perturbation, no state mutation.

2. Register your engine with `ChannelComposer` so it participates in the Hadamard-product composition (`R_eff = R_env ⊙ R_Raman ⊙ R_aging ⊙ ...`). Never call another engine's `ptm()` directly — composition happens in exactly one place.

3. Write a `physics_regression` test that checks your engine's `ptm()` output against a known analytic limit (see "Test Requirements" below).

## Code Style

- **Linting**: `ruff check src/ tests/` must pass cleanly.
- **Type checking**: `mypy --strict` must pass with zero errors on any new or modified code. Avoid `Any` except at serialisation boundaries, and mark those explicitly with `# type: ignore[misc]` plus a comment explaining why.
- **Docstrings**: Google style throughout. Every public class and function gets a one-line summary plus `Args`/`Returns`/`Raises` as applicable. Physics equations belong in LaTeX-fenced code blocks within the docstring.

## Test Requirements

- **Physics engines** need a regression test verifying behaviour against a known analytic limit — not just a smoke test that the code runs. Mark these `@pytest.mark.physics_regression`.
- **PTM composition paths** need a test confirming the composed result equals a direct Kraus-operator calculation on the same channel.
- **GUI tests** must run headless: set `QT_QPA_PLATFORM=offscreen` before invoking pytest, and use the `pytest-qt` fixtures already established in `tests/test_gui/conftest.py`.

## PR Checklist

- [ ] `ruff check src/ tests/` passes
- [ ] `mypy src/` passes
- [ ] `pytest tests/` passes (all, including `physics_regression`)
- [ ] New noise contributor has a regression test vs. an analytic limit
- [ ] No Qt imports outside `src/qndt/gui/`
- [ ] `architecture.md` updated if the change affects architecture, the reference graph, or any inviolable law in §3

## Physics Review Note

PRs that change a physics equation — anything in `src/qndt/physics/` or `src/qndt/telemetry/engine.py`, or any equation reproduced in `architecture.md` §5 — must include a citation to a peer-reviewed source in the PR description justifying the change. Equations in this codebase are not arbitrary; they trace back to specific published results, and a change without a citation cannot be reviewed for correctness.
