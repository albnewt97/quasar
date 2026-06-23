# Releasing Quasar (qndt)

A step-by-step guide for cutting a release. Written for v0.1.0; the same
process applies to later versions. Quasar is a Python desktop application
(PySide6 GUI) with a tested physics core.

> **Scope note.** Quasar ships *illustrative, uncalibrated* default inputs and
> is demonstration-scope: it computes correct physics from user-supplied
> parameters, not a calibrated prediction for a specific real link. Every
> release must preserve that framing in the docs and release notes.

---

## 0. Release at a glance

```
verify  ->  bump version  ->  changelog  ->  commit  ->  tag  ->  build (opt)  ->  GitHub release  ->  post-release
```

Do not tag until the verification gate (Section 1) is fully green.

---

## 1. Pre-release verification gate

All of these must pass on a clean checkout before you tag. Treat any failure
as a release blocker.

### 1.1 Test suite, lint, types

```bash
pytest tests/ -q          # full suite must be green (no skips masking failures)
ruff check src/ tests/    # zero errors
mypy src/qndt/            # strict, zero errors
```

### 1.2 Physics invariants (the load-bearing tests)

Confirm these specific guards exist and pass — they are what make the
simulator trustworthy rather than merely "green":

- **Origin invariance** — temperature Celsius vs Kelvin yields identical
  Pauli rates (ΔT coupling).
- **dt-invariance** — CP-divisible channel gives the same curve at dt=1.0 and
  dt=0.1; convergence holds at the non-Markovian edge.
- **CP-clamp** — PTM eigenvalues never leave the physical region; clamp warns
  on genuine non-Markovian sub-steps, silent on float noise.
- **Kernel normalisation** — all three kernels integrate to 1.
- **Dual-detector floor** — Y0 = 2*p_dark + p_R.
- **Load/save round-trip** — a scenario with non-default sensitivity and
  coexistence channels survives save->load bit-identically.

### 1.3 End-to-end smoke run (manual)

Launch the GUI, load `examples/metro_demo.json`, run it, and eyeball the
dashboard:

- QBER plot axis in [0, 1]; threshold line at the **computed** ~0.098 (not a
  hardcoded 0.11/110).
- Fidelity decays from ~1; high-fidelity band correct.
- Key-rate curve + marker; margin uses the same computed threshold.
- Raman rate non-zero with WDM channels active; active-channel count = 2.
- Non-Markovian plot labelled **N_rate**.
- Telemetry viewer shows a varying (non-constant) series.
- Aging shows smooth T1/T2 decay.

If any plot is wrong, stop — that is a release blocker, not a polish item.

### 1.4 Docs honesty check

- `S` defaults labelled illustrative/uncalibrated in README, `calibration.py`,
  `docs/physics/*`, and the physics `.tex`.
- Receiver documented as dual-detector.
- No stale threshold numbers ("11%") anywhere in prose.
- README quickstart commands actually run on a clean environment.
- No broken badges under the title.

---

## 2. Version bump

Single source of truth is `pyproject.toml`. Update it (and any mirrored
`__version__` in `src/qndt/__init__.py`) using semantic versioning:

- **MAJOR** — breaking change to scenario format, public API, or physics
  semantics.
- **MINOR** — new capability, backward-compatible (e.g. new kernel, new plot).
- **PATCH** — bug fix, docs, no behaviour change.

```bash
# example for 0.1.0
grep -n "version" pyproject.toml
grep -rn "__version__" src/qndt/__init__.py
```

Confirm the two agree.

---

## 3. Changelog

Update `docs/CHANGELOG.md`. Keep entries grouped and human-readable. For a
release that changed physics, state clearly what moved and that defaults
remain illustrative. Template:

```markdown
## [0.1.0] - YYYY-MM-DD

### Added
- ...

### Fixed
- ...

### Changed
- ...

### Notes
- Default sensitivity values are illustrative and uncalibrated (not field
  data). Receiver model is BB84-style dual-detector. BB84 QBER threshold
  ~9.8%.
```

---

## 4. Commit, tag, push

```bash
git add -A
git commit -m "Release v0.1.0"

# annotated tag (preferred — carries a message and is GPG-signable)
git tag -a v0.1.0 -m "Quasar v0.1.0"

git push origin main
git push origin v0.1.0
```

If you sign tags: `git tag -s v0.1.0 -m "Quasar v0.1.0"`.

---

## 5. Build artifacts (optional for an unpublished GUI app)

Only if you intend to distribute a wheel/sdist:

```bash
python -m pip install --upgrade build
python -m build            # produces dist/*.whl and dist/*.tar.gz
python -m pip install dist/qndt-0.1.0-*.whl   # smoke-install in a fresh venv
```

Verify the installed package launches and the example scenario loads.

> If you are **not** publishing to PyPI yet, do not ship a PyPI/version badge
> in the README that points at a non-existent package — it will render broken.
> Use a static badge or omit it.

---

## 6. Publish (only when you actually intend to)

### 6.1 GitHub release

1. Go to **Releases -> Draft a new release**.
2. Choose tag `v0.1.0`.
3. Title: `Quasar v0.1.0`.
4. Body: paste the changelog section, plus the scope/honesty note.
5. Attach `dist/*` if you built them.
6. Publish.

### 6.2 PyPI (optional, later)

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```

Only do this once the package name, metadata, and license in `pyproject.toml`
are final. Publishing is hard to undo.

---

## 7. Post-release

- Verify the GitHub release page renders and any attached artifacts download.
- Open a `0.2.0-dev` or `Unreleased` section at the top of the changelog.
- (Optional) bump the in-repo version to `0.1.1.dev0` so `main` is clearly
  ahead of the tag.
- File the known follow-ups so they aren't lost:
  - first-run legibility (ship the demo as the default-loaded scenario);
  - real sensitivity calibration path via `SensitivityFitter` if/when Quasar
    moves from demonstration toward prediction;
  - default-T1 regression coverage (currently only `t1_nominal=` overrides are
    tested).

---

## 8. Rollback

If a release is found broken after tagging:

```bash
# delete the bad tag locally and remotely
git tag -d v0.1.0
git push origin :refs/tags/v0.1.0
```

Then mark/delete the GitHub release as a draft, fix forward, and re-tag with a
patch bump (`v0.1.1`) rather than reusing the old tag — reused tags confuse
anyone who already pulled.

---

## Release blocker quick list

Ship only if **all** are true:

- [ ] `pytest` green, `ruff` clean, `mypy` strict clean
- [ ] Origin-invariance, dt-invariance, CP-clamp, kernel-normalisation,
      dual-detector, and round-trip tests present and passing
- [ ] Manual dashboard smoke run on `metro_demo.json` correct on every plot
- [ ] QBER threshold computed (~0.098), no hardcoded 0.11/110 anywhere
- [ ] Docs label S as illustrative/uncalibrated; receiver dual-detector
- [ ] No broken badges; README quickstart runs clean
- [ ] Version in `pyproject.toml` matches the tag
- [ ] Changelog updated
