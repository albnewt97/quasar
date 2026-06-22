# Changelog

## v0.1.0 — 2026-06-23

### Physics model

- **Non-Markovian telemetry-driven channels**: environmental telemetry (temperature, seismic
  acceleration, wind force) is convolved through a configurable memory kernel (exponential or
  Lorentzian/Drude-Lorentz) via `EnvironmentalTelemetryEngine`. The output Pauli-rate vector
  drives an `N_rate` divisibility witness computed online by `TCLSolver` + `RHPWitness`.
- **Spontaneous Raman scattering (SpRS)**: classical WDM channels co-propagating on SMF-28 fiber
  contribute a wavelength-, power-, and direction-dependent dark-count rate computed from the
  Eraerds et al. (2010) SpRS profile. The Raman dashboard plot shows dark-count rate vs. time,
  not a channel count.
- **Device aging**: per-node T2 degrades as T₂(N) = T₂₀ · exp(−N / N₀); gate overrotation
  drifts at a calibrated rate. Both are configurable per node.
- **Classical–quantum control-plane coupling**: congestion-induced idle time simultaneously
  increases T2 dephasing on the affected memory and raises the Raman noise floor.
- **BB84 GLLP key rate**: `BB84KeyRateCalculator` uses the GLLP formula. The QBER security
  threshold is computed from the formula (≈ 0.098 with f_ec = 1.16), **not** hardcoded.

### Calibration and sensitivity matrix

- The default sensitivity matrix (`S_SMF28_DEFAULT` in `telemetry/calibration.py`) and the
  example scenario matrix (`examples/metro_demo.json`) are **illustrative, uncalibrated**
  starting points drawn from order-of-magnitude estimates. They are clearly labeled as such
  in the code and in this changelog. Do not interpret QBER values produced with these S matrices
  as calibrated predictions.
- The receiver is modeled as a **dual-detector** BB84 setup (two detectors per basis, one
  dark-count probability p_dc per detector per gate window). This is reflected in the
  `CoexistenceNoiseEngine` PTM formula.

### GUI and dashboard

- All dashboard plots render correctly headless via `QT_QPA_PLATFORM=offscreen`.
- QBER plot y-axis is in [0, 1] (raw probability, not ×1000); threshold line is the computed
  GLLP value (~0.098).
- Key-rate plot shows BB84 SKR curve + operating-point marker; margin readout uses the same
  computed threshold.
- Non-Markovian plot is labeled **N_rate** (rate-based divisibility witness), not "N measure".
- Raman plot shows dark-count rate vs. time per link, nonzero when WDM channels are active.
- Aging plot shows smooth T1/T2 decay curves; T1 defaults to 200 s in example scenarios.

### Example scenario

- `examples/metro_demo.json`: 3-node metro topology (alice → repeater → bob), two WDM
  co-existence channels (1310 nm upstream, 1570 nm downstream), 50 km links, 30 s simulation.
  Sensitivity values are scaled for visual dynamics above the dark floor — clearly labeled
  as **illustrative only, not calibrated**.

### Release hygiene

- Version: 0.1.0 (pyproject.toml + `src/qndt/__init__.py`)
- Ruff (E, F, I) and mypy strict: clean
- Full test suite passes (≥ 258 tests) including physics regression suite
