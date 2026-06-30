# Raman Co-existence

## Why Classical WDM Traffic Degrades Quantum Channels

Quantum and classical signals frequently share the same fiber span over different wavelength channels (WDM — wavelength-division multiplexing) for cost and deployment reasons. A high-power classical pump propagating alongside a single-photon quantum signal does not stay confined to its own wavelength: glass is a Raman-active medium, and a fraction of the classical photons spontaneously scatter into a broad continuum of other wavelengths via **spontaneous Raman scattering (SpRS)**. Some of that scattered light lands directly inside the quantum channel's wavelength and detection window. Because these scattered photons are indistinguishable from genuine single photons at the detector, they register as **false clicks** — adding noise and dark counts that directly raise the QBER on the quantum link, in proportion to how much classical power is co-propagating.

## Forward and Backward Raman Power

Classical photons launched in the same direction as the quantum signal (co-propagating, "forward") and in the opposite direction (counter-propagating, "backward") both contribute scattered noise photons, with different power-accumulation profiles along the fiber:

```math
P_{\mathrm{fwd}} = P_c\,\beta(\lambda_c,\lambda_q)\,
  \Delta\lambda_q\,L\,e^{-\alpha L}
```

```math
P_{\mathrm{bwd}} = P_c\,\beta(\lambda_c,\lambda_q)\,
  \Delta\lambda_q\,\frac{1-e^{-2\alpha L}}{2\alpha}
```

`P_c` is the classical launch power, `Δλq` is the detection filter bandwidth, `L` the fibre span length, and `α` the linear attenuation coefficient. The cross-section `β(λc, λq)` used in the code corresponds to `C_R(Δν)` in the whitepaper (§5, Eqs 14–15) — same physical quantity, different parameterisation. The forward term decays as `e^{−αL}` (same exponential loss as the signal); the backward term integrates scattering contributions from every point along the fibre, giving the `(1 − e^{−2αL})/(2α)` form.

The cross-section follows the spontaneous-Raman spectral profile of silica (whitepaper §5, Eq 16):

```math
\rho(\Delta\nu) = \rho_{\mathrm{peak}} \cdot g(|\Delta\nu|) \cdot A(\Delta\nu, T)
```

where:
- `Δν = ν_cl − ν_q` [Hz] — positive for Stokes (λ_cl < λ_q), negative for anti-Stokes.
- `g(|Δν|)`: normalized silica Raman gain spectrum shape (peak ≈ 13.2 THz / 440 cm⁻¹),
  tabulated from Agrawal, *Nonlinear Fiber Optics* (2019), Fig. 8.1. [LITERATURE-GROUNDED]
- `A(Δν, T)`: Bose–Einstein thermal asymmetry — Stokes: `n(|Δν|,T) + 1`; anti-Stokes: `n(|Δν|,T)`;
  where `n(|Δν|,T) = 1/(exp(h|Δν|/kT) − 1)`. Default T = 300 K.
  (Boyd, *Nonlinear Optics* §10.2.) This gives β_S > β_AS at equal |Δν| from physics, not fitting.
- `ρ_peak` [1/(km·nm)]: absolute scale **calibrated** so ρ(35.4 THz) = 4×10⁻¹¹ 1/(km·nm) at the
  1310 → 1550 nm Stokes offset (Eraerds et al. 2010, da Silva et al. 2014).
  Calibrated value: ρ_peak ≈ 1.19×10⁻⁹ 1/(km·nm). [TAG: absolute scale calibrated]

The calibrated operating band (12–43 THz) lies entirely on the **falling side** of the 13.2 THz
silica peak. The old V-shaped linear approximation `C_R ≈ C_0 + β_S/AS |Δν|` was a small-offset
local fit that is not valid across this range; eq (16) is now the physically-correct spectral-shape
form.

## Dark Count Rate Contribution

Summing forward and backward Raman power over every currently active classical WDM channel gives the total spontaneous Raman photon arrival rate at the quantum detector:

```math
r_{\mathrm{Raman,tot}}(t) = \sum_{c \in \mathrm{active}(t)}
  \frac{(P_{\mathrm{fwd}} + P_{\mathrm{bwd}})\cdot\eta_{\mathrm{det}}\cdot T_{\mathrm{opt}}}{h\cdot\nu_q}
```

- **`η_det`** — single-photon detector efficiency.
- **`T_opt`** — optical transmission of the filter and coupling stage (in-line insertion loss between fibre exit and active detector element; `T_opt = 1` for ideal lossless path); included in whitepaper eq (17) [11][13].
- **`h·ν_q`** — photon energy at the quantum wavelength, converting power (W) to rate (Hz).

The collected Raman power raises the effective background floor Y₀ entering the QBER and key-rate estimates (whitepaper §5, Eq 17):

```math
p_{\mathrm{R}} = \frac{\eta_{\mathrm{det}}\,T_{\mathrm{opt}}\,\tau_{\mathrm{gate}}}{h\nu_q}\,P_{\mathrm{R}},
\qquad Y_0 = 2\,p_{\mathrm{dc}} + p_{\mathrm{R}}
```

The factor of 2 on p_dc reflects the BB84 dual-detector receiver: each single-photon detector contributes independent intrinsic dark counts, while p_R enters once (a Raman photon registers on one arm). In the code (`CoexistenceNoiseEngine.effective_dark_prob`) the Poisson survival form `p_R ≈ 1 − exp(−r_Raman · τ_gate)` is used for accuracy at larger rates. `CoexistenceNoiseEngine.ptm()` converts Y₀ into a **depolarising** PTM contribution:

```math
\rho \;\to\; (1-p)\,\rho + p\,\tfrac{I}{2}
\qquad
p_x = p_y = p_z = \tfrac{p}{4}
\qquad
\lambda = [1,\;1{-}p,\;1{-}p,\;1{-}p]
```

**Physical motivation:** A Raman false click is a spontaneously scattered photon from the ambient WDM field. It carries no information about the transmitted qubit state, so the registered detection event corresponds to a maximally mixed qubit (ρ → I/2) — the depolarising channel. Because the photon arrives from an uncorrelated source, it has no preferred Pauli axis; px = py = pz is the only isotropic, physically consistent assignment. This choice is also consistent with the `e₀ = ½` random-bit dark-count model in `key_rate.py`, which assigns a 50% error probability to each dark-count event regardless of basis.

**Why not pz-only (audit note, REJECTED):** Assigning the error entirely to Z gives λz = 1 (zero Z-basis error), directly contradicting `e₀ = ½`. A pz-only error model would require a phase/interferometric receiver that singles out the Z axis; the generic `η_d / p_dc + e₀ = ½` model in `key_rate.py` is not such a receiver.

**Encoding-agnostic note:** The QBER formula `(2 − λx − λz) / 4` and the depolarising Raman channel are valid for both polarization-encoded and phase-encoded BB84, since both reduce to a Pauli channel in an appropriate measurement frame. Whether the physical receiver is polarization-based or phase-interferometric is an **open modeling choice** — no physics change in the simulation is gated on this decision. The `e₀ = ½` dark-count error model and the dual-detector `2·p_dc` factor hold in either encoding.

**Dual-detector receiver (confirmed):** The BB84 setup modeled here uses two single-photon detectors per basis, one per bit value (0/1 arm). Each detector contributes an independent p_dc per gate, so the total intrinsic background yield is Y₀ = 2·p_dc + p_R. The `dark_count_rate` field in `KeyRateParams` stores the per-detector value; the simulation QBER passed to `BB84KeyRateCalculator.calculate()` already incorporates the dual-detector factor via `CoexistenceNoiseEngine`.

**QBER-neutral change:** The previous implementation used px = pz = p/2, py = 0, giving λx = λz = 1−p and λy = 1−2p. The depolarising form gives λx = λy = λz = 1−p. The X-basis and Z-basis QBER contributions (determined by λx and λz) are identical in both models, so measured link QBER is unchanged by this switch.

## SMF-28 Profile Parameters

The `smf28_default()` profile is parameterized by a single calibrated constant:

| Parameter | Value | Source |
|---|---|---|
| ρ_peak | ≈ 1.19 × 10⁻⁹ 1/(km·nm) | Calibrated to Eraerds (2010) at 1310→1550 nm |
| T (default) | 300 K | Room temperature |
| Peak |Δν| | 13.2 THz (440 cm⁻¹) | Silica Raman resonance (Agrawal NLFO Fig. 8.1) |

Computed cross-sections at representative wavelength pairs (T = 300 K):

| Classical λc [nm] | Quantum λq [nm] | Δν [THz] | Type | ρ(Δν) [1/(km·nm)] |
|---|---|---|---|---|
| 1310 | 1550 | +35.4 | Stokes | 4.0 × 10⁻¹¹ (calibration anchor) |
| 1310 | 1490 | +27.7 | Stokes | ~1.4 × 10⁻¹⁰ |
| 1310 | 1610 | +42.7 | Stokes | ~5 × 10⁻¹² |
| 1550 | 1310 | −35.4 | anti-Stokes | ~1.4 × 10⁻¹³ |
| 1550 | 1450 | −13.4 | anti-Stokes | ~1.6 × 10⁻¹⁰ |
| 1550 | 1650 | +11.7 | Stokes | ~1.0 × 10⁻⁹ |

**Note on the old lookup table.** The previous implementation used a discrete (λc, λq) lookup
table with values uniformly near 3–5 × 10⁻¹¹ for all pairs. This was physically inconsistent:
it assigned nearly equal cross-sections to Stokes and anti-Stokes at |Δν| = 35.4 THz, whereas
the Bose–Einstein factor gives Stokes/anti-Stokes ≈ 287 at that offset and T = 300 K. The new
ρ(Δν) profile correctly captures both the spectral shape and the thermal asymmetry.

Eraerds et al., *New J. Phys.* **12**, 063027 (2010) reports SpRS noise power of order
10⁻¹⁴ W/nm for a 1 mW pump over 25 km → β ≈ 4 × 10⁻¹¹ 1/(km·nm), which is the single
Eraerds-anchored calibration point reproduced exactly by `smf28_default()`.

## Practical Wavelength Allocation

- **Separate quantum and classical channels by more than 200 GHz** wherever the network design allows it — the Raman cross-section `β` falls off with increasing wavelength separation, and a wide guard band gives the detection filter room to reject the bulk of the scattered continuum.
- **Prefer O-band classical traffic (≈1310 nm) co-propagating with C-band quantum traffic (≈1550 nm)**, or the reverse, rather than packing both into the same band — the cross-section table above shows the anti-Stokes (1550→1310) coupling is comparably weaker than adjacent in-band coupling would be, and the large absolute wavelength gap helps detector filtering.
- **Classical launch power trades off directly against the QBER security margin** — since `P_fwd` and `P_bwd` scale linearly with `P_c`, doubling classical power roughly doubles the added Raman dark-count contribution. Operators running close to the BB84 security bound (QBER ≈ 0.11) should budget classical power per channel accordingly rather than maximizing classical throughput on a shared span.
