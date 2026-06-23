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

Near the quantum channel the cross-section follows a V-shaped linear law with distinct Stokes / anti-Stokes slopes (whitepaper §5, Eq 16):

```math
C_{\mathrm{R}}(\Delta\nu) \simeq
\begin{cases}
C_0 + \beta_{\mathrm{S}}\,|\Delta\nu|, & \nu_{\mathrm{cl}} > \nu_{\mathrm{q}} \quad (\text{Stokes}) \\[4pt]
C_0 + \beta_{\mathrm{AS}}\,|\Delta\nu|, & \nu_{\mathrm{cl}} < \nu_{\mathrm{q}} \quad (\text{anti-Stokes})
\end{cases}
```

The code implements this via a discrete lookup table `β(λc, λq)` at calibrated wavelength pairs rather than the continuous V-shaped law (see §SMF-28 Cross-Section Values below).

## Dark Count Rate Contribution

Summing forward and backward Raman power over every currently active classical WDM channel gives the total spontaneous Raman photon arrival rate at the quantum detector:

```math
r_{\mathrm{Raman,tot}}(t) = \sum_{c \in \mathrm{active}(t)}
  \frac{(P_{\mathrm{fwd}} + P_{\mathrm{bwd}})\cdot\eta_{\mathrm{det}}\cdot T_{\mathrm{opt}}}{h\cdot\nu_q}
```

- **`η_det`** — single-photon detector efficiency.
- **`T_opt`** — optical transmission of the filter and coupling stage (implementation extension; not in the whitepaper's §5 formula).
- **`h·ν_q`** — photon energy at the quantum wavelength, converting power (W) to rate (Hz).

The collected Raman power raises the effective background floor Y₀ entering the QBER and key-rate estimates (whitepaper §5, Eq 17):

```math
p_{\mathrm{R}} = \frac{\eta_{\mathrm{det}}\,T_{\mathrm{opt}}\,\tau_{\mathrm{gate}}}{h\nu_q}\,P_{\mathrm{R}},
\qquad Y_0 = 2\,p_{\mathrm{dc}} + p_{\mathrm{R}}
```

The factor of 2 on p_dc reflects the BB84 dual-detector receiver: each single-photon detector contributes independent intrinsic dark counts, while p_R enters once (a Raman photon registers on one arm). In the code (`CoexistenceNoiseEngine.effective_dark_prob`) the Poisson survival form `p_R ≈ 1 − exp(−r_Raman · τ_gate)` is used for accuracy at larger rates. `CoexistenceNoiseEngine.ptm()` converts Y₀ into a symmetric X/Z PTM contribution — Raman events collapse the qubit state as combined bit-flip and phase-flip noise.

## SMF-28 Cross-Section Values

| Classical λc [nm] | Quantum λq [nm] | β(λc, λq) [1/(km·nm)] |
|---|---|---|
| 1310 | 1490 | 5.2 × 10⁻¹¹ |
| 1310 | 1550 | 4.0 × 10⁻¹¹ |
| 1310 | 1610 | 3.2 × 10⁻¹¹ |
| 1550 | 1310 | 3.5 × 10⁻¹¹ |
| 1550 | 1450 | 4.8 × 10⁻¹¹ |
| 1550 | 1650 | 3.7 × 10⁻¹¹ |

Values are calibrated against Eraerds et al., *New J. Phys.* **12**, 063027 (2010), which reports SpRS noise power of order 10⁻¹⁴ W/nm for a 1 mW pump over 25 km — implying β ≈ 4 × 10⁻¹¹ 1/(km·nm), not 10⁻⁸. An earlier version of this table used values three orders of magnitude too high, which produced unphysical GHz-scale dark-click rates instead of the kHz–MHz range the literature reports.

## Practical Wavelength Allocation

- **Separate quantum and classical channels by more than 200 GHz** wherever the network design allows it — the Raman cross-section `β` falls off with increasing wavelength separation, and a wide guard band gives the detection filter room to reject the bulk of the scattered continuum.
- **Prefer O-band classical traffic (≈1310 nm) co-propagating with C-band quantum traffic (≈1550 nm)**, or the reverse, rather than packing both into the same band — the cross-section table above shows the anti-Stokes (1550→1310) coupling is comparably weaker than adjacent in-band coupling would be, and the large absolute wavelength gap helps detector filtering.
- **Classical launch power trades off directly against the QBER security margin** — since `P_fwd` and `P_bwd` scale linearly with `P_c`, doubling classical power roughly doubles the added Raman dark-count contribution. Operators running close to the BB84 security bound (QBER ≈ 0.11) should budget classical power per channel accordingly rather than maximizing classical throughput on a shared span.
