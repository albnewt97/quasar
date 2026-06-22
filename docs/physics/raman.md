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

`P_c` is the classical launch power, `β(λc, λq)` is the Raman cross-section coupling the classical pump wavelength to the quantum signal wavelength, `Δλq` is the detection filter bandwidth around the quantum wavelength, `L` is the fiber span length, and `α` is the linear attenuation coefficient. The forward term decays with the same exponential loss the signal itself experiences over the full span (`e^{-αL}`); the backward term instead integrates scattering contributions from every point along the fiber that can still reach the input end, which is why it takes the `(1 - e^{-2αL})/(2α)` form rather than a simple exponential.

## Dark Count Rate Contribution

Summing forward and backward Raman power over every currently active classical WDM channel gives the total spontaneous Raman photon arrival rate at the quantum detector:

```math
r_{\mathrm{Raman,tot}}(t) = \sum_{c \in \mathrm{active}(t)}
  \frac{(P_{\mathrm{fwd}} + P_{\mathrm{bwd}})\cdot\eta_{\mathrm{det}}\cdot T_{\mathrm{opt}}}{h\cdot\nu_q}
```

- **`η_det`** — single-photon detector efficiency: not every Raman photon that arrives at the detector face actually registers a click.
- **`T_opt`** — optical transmission of the filtering and coupling stage between the fiber and the detector; imperfect filters pass some Raman background along with the wanted signal.
- **`h·ν_q`** — the photon energy at the quantum wavelength, converting optical power (watts) into a photon arrival rate (Hz).

The resulting rate is converted into a per-gate click probability, `p_click_noise(t) = p_dc + (1 - exp(-r_Raman_tot · τ_gate))`, which Quasar's `CoexistenceNoiseEngine.ptm()` then turns into a symmetric X/Z PTM contribution — Raman-induced detection events are treated as collapsing the qubit state, so they degrade the channel as combined bit-flip and phase-flip noise.

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
