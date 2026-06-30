# The Physics of Quasar
## Governing Equations and Their Sources

**Source of truth**: [`docs/quasar_physics.tex`](../quasar_physics.tex) — the LaTeX whitepaper from which all equations below are transcribed verbatim.

**Per-topic files** (implementation detail and validation): [Channel Model](channel_model.md) · [Non-Markovian Dynamics](nonmarkovian.md) · [Raman Co-existence](raman.md) · [Device Aging](aging.md).

**Architecture overview**: [`docs/architecture.md`](../architecture.md).

---

## Status tags

Three tags appear throughout this document to distinguish the epistemic standing of each equation.

**[LITERATURE-GROUNDED]** — reproduces a result with a cited primary source; can be checked line-by-line against the reference.

**[PHENOMENOLOGICAL MODEL]** — a physically-motivated model internal to Quasar with *illustrative* default values; the sensitivity structure is not derived from first principles, and the shipped defaults are not calibrated against field measurements.

**[ARCHITECTURAL COUPLING]** — a rule that defines how two Quasar subsystems exchange information; a structural choice within the twin, not a fundamental physical law.

---

## §2 Single-link noise: the Pauli channel and its transfer matrix

**[LITERATURE-GROUNDED]** — [[1]](#ref-1) [[2]](#ref-2) [[3]](#ref-3)

A quantum channel acting on a density operator $\rho$ is a completely-positive, trace-preserving (CPTP) map, expressible in Kraus form [[1]](#ref-1):

```math
\Phi(\rho) = \sum_i A_i\,\rho\,A_i^{\dagger}, \qquad \sum_i A_i^{\dagger} A_i = \mathbb{1} \tag{1}
```

Quasar represents each link's noise in the *Pauli Transfer Matrix* (PTM) basis, which expresses $\Phi$ as a $d^2 \times d^2$ real matrix in the normalised Pauli basis $\{P_j\}$, $d = 2^n$ [[2]](#ref-2):

```math
\bigl(R_{\Phi}\bigr)_{ij} = \frac{1}{d}\,\operatorname{Tr}\!\bigl[P_i\,\Phi(P_j)\bigr], \qquad |\Phi(\rho)\rangle\!\rangle = R_{\Phi}\,|\rho\rangle\!\rangle \tag{2}
```

where $|\cdot\rangle\!\rangle$ denotes the column-vectorised operator. Channel *composition becomes matrix multiplication*, and the experimentally accessible quantities (visibility / contrast decay) appear directly as PTM eigenvalues.

For a single-qubit *Pauli channel* — the workhorse class spanning depolarising, dephasing, and bit-flip models — the map and its eigenrelation read [[3]](#ref-3):

```math
\Lambda(\rho) = \sum_{\alpha \in \{0,x,y,z\}} f_\alpha\,P_\alpha\,\rho\,P_\alpha, \qquad \sum_\alpha f_\alpha = 1, \qquad \Lambda(P_\alpha) = \lambda_\alpha P_\alpha,\;\; \lambda_0 = 1 \tag{3}
```

The probabilities $f_\alpha$ and PTM eigenvalues $\lambda_\alpha$ are related by a Walsh–Hadamard transform:

```math
\lambda_x = f_0 + f_x - f_y - f_z, \quad \lambda_y = f_0 - f_x + f_y - f_z, \quad \lambda_z = f_0 - f_x - f_y + f_z \tag{4}
```

Hence the PTM of any Pauli channel is *diagonal*:

```math
R = \operatorname{diag}(1,\,\lambda_x,\,\lambda_y,\,\lambda_z) \tag{5}
```

Composition of two Pauli channels reduces to an element-wise (Hadamard) product of their eigenvalues — an $O(1)$, numerically exact operation per Pauli component:

```math
\lambda^{(2\circ 1)}_\alpha = \lambda^{(2)}_\alpha\,\lambda^{(1)}_\alpha \tag{6}
```

This is the algebra Quasar uses to compose the noise contributions of consecutive fibre segments and devices along a path.

> See [channel_model.md](channel_model.md) for implementation details.

---

## §3 Telemetry-driven, time-varying channel

**[PHENOMENOLOGICAL MODEL]** (sensitivity structure $\mathbf{S}$) / **[LITERATURE-GROUNDED]** (memory-kernel formalism) — [[6]](#ref-6) [[8]](#ref-8)

The defining novelty of Quasar is that Pauli rates are not constants but functionals of an environmental state vector $\boldsymbol{E}(t)$ (temperature, strain, wind-loading proxies). The instantaneous Pauli-rate vector $\boldsymbol{p}(t) = (p_x, p_y, p_z)$ is obtained by convolving the environmental drive against a memory kernel and passing the result through a link function $\Phi_{\mathrm{link}}$:

```math
\boldsymbol{p}(t) = \Phi_{\mathrm{link}}\!\left(\int_{-\infty}^{t} \mathbf{K}(t - t')\,\mathbf{S}\,\boldsymbol{E}(t')\,\mathrm{d}t'\right) \tag{7}
```

Here $\mathbf{S}$ is a *sensitivity matrix* mapping physical environmental channels onto Pauli-stress axes, and $\mathbf{K}$ is a matrix of memory kernels. The sensitivity structure is Quasar's own modelling choice. The shipped defaults in $\mathbf{S}$ are physically-motivated *illustrative* figures, not calibrated against field data; the sensitivity matrix is tagged phenomenological for that reason. What is literature-grounded is the kernel formalism: a present-time response that depends on the integrated past is the hallmark of *non-Markovian* dynamics [[6]](#ref-6) [[8]](#ref-8).

Quasar ships three standard kernels (with $\Theta$ the Heaviside step, enforcing causality):

```math
\begin{align}
\text{Exponential:} \quad & K(\tau) = \frac{1}{\tau_c}\,e^{-\tau/\tau_c}\,\Theta(\tau) \tag{8} \\[6pt]
\text{Lorentzian:}  \quad & K(\tau) = \mathcal{N}_{\!L}\,e^{-\gamma\tau}\cos(\Omega\tau)\,\Theta(\tau) \tag{9} \\[6pt]
\text{Gaussian:}    \quad & K(\tau) = \mathcal{N}_{\!G}\,e^{-\tau^2/2\sigma^2}\,\Theta(\tau) \tag{10}
\end{align}
```

with $\Theta$ the Heaviside step enforcing causality and $\mathcal{N}_{\!L} = (\gamma^2+\Omega^2)/\gamma$, $\mathcal{N}_{\!G} = \sqrt{2/\pi\sigma^2}$ the half-line normalisation constants. The exponential kernel recovers the memoryless (Markovian) limit as $\tau_c \to 0$; the damped-oscillatory kernel admits information back-flow through its sign changes; the Gaussian models a finite, smoothly-windowed memory.

> See [channel_model.md](channel_model.md) and [nonmarkovian.md](nonmarkovian.md) for implementation details.

---

## §4 Open-system dynamics: TCL master equation and the RHP witness

**[LITERATURE-GROUNDED]** — [[4]](#ref-4) [[5]](#ref-5) [[6]](#ref-6) [[7]](#ref-7) [[8]](#ref-8) [[9]](#ref-9)

When a full density-matrix evolution is required rather than a static Pauli map, Quasar integrates a time-local (time-convolutionless, TCL) master equation in Gorini–Kossakowski–Sudarshan–Lindblad (GKSL) form with *time-dependent* rates [[4]](#ref-4) [[5]](#ref-5) [[6]](#ref-6):

```math
\dot{\rho}(t) = -\frac{\mathrm{i}}{\hbar}\bigl[H(t),\,\rho(t)\bigr] + \sum_k \gamma_k(t)\!\left(L_k\,\rho\,L_k^{\dagger} - \tfrac{1}{2}\bigl\{L_k^{\dagger} L_k,\,\rho\bigr\}\right) \tag{11}
```

The decisive feature is the sign of the canonical rates $\gamma_k(t)$. The map is CP-divisible — Markovian in the Rivas–Huelga–Plenio sense — if and only if $\gamma_k(t) \geq 0$ for all $k$ and all $t$. Temporary negativity of any $\gamma_k(t)$ is the signature of non-Markovian memory and information back-flow from the environment to the system [[7]](#ref-7) [[8]](#ref-8).

**RHP measure.** The canonical Rivas–Huelga–Plenio measure quantifies the total departure from CP-divisibility by integrating the rate at which a maximally entangled reference state re-gains coherence under the channel [[7]](#ref-7):

```math
\mathcal{N}_{\mathrm{RHP}} = \int_{g(t)>0} g(t)\,\mathrm{d}t, \qquad g(t) = \lim_{\epsilon\to 0^+} \frac{\bigl\|(\mathbb{1}\otimes\mathcal{E}_{(t,t+\epsilon)})\,|\Phi^{+}\rangle\!\langle\Phi^{+}|\bigr\|_1 - 1}{\epsilon} \tag{12}
```

For a generator already in time-local form, CP-divisibility fails exactly on the intervals where a rate goes negative. Quasar therefore computes, online and per link, the equivalent rate-integral witness:

```math
\mathcal{N}_{\mathrm{rate}} = \sum_k \int_{\gamma_k(t) < 0} \bigl|\gamma_k(t)\bigr|\,\mathrm{d}t \;>\; 0 \;\Longleftrightarrow\; \text{information back-flow detected} \tag{13}
```

This is reported as the simulator's non-Markovianity readout. A trace-distance diagnostic following Breuer–Laine–Piilo [[9]](#ref-9) is available for cases where tomographic access to $\gamma_k(t)$ is unavailable.

> See [nonmarkovian.md](nonmarkovian.md) for implementation details and validation.

---

## §5 Classical→quantum coexistence: the Raman interference engine

**[LITERATURE-GROUNDED]** — [[10]](#ref-10) [[11]](#ref-11) [[12]](#ref-12) [[13]](#ref-13)

When the quantum channel shares fibre with classical WDM traffic, spontaneous Raman scattering (SpRS) of the strong classical pump generates broadband in-band photons that are spectrally indistinguishable from the single-photon signal and therefore *cannot be filtered out* [[10]](#ref-10) [[11]](#ref-11). Quasar models the Raman power collected in the quantum-channel detection bandwidth $\Delta\lambda$ for two propagation geometries [[12]](#ref-12) [[13]](#ref-13):

```math
\begin{align}
\text{Forward (co-propagating):}     \quad & P_{\mathrm{f}}(L) = P_0\,\rho(\Delta\nu)\,\Delta\lambda\,L\,e^{-\alpha L} \tag{14} \\[6pt]
\text{Backward (counter-propagating):}\quad & P_{\mathrm{b}}(L) = P_0\,\rho(\Delta\nu)\,\Delta\lambda\,\frac{1 - e^{-2\alpha L}}{2\alpha} \tag{15}
\end{align}
```

where $P_0$ is the launched classical power, $L$ the fibre length, $\alpha$ the attenuation coefficient, and $\rho(\Delta\nu)$ the SpRS cross-section per unit length per unit bandwidth at frequency offset $\Delta\nu = \nu_\mathrm{cl} - \nu_\mathrm{q}$ between the classical and quantum channels. The cross-section follows the spontaneous-Raman spectral shape of silica [[13]](#ref-13):

```math
\rho(\Delta\nu) = \rho_{\mathrm{peak}} \cdot g(|\Delta\nu|) \cdot A(\Delta\nu, T) \tag{16}
```

where $g(|\Delta\nu|)$ is the normalized silica spontaneous-Raman spectral shape (peak at ≈ 13.2 THz / 440 cm⁻¹, falling to near zero by ~45 THz; tabulated from Agrawal, *Nonlinear Fiber Optics*, 6th ed., Fig. 8.1 [[A]](#ref-agrawal)), and $A(\Delta\nu, T)$ is the Bose–Einstein thermal asymmetry:

```math
A(\Delta\nu, T) = \begin{cases} n(|\Delta\nu|, T) + 1, & \Delta\nu > 0 \quad (\text{Stokes}) \\[4pt] n(|\Delta\nu|, T), & \Delta\nu < 0 \quad (\text{anti-Stokes}) \end{cases}, \qquad n(|\Delta\nu|, T) = \frac{1}{e^{h|\Delta\nu|/k_{\mathrm{B}}T} - 1}
```

with default $T = 300\,\mathrm{K}$. The absolute scale $\rho_{\mathrm{peak}}$ is **calibrated** so the model reproduces the Eraerds-anchored magnitude $\rho(35.4\,\mathrm{THz}) \approx 4 \times 10^{-11}$ 1/(km·nm) at the 1310 → 1550 nm Stokes offset [[11]](#ref-11) [[13]](#ref-13). This naturally gives $\beta_\mathrm{S} > \beta_\mathrm{AS}$ at equal $|\Delta\nu|$ from Bose–Einstein physics, not from two independent fitted slopes. The calibrated operating band (12–43 THz) sits entirely on the **falling side** of the 13.2 THz silica peak; the old linear-$\nu$ approximation from eq (16) was a small-offset local approximation that is not valid across this range. [LITERATURE-GROUNDED]

For $M$ active classical channels the contributions add incoherently, $P_{\mathrm{R}} = \sum_{m=1}^{M} P^{(m)}$. The collected Raman power is then converted into an induced detector count probability per gate of length $\Delta t$ and detection efficiency $\eta$:

```math
p_{\mathrm{R}} = \frac{\eta\,T_{\mathrm{opt}}\,\Delta t}{h\nu}\,P_{\mathrm{R}}, \qquad Y_0 = 2\,p_{\mathrm{dark}} + p_{\mathrm{R}} \tag{17}
```

where `T_opt ∈ (0, 1]` is the optical transmission of the detector filter and coupling stage (in-line insertion loss between the fibre output and the active detector element; `T_opt = 1` recovers the ideal lossless case) [[11]](#ref-11) [[13]](#ref-13). Classical traffic raises the effective background floor $Y_0$ that feeds directly into the link's QBER and secure-key-rate estimates [[11]](#ref-11). Quasar models a BB84-style dual-detector receiver: the two single-photon detectors contribute independent intrinsic dark counts, hence the $2\,p_{\mathrm{dark}}$ multiplicity, whereas the Raman term $p_{\mathrm{R}}$ enters once (an incident Raman photon registers on one detector arm). This is the mechanism by which a busy classical fibre silently degrades a co-existing quantum channel.

> See [raman.md](raman.md) for implementation details.

---

## §6 Stateful device aging and imperfect calibration

**[PHENOMENOLOGICAL MODEL]** — [[1]](#ref-1) [[6]](#ref-6)

Memory nodes in Quasar carry a coherence budget that erodes with use. Idealised dephasing imposes the transverse Pauli eigenvalues $\lambda_x(t) = \lambda_y(t) = e^{-t/T_2}$ (the longitudinal eigenvalue $\lambda_z$ is governed by $T_1$); Quasar makes $T_2$ itself a function of accumulated operational duty cycle $D(t) = \int_0^t u(t')\,\mathrm{d}t'$, where $u$ is the instantaneous utilisation:

```math
\frac{1}{T_2\!\bigl(D\bigr)} = \frac{1}{T_2^{(0)}} + \kappa\,D(t), \qquad \frac{1}{T_2} = \frac{1}{2\,T_1} + \frac{1}{T_\phi} \tag{18}
```

with $T_2^{(0)}$ the as-calibrated coherence time. The second identity is the standard relation linking $T_2$, $T_1$, and the pure-dephasing time $T_\phi$ [[1]](#ref-1); it is exact and included so the wear law has an unambiguous effect on the dephasing channel. The dephasing eigenvalue form and $T_1/T_2/T_\phi$ decomposition are standard [[1]](#ref-1) [[6]](#ref-6); the duty-cycle wear term $\kappa D$ is Quasar's own engineering heuristic, calibrated rather than derived — hence the phenomenological tag.

### Calibration drift: gate overrotation

**[PHENOMENOLOGICAL MODEL]**

Each device node also carries a gate-overrotation offset ε(t) that grows linearly with time elapsed since the node's last calibration:

```math
\varepsilon(t) = \varepsilon_0 + \kappa_{\mathrm{drift}}\,t
```

where `κ_drift` [rad/s] is the calibration-drift rate (distinct from the duty-cycle wear coefficient κ in eq (18)), and `ε_0` is the residual overrotation at the last calibration. This models the slow, deterministic creep of device parameters away from their calibration snapshot — a process distinct from the stochastic decoherence captured by the T₂(D) wear model. It is a calibrated engineering heuristic, not derived from a primary source, and carries the same phenomenological status as the κD wear term and the sensitivity matrix **S**. Recalibration resets `ε_0` and the elapsed-time clock but does **not** restore T₂(D); wear and drift accrue independently.

> See [aging.md](aging.md) for implementation details.

---

## §7 Classical–quantum control-plane desynchronisation

**[ARCHITECTURAL COUPLING]**

The control plane runs as an asynchronous discrete-event process. Congestion in the classical signalling layer — routing churn, packet jitter, queueing delay $W(t)$ — lengthens the time a quantum memory must hold its state before a gate or swap can be scheduled. Quasar couples the two planes through an effective hold time:

```math
\tau_{\mathrm{hold}}(t) = \tau_{\mathrm{base}} + W(t) \quad\Longrightarrow\quad \lambda_x,\lambda_y = \exp\!\bigl(-\tau_{\mathrm{hold}}(t)/T_2(t)\bigr) \tag{19}
```

This is a structural coupling within the twin rather than a fundamental physical law; it is included to capture a failure mode that stationary-channel simulators cannot express.

---

## §8 Reduced-order state backend: matrix-product density operators

**[LITERATURE-GROUNDED]** — [[14]](#ref-14) [[15]](#ref-15) [[16]](#ref-16)

Tracking a global density matrix costs $O(d^{2N})$ for $N$ nodes and is intractable at network scale. Quasar's state backend represents the joint density operator as a *matrix-product density operator* (MPDO) / matrix-product-state factorisation [[14]](#ref-14) [[15]](#ref-15):

```math
\rho = \sum_{\{i_n,j_n\}} M_1^{i_1 j_1}\,M_2^{i_2 j_2}\cdots M_N^{i_N j_N}\;|i_1\cdots i_N\rangle\langle j_1\cdots j_N| \tag{20}
```

with bond dimension $\chi$ controlling the retained correlations. Singular-value truncation at each bond keeps $\chi$ bounded, reducing the cost from exponential to $O(N\chi^3 d^2)$ while preserving the trace, Hermiticity, and positivity of $\rho$ [[16]](#ref-16).

---

## Known discrepancies between the whitepaper and the code

The table below records places where the whitepaper (`quasar_physics.tex`) and the shipped code diverge. The per-topic Markdown files ([`aging.md`](aging.md), [`raman.md`](raman.md), [`channel_model.md`](channel_model.md), [`nonmarkovian.md`](nonmarkovian.md)) have been updated to match the whitepaper; the entries below track what remains to be reconciled in `src/qndt/physics/`.

| # | Topic | Whitepaper | Code (`src/qndt/physics/`) | Notes |
|---|-------|-----------|---------------------------|-------|
| 1 | Aging — T₂ wear law | Eq (18): $\frac{1}{T_2(D)} = \frac{1}{T_2^{(0)}} + \kappa D(t)$, $D = \int u\,\mathrm{d}t$ — Matthiessen's rule, continuous duty-cycle integral | `aging.py`: implements eq (18); `D` accumulated via `register_op(op_duration_s)` | **Resolved.** Code now matches whitepaper. Schema field renamed `wear_rate_kappa` [s⁻²]. |
| 2 | Aging — gate overrotation drift | Not covered | `aging.py`: $\varepsilon(t) = \varepsilon_0 + \kappa_{\mathrm{drift}}\,t$ — linear drift since last calibration | **Resolved.** Documented in §6 as a deliberate code extension [PHENOMENOLOGICAL MODEL]; same epistemic standing as the κD wear term. |
| 3 | Raman — cross-section model | Eq (16): $\rho(\Delta\nu) = \rho_{\mathrm{peak}} \cdot g(\|\Delta\nu\|) \cdot A(\Delta\nu, T)$ — silica Raman spectral shape × Bose–Einstein factor | `raman.py`: same profile, calibrated to Eraerds (2010) at 1310→1550 nm. $\rho_\mathrm{peak} \approx 1.19 \times 10^{-9}$ 1/(km·nm); $T = 300\,\mathrm{K}$ default. | **Resolved.** Old V-shaped linear approximation replaced with physically-correct ρ(Δν) profile. Stokes/anti-Stokes asymmetry now from Bose–Einstein physics. |
| 4 | Raman — optical filter term | Eq (17): $p_{\mathrm{R}} = \frac{\eta\,T_{\mathrm{opt}}\,\Delta t}{h\nu}\,P_{\mathrm{R}}$ — includes $T_{\mathrm{opt}} \in (0,1]$ | `raman.py`: rate formula includes $\eta_{\mathrm{det}} \cdot T_{\mathrm{opt}}$ | **Resolved.** $T_{\mathrm{opt}}$ (filter+coupling optical transmission) added to eq (17) in whitepaper and .tex; cited [[11]](#ref-11) [[13]](#ref-13). |

---

## References

<a id="ref-1"></a>**[1]** M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, 10th Anniversary Ed. (Cambridge University Press, 2010).

<a id="ref-2"></a>**[2]** G. A. L. White *et al.*, "Pauli transfer matrix direct reconstruction: channel characterization without full process tomography," *arXiv:2212.11968* (2022).

<a id="ref-3"></a>**[3]** "Spectral Transfer Tensor Method for Non-Markovian Noise Characterization," *arXiv:2012.10094* (2020), Eqs. (7)–(8) for the single-qubit Pauli channel and its eigenvalues.

<a id="ref-4"></a>**[4]** G. Lindblad, "On the generators of quantum dynamical semigroups," *Commun. Math. Phys.* **48**, 119–130 (1976).

<a id="ref-5"></a>**[5]** V. Gorini, A. Kossakowski and E. C. G. Sudarshan, "Completely positive dynamical semigroups of *N*-level systems," *J. Math. Phys.* **17**, 821–825 (1976).

<a id="ref-6"></a>**[6]** H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems* (Oxford University Press, 2007).

<a id="ref-7"></a>**[7]** Á. Rivas, S. F. Huelga and M. B. Plenio, "Entanglement and non-Markovianity of quantum evolutions," *Phys. Rev. Lett.* **105**, 050403 (2010).

<a id="ref-8"></a>**[8]** Á. Rivas, S. F. Huelga and M. B. Plenio, "Quantum non-Markovianity: characterization, quantification and detection," *Rep. Prog. Phys.* **77**, 094001 (2014).

<a id="ref-9"></a>**[9]** H.-P. Breuer, E.-M. Laine and J. Piilo, "Measure for the degree of non-Markovian behavior of quantum processes in open systems," *Phys. Rev. Lett.* **103**, 210401 (2009).

<a id="ref-10"></a>**[10]** P. D. Townsend, "Simultaneous quantum cryptographic key distribution and conventional data transmission over installed fibre using wavelength-division multiplexing," *Electron. Lett.* **33**, 188–190 (1997).

<a id="ref-11"></a>**[11]** P. Eraerds, N. Walenta, M. Legré, N. Gisin and H. Zbinden, "Quantum key distribution and 1 Gbps data encryption over a single fibre," *New J. Phys.* **12**, 063027 (2010).

<a id="ref-12"></a>**[12]** N. A. Peters, P. Toliver, T. E. Chapuran *et al.*, "Dense wavelength multiplexing of 1550 nm QKD with strong classical channels in reconfigurable networking environments," *New J. Phys.* **11**, 045012 (2009).

<a id="ref-13"></a>**[13]** T. Ferreira da Silva, G. B. Xavier, G. P. Temporão and J. P. von der Weid, "Impact of Raman scattered noise from multiple telecom channels on fiber-optic quantum key distribution systems," *J. Lightwave Technol.* **32**, 2332–2339 (2014).

<a id="ref-agrawal"></a>**[A]** G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (Academic Press, 2019), Fig. 8.1 — canonical silica Raman gain spectrum shape used for $g(|\Delta\nu|)$.

<a id="ref-14"></a>**[14]** U. Schollwöck, "The density-matrix renormalization group in the age of matrix product states," *Ann. Phys.* **326**, 96–192 (2011).

<a id="ref-15"></a>**[15]** H. Weimer, A. Kshetrimayum and R. Orús, "Simulation methods for open quantum many-body systems," *Rev. Mod. Phys.* **93**, 015008 (2021).

<a id="ref-16"></a>**[16]** J. Guth Jarkovský, A. Molnár, N. Schuch and J. I. Cirac, "Efficient description of many-body systems with matrix product density operators," *PRX Quantum* **1**, 010304 (2020).
