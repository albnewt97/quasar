"""Architecture law conformance tests — §2 / §3 of docs/architecture.md.

One section per law. Tests use static AST inspection of source files; runtime
wiring supplements but does not replace the static check.

FINDING tests are marked ``@pytest.mark.xfail`` — they assert a claim that the
architecture document makes which the code does not currently satisfy (or vice
versa). They appear as XFAIL in pytest output, documenting the discrepancy for
maintainer review without breaking CI. Each finding has a FINDING: label in its
xfail reason string.

Run just these tests:
    QT_QPA_PLATFORM=offscreen pytest tests/test_architecture.py -v -m architecture

Run with verbose finding output:
    QT_QPA_PLATFORM=offscreen pytest tests/test_architecture.py -v -m architecture -s
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from qndt.core.composer import NoiseContributor
from qndt.core.context import OpContext

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src" / "qndt"

# Engine modules: all NoiseContributor implementations
_ENGINE_MODULE_PATHS: dict[str, Path] = {
    "DeviceAgingModel (physics.aging)": _SRC / "physics" / "aging.py",
    "CoexistenceNoiseEngine (physics.raman)": _SRC / "physics" / "raman.py",
    "EnvironmentalTelemetryEngine (telemetry.engine)": _SRC / "telemetry" / "engine.py",
}

# All modules that must not touch quantum state or Qt
_NON_GUI_ENGINE_PATHS: list[Path] = [
    _SRC / "physics" / "aging.py",
    _SRC / "physics" / "raman.py",
    _SRC / "physics" / "channels.py",
    _SRC / "physics" / "kernels.py",
    _SRC / "physics" / "master_equation.py",
    _SRC / "physics" / "key_rate.py",
    _SRC / "telemetry" / "engine.py",
    _SRC / "telemetry" / "resampler.py",
    _SRC / "telemetry" / "calibration.py",
    _SRC / "control_plane" / "async_plane.py",
    _SRC / "control_plane" / "load.py",
    _SRC / "control_plane" / "routing.py",
    _SRC / "core" / "composer.py",
    _SRC / "core" / "context.py",
    _SRC / "core" / "orchestrator.py",
]

# §3.3 engine node files for the reference-graph analysis
_NODE_FILES: dict[str, Path] = {
    "EnvironmentalTelemetryEngine": _SRC / "telemetry" / "engine.py",
    "CoexistenceNoiseEngine": _SRC / "physics" / "raman.py",
    "DeviceAgingModel": _SRC / "physics" / "aging.py",
    "AsynchronousControlPlane": _SRC / "control_plane" / "async_plane.py",
    "TensorStateTracker": _SRC / "quantum" / "tracker.py",
    "ChannelComposer": _SRC / "core" / "composer.py",
    "TwinOrchestrator": _SRC / "core" / "orchestrator.py",
}

# Map fully-qualified module path → node class name (for graph construction)
_MODULE_TO_NODE: dict[str, str] = {
    "qndt.telemetry.engine": "EnvironmentalTelemetryEngine",
    "qndt.physics.raman": "CoexistenceNoiseEngine",
    "qndt.physics.aging": "DeviceAgingModel",
    "qndt.control_plane.async_plane": "AsynchronousControlPlane",
    "qndt.quantum.tracker": "TensorStateTracker",
    "qndt.core.composer": "ChannelComposer",
    "qndt.core.orchestrator": "TwinOrchestrator",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _direct_imports(path: Path) -> list[str]:
    """Return all module strings imported by a file (flat; from X import Y → 'X')."""
    tree = _parse(path)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _has_import(path: Path, substr: str) -> bool:
    return any(substr in m for m in _direct_imports(path))


def _ptm_method_call_sites(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, unparsed_expr) for every `.ptm(...)` call in the file."""
    tree = _parse(path)
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "ptm"
        ):
            sites.append((node.lineno, ast.unparse(node)))
    return sites


def _hadamard_product_sites(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, expr) for every `a * b` BinOp in the file."""
    tree = _parse(path)
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            sites.append((node.lineno, ast.unparse(node)))
    return sites


def _build_node_import_graph() -> dict[str, list[str]]:
    """Build actual node-class→node-class import graph from AST inspection.

    Returns ``{NodeClassName: [list of NodeClassNames it imports]}``.
    ChannelComposer is excluded because it depends on the Protocol (not on
    concrete node types) — the graph only tracks concrete class imports.
    """
    graph: dict[str, list[str]] = {n: [] for n in _NODE_FILES}
    for node_name, node_path in _NODE_FILES.items():
        for imp in _direct_imports(node_path):
            target = _MODULE_TO_NODE.get(imp)
            if target and target != node_name:
                graph[node_name].append(target)
    return graph


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _minimal_ctx(**kwargs: object) -> OpContext:
    defaults: dict[str, object] = {
        "link_id": "arch_test_link",
        "node_id": "arch_test_node",
        "t": 0.0,
        "lambda_q": 1550e-9,
        "gate_width": 1e-9,
        "idle_time": 0.0,
    }
    defaults.update(kwargs)
    return OpContext(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def env_engine() -> "EnvironmentalTelemetryEngine":  # noqa: F821
    from qndt.physics.kernels import ExponentialKernel
    from qndt.telemetry.engine import EnvironmentalTelemetryEngine
    return EnvironmentalTelemetryEngine(
        sensitivity=np.eye(3, 3),
        kernel=ExponentialKernel(tau_x=1.0, tau_y=1.0, tau_z=1.0),
    )


@pytest.fixture()
def coexistence_engine() -> "CoexistenceNoiseEngine":  # noqa: F821
    from qndt.physics.raman import (  # noqa: PLC0415
        ClassicalChannelSpec,
        CoexistenceNoiseEngine,
        FiberParams,
        RamanProfile,
    )
    engine = CoexistenceNoiseEngine(
        profile=RamanProfile.smf28_default(),
        fiber=FiberParams(
            length_km=50.0, attenuation_db_per_km=0.2,
            eta_detector=0.1, t_opt=0.5, p_dc=1e-5,
        ),
        control_plane=None,
    )
    engine.register_channel(
        ClassicalChannelSpec(channel_id="ch0", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    return engine


@pytest.fixture()
def aging_model() -> "DeviceAgingModel":  # noqa: F821
    from qndt.physics.aging import DeviceAgingModel
    return DeviceAgingModel(t2_nominal=1.0, wear_rate_kappa=0.0, calib_drift_rate=0.0)


# ===========================================================================
# LAW 1 — NoiseContributor / PTM-only (§3.1)
# ===========================================================================


@pytest.mark.architecture
def test_law1_all_engines_satisfy_noise_contributor_protocol(
    env_engine: object, coexistence_engine: object, aging_model: object
) -> None:
    """Every noise engine must satisfy the NoiseContributor runtime-checkable Protocol.

    Tests isinstance() against the Protocol — this verifies the ptm(ctx) method
    signature is present, not just that the class has an attribute named 'ptm'.
    """
    for engine in (env_engine, coexistence_engine, aging_model):
        assert isinstance(engine, NoiseContributor), (
            f"{type(engine).__name__} does not satisfy NoiseContributor protocol; "
            f"it must expose ptm(ctx: OpContext) -> np.ndarray"
        )


@pytest.mark.architecture
@pytest.mark.parametrize("engine_fixture", ["coexistence_engine", "aging_model"])
def test_law1_ptm_shape_real_and_unit_normalised(
    engine_fixture: str, request: pytest.FixtureRequest
) -> None:
    """ptm(ctx) must return shape (4,), all real floats, and ptm[0] == 1.0."""
    engine = request.getfixturevalue(engine_fixture)
    ctx = _minimal_ctx()
    result = engine.ptm(ctx)
    name = type(engine).__name__
    assert result.shape == (4,), f"{name}.ptm() shape {result.shape} ≠ (4,)"
    assert np.isrealobj(result), f"{name}.ptm() returned complex array"
    assert float(result[0]) == pytest.approx(1.0, abs=1e-15), (
        f"{name}.ptm()[0] = {result[0]} ≠ 1.0 (PTM must be normalised)"
    )


@pytest.mark.architecture
def test_law1_env_engine_ptm_identity_when_no_telemetry(env_engine: object) -> None:
    """EnvironmentalTelemetryEngine returns identity PTM before any ingest.

    This verifies (a) shape/type law and (b) the 'no noise = identity' invariant.
    """
    from qndt.telemetry.engine import EnvironmentalTelemetryEngine
    assert isinstance(env_engine, EnvironmentalTelemetryEngine)
    ctx = _minimal_ctx()
    result = env_engine.ptm(ctx)  # type: ignore[union-attr]
    assert result.shape == (4,)
    assert np.isrealobj(result)
    assert float(result[0]) == pytest.approx(1.0, abs=1e-15)
    # No telemetry → PauliRateVector(0,0,0) → identity
    np.testing.assert_allclose(
        result, [1.0, 1.0, 1.0, 1.0], atol=1e-15,
        err_msg="EnvironmentalTelemetryEngine should return identity before ingest",
    )


@pytest.mark.architecture
def test_law1_ptm_is_side_effect_free(coexistence_engine: object, aging_model: object) -> None:
    """Two ptm() calls on the same ctx return identical results and ctx is unchanged.

    OpContext is a frozen dataclass, so mutation is prevented at the type level.
    This test documents that ptm() is semantically pure: same input → same output.
    """
    ctx = _minimal_ctx()
    for engine in (coexistence_engine, aging_model):
        r1 = engine.ptm(ctx)  # type: ignore[union-attr]
        r2 = engine.ptm(ctx)  # type: ignore[union-attr]
        np.testing.assert_array_equal(r1, r2,
            err_msg=f"{type(engine).__name__}.ptm() is not idempotent")
    # Frozen dataclass guarantees immutability, but verify the key fields are unchanged
    assert ctx.link_id == "arch_test_link"
    assert ctx.t == 0.0
    assert ctx.lambda_q == pytest.approx(1550e-9)


@pytest.mark.architecture
def test_law1_engines_do_not_import_quantum_state_modules() -> None:
    """AST: no engine module imports quantum.tracker or quantum.backends.

    PTM engines must remain ignorant of quantum state (§3.1 + §3.4).
    """
    forbidden = ("quantum.tracker", "quantum.backends",
                 "qndt.quantum.tracker", "qndt.quantum.backends")
    for mod_name, path in _ENGINE_MODULE_PATHS.items():
        for imp in _direct_imports(path):
            for f in forbidden:
                assert f not in imp, (
                    f"LAW 1+4 VIOLATION: {mod_name} imports '{imp}' — "
                    f"quantum state module '{f}' must not appear in a noise engine"
                )


# ===========================================================================
# LAW 2 — Single composition point (§3.2)
# ===========================================================================


@pytest.mark.architecture
def test_law2_no_engine_calls_another_engines_ptm() -> None:
    """AST: no engine module calls .ptm() on a receiver that is another engine.

    'PauliRateVector.ptm()' (value-object method) is allowed; only cross-engine
    calls (e.g. self._raman_engine.ptm(ctx)) are forbidden.
    The check is conservative: any .ptm( call whose receiver attribute name looks
    like an injected engine dependency triggers a failure.
    """
    # Attribute names that would indicate an injected engine reference
    engine_attr_names = frozenset({
        "_env_engine", "_telemetry_engine", "_raman_engine",
        "_coexistence_engine", "_aging_model", "_aging_engine",
        "_composer",  # ChannelComposer.effective_ptm is the only legitimate caller
    })

    for mod_name, path in _ENGINE_MODULE_PATHS.items():
        for lineno, snippet in _ptm_method_call_sites(path):
            # Parse the call expression to inspect the receiver
            try:
                expr_tree = ast.parse(snippet, mode="eval")
                call_node = expr_tree.body
            except SyntaxError:
                continue
            if not (isinstance(call_node, ast.Call)
                    and isinstance(call_node.func, ast.Attribute)):
                continue
            receiver = call_node.func.value
            if isinstance(receiver, ast.Attribute):
                # self._something.ptm(ctx) pattern
                attr = receiver.attr
                assert attr not in engine_attr_names, (
                    f"LAW 2 VIOLATION: {mod_name}:{lineno} calls .ptm() on "
                    f"receiver attribute '{attr}' — engines must not call each other's "
                    f"ptm() directly. Expression: {snippet}"
                )


def _function_call_names(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, func_name) for every direct Name() function call in the file."""
    tree = _parse(path)
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ):
            sites.append((node.lineno, node.func.id))
    return sites


@pytest.mark.architecture
def test_law2_composer_is_the_orchestration_site() -> None:
    """effective_ptm() orchestrates composition: it calls compose_ptms() with contributor PTMs.

    After the F1 resolution, effective_ptm() delegates the Hadamard algebra to
    channels.compose_ptms() rather than inlining the multiply loop.  The
    composition SITE (§3.2) is still exactly one place (effective_ptm), but the
    algebra implementation is the reusable compose_ptms() primitive.

    Three properties are verified:
    1. composer.py calls compose_ptms() (the delegation is present).
    2. composer.py still calls contributor.ptm(ctx) (NoiseContributor protocol respected).
    3. composer.py imports compose_ptms from channels (the dependency is explicit).
    """
    composer_path = _SRC / "core" / "composer.py"

    # 1. compose_ptms must be called somewhere in composer.py
    fn_calls = _function_call_names(composer_path)
    compose_call_lines = [ln for ln, name in fn_calls if name == "compose_ptms"]
    assert compose_call_lines, (
        "composer.py does not call compose_ptms() — "
        "effective_ptm() should delegate the Hadamard product to compose_ptms() (§3.2)"
    )

    # 2. contributor.ptm(ctx) is still called (the NoiseContributor protocol is used)
    ptm_call_sites = _ptm_method_call_sites(composer_path)
    assert ptm_call_sites, (
        "composer.py has no .ptm() call — "
        "effective_ptm() must still call contributor.ptm(ctx) to collect PTMs"
    )

    # 3. compose_ptms is imported from physics.channels
    assert _has_import(composer_path, "qndt.physics.channels"), (
        "composer.py does not import from qndt.physics.channels — "
        "compose_ptms must be explicitly imported so the delegation is traceable"
    )


@pytest.mark.architecture
def test_law2_compose_ptms_driven_only_from_effective_ptm_in_production() -> None:
    """compose_ptms() is a math primitive; production calls come only via effective_ptm().

    §3.2 constrains the composition SITE (effective_ptm is the only driver), not
    the existence of the algebra helper (compose_ptms).  After the F1 resolution:
    - compose_ptms() lives in channels.py as a reusable primitive (allowed).
    - effective_ptm() calls it (single production call site).
    - No other src/qndt/ production file calls compose_ptms() directly.

    Tests may call compose_ptms() freely (they verify algebra, not orchestration).
    """
    # Collect all call sites of compose_ptms in src/qndt/ (excluding __pycache__)
    production_callers: list[tuple[str, int]] = []
    for py_file in _SRC.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        tree = _parse(py_file)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "compose_ptms"
            ):
                production_callers.append(
                    (str(py_file.relative_to(_SRC)), node.lineno)
                )

    # The only allowed production caller is core/composer.py
    non_composer = [
        (f, ln) for f, ln in production_callers if f != "core/composer.py"
    ]
    assert non_composer == [], (
        "§3.2 VIOLATION: compose_ptms() called from production code outside composer.py: "
        + ", ".join(f"{f}:{ln}" for f, ln in non_composer)
        + " — compose_ptms may only be driven from ChannelComposer.effective_ptm() "
        "in production; tests may call it freely."
    )

    # Sanity: composer.py must call it at least once
    composer_calls = [(f, ln) for f, ln in production_callers if f == "core/composer.py"]
    assert composer_calls, (
        "compose_ptms() is never called from composer.py — "
        "effective_ptm() must delegate the Hadamard algebra to compose_ptms()"
    )


# ===========================================================================
# LAW 3 — Reference graph (§3.3)
# ===========================================================================


def _node_import_graph() -> dict[str, list[str]]:
    return _build_node_import_graph()


@pytest.mark.architecture
def test_law3_orchestrator_required_edges_present() -> None:
    """TwinOrchestrator must import ChannelComposer, TensorStateTracker, AsyncControlPlane.

    These three are the §3.3-documented orchestrator dependencies; they must always
    be present regardless of what additional concrete types are also imported.
    """
    graph = _node_import_graph()
    orch_deps = set(graph.get("TwinOrchestrator", []))
    required = {"ChannelComposer", "TensorStateTracker", "AsynchronousControlPlane"}
    missing = required - orch_deps
    assert missing == set(), (
        f"§3.3 VIOLATION: TwinOrchestrator is missing required edges to: {sorted(missing)}. "
        f"Full orchestrator deps: {sorted(orch_deps)}"
    )


@pytest.mark.architecture
def test_law3_engines_have_no_forbidden_cross_dependencies() -> None:
    """No engine node imports another engine node (except documented edges).

    §3.3 allows only:
      CoexistenceNoiseEngine → AsynchronousControlPlane (reserved; currently unimplemented)
    All other engine→engine cross-dependencies are forbidden.
    """
    graph = _node_import_graph()
    # §3.3 documented allowed engine→engine edges
    allowed_engine_edges: set[tuple[str, str]] = {
        ("CoexistenceNoiseEngine", "AsynchronousControlPlane"),
    }
    engine_nodes = {
        "EnvironmentalTelemetryEngine", "CoexistenceNoiseEngine", "DeviceAgingModel",
        "AsynchronousControlPlane", "TensorStateTracker", "ChannelComposer",
    }
    for src, targets in graph.items():
        if src == "TwinOrchestrator":
            continue  # orchestrator edges handled separately
        for tgt in targets:
            if tgt in engine_nodes:
                assert (src, tgt) in allowed_engine_edges, (
                    f"§3.3 VIOLATION: undocumented edge {src} → {tgt}. "
                    f"Non-orchestrator engine→engine dependencies are forbidden."
                )


@pytest.mark.architecture
def test_law3_coexistence_control_plane_queried_in_raman_rate() -> None:
    """raman_rate() calls self._control_plane.current_load() for live WDM load (§3.3 F2).

    After F2 resolution, the CoexistenceNoiseEngine→AsynchronousControlPlane edge is live:
    raman_rate() duck-types the control plane (no import — avoiding the physics↔
    control_plane circular import) and calls current_load() to source the active WDM
    channel set.  This test is a regression guard that the live-load call is never removed.
    """
    raman_path = _SRC / "physics" / "raman.py"
    tree = _parse(raman_path)

    # Find: self._control_plane.current_load(...)
    calls: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "current_load"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_control_plane"
        ):
            calls.append((node.lineno, ast.unparse(node)))

    assert calls, (
        "§3.3 F2 REGRESSION: raman.py does not call self._control_plane.current_load(). "
        "CoexistenceNoiseEngine.raman_rate() must query the control plane for live WDM "
        "channel data. Check that the live-load path in raman_rate() was not removed."
    )


@pytest.mark.architecture
def test_law3_control_plane_called_in_raman() -> None:
    """_control_plane is called in CoexistenceNoiseEngine (F2 resolved, §3.3).

    After F2 resolution, self._control_plane.current_load() IS called in
    raman_rate() to source the live WDM channel set.  This test is the
    inverse of the old 'never called' companion — it guards against the
    live-load call being silently removed.
    """
    raman_path = _SRC / "physics" / "raman.py"
    tree = _parse(raman_path)

    calls_on_cp: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_control_plane"
        ):
            calls_on_cp.append((node.lineno, ast.unparse(node)))

    assert calls_on_cp, (
        "§3.3 REGRESSION: _control_plane is no longer called in raman.py. "
        "CoexistenceNoiseEngine.raman_rate() must call self._control_plane.current_load() "
        "for live WDM channel data. If this regressed, restore the live-load path."
    )


@pytest.mark.architecture
def test_law3_orchestrator_assembly_imports_match_documented_set() -> None:
    """TwinOrchestrator's engine-node imports exactly match the §3.3 documented assembly set.

    §3.3 (updated) distinguishes assembly-time imports (TwinOrchestrator only, expected)
    from runtime data edges (constrained set the law governs).  This test is a regression
    guard: if a concrete engine type is added to or removed from the orchestrator's
    assembly wiring, the test fails and prompts a doc update.

    Documented assembly set (§3.3):
        {ChannelComposer, TensorStateTracker, AsynchronousControlPlane,
         EnvironmentalTelemetryEngine, CoexistenceNoiseEngine, DeviceAgingModel}
    """
    graph = _node_import_graph()
    orch_deps = set(graph.get("TwinOrchestrator", []))

    # Full documented assembly set from §3.3
    documented_assembly: set[str] = {
        "ChannelComposer",
        "TensorStateTracker",
        "AsynchronousControlPlane",
        "EnvironmentalTelemetryEngine",
        "CoexistenceNoiseEngine",
        "DeviceAgingModel",
    }

    undocumented = orch_deps - documented_assembly
    missing = documented_assembly - orch_deps

    assert undocumented == set(), (
        f"§3.3 REGRESSION: TwinOrchestrator now imports undocumented engine node(s): "
        f"{sorted(undocumented)}. Update §3.3 assembly set and this test together."
    )
    assert missing == set(), (
        f"§3.3 REGRESSION: TwinOrchestrator no longer imports expected engine node(s): "
        f"{sorted(missing)}. Update §3.3 assembly set and this test together."
    )


@pytest.mark.architecture
def test_law3_reference_graph_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Build and print the full actual node→node import graph for auditing.

    Always passes. Run with -s to see the graph output.
    """
    graph = _node_import_graph()
    doc_edges: set[tuple[str, str]] = {
        ("CoexistenceNoiseEngine", "AsynchronousControlPlane"),
        ("TwinOrchestrator", "ChannelComposer"),
        ("TwinOrchestrator", "TensorStateTracker"),
        ("TwinOrchestrator", "AsynchronousControlPlane"),
    }
    actual_edges: set[tuple[str, str]] = {
        (src, tgt) for src, targets in graph.items() for tgt in targets
    }

    print("\n=== §3.3 Reference Graph — Actual vs Documented ===")
    print("\nActual node→node import edges (from AST):")
    for edge in sorted(actual_edges):
        status = "✓ documented" if edge in doc_edges else "⚠ NOT in §3.3"
        print(f"  {edge[0]:30s} → {edge[1]:30s}  [{status}]")

    print("\nDocumented §3.3 edges missing from code:")
    for edge in sorted(doc_edges - actual_edges):
        print(f"  {edge[0]:30s} → {edge[1]:30s}  [MISSING from code]")

    assert True  # always passes; output is the audit report


# ===========================================================================
# LAW 4 — State ownership (§3.4)
# ===========================================================================


@pytest.mark.architecture
def test_law4_no_engine_imports_quantum_backends() -> None:
    """AST: no engine module imports quantum.tracker or quantum.backends.

    Only TensorStateTracker (and its QuimbAdapter backend) may hold quantum state.
    """
    forbidden = ("quantum.tracker", "quantum.backends",
                 "qndt.quantum.tracker", "qndt.quantum.backends")
    for mod_name, path in _ENGINE_MODULE_PATHS.items():
        for imp in _direct_imports(path):
            for f in forbidden:
                assert f not in imp, (
                    f"LAW 4 VIOLATION: {mod_name} imports '{imp}' — "
                    f"'{f}' is a quantum-state module forbidden in engine files"
                )


@pytest.mark.architecture
def test_law4_async_plane_does_not_import_quantum_state() -> None:
    """AST: control_plane/async_plane.py does not import quantum state modules."""
    path = _SRC / "control_plane" / "async_plane.py"
    for imp in _direct_imports(path):
        assert "quantum" not in imp, (
            f"LAW 4 VIOLATION: async_plane.py imports '{imp}' containing 'quantum'"
        )


@pytest.mark.architecture
def test_law4_no_engine_has_quantum_state_attribute(
    coexistence_engine: object, aging_model: object, env_engine: object
) -> None:
    """No engine class instance has attributes that suggest quantum state storage.

    Checks instance __dict__ for attribute names associated with quantum state.
    NOTE: raman.py has '_rho_peak' which is the Raman cross-section scalar —
    NOT a density matrix. This test explicitly allows it.
    """
    quantum_state_names = frozenset({
        "_dm", "_rho", "_state", "_state_vector", "_density_matrix",
        "_tensor", "_mpo", "_mpdo", "_psi", "_backend", "_ket",
    })
    for engine in (coexistence_engine, aging_model, env_engine):
        attrs = set(vars(engine).keys())
        suspicious = attrs & quantum_state_names
        assert suspicious == set(), (
            f"LAW 4 possible violation: {type(engine).__name__} has attribute(s) "
            f"{suspicious} that suggest quantum-state storage. "
            "If these are physics scalars (not matrices), rename them to avoid ambiguity."
        )


# ===========================================================================
# LAW 5 — Telemetry path (§3.5)
# ===========================================================================


@pytest.mark.architecture
def test_law5_no_io_in_engine_modules() -> None:
    """AST: engine and control_plane modules must not import csv/json or call open().

    I/O is confined to telemetry/sources.py and io/ per §3.5.
    """
    io_confined_paths = [
        _SRC / "physics" / "aging.py",
        _SRC / "physics" / "raman.py",
        _SRC / "physics" / "channels.py",
        _SRC / "physics" / "kernels.py",
        _SRC / "physics" / "master_equation.py",
        _SRC / "telemetry" / "engine.py",
        _SRC / "telemetry" / "resampler.py",
        _SRC / "control_plane" / "async_plane.py",
        _SRC / "control_plane" / "load.py",
        _SRC / "control_plane" / "routing.py",
        _SRC / "core" / "composer.py",
        _SRC / "core" / "context.py",
    ]
    forbidden_io_modules = frozenset({"csv", "json", "urllib", "urllib.request"})

    for path in io_confined_paths:
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_io_modules, (
                        f"LAW 5 VIOLATION: {path.name} imports '{alias.name}' — "
                        "I/O modules must only appear in telemetry/sources.py or io/"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module in forbidden_io_modules:
                    pytest.fail(
                        f"LAW 5 VIOLATION: {path.name} does 'from {node.module} import ...' — "
                        "I/O modules must only appear in telemetry/sources.py or io/"
                    )
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    pytest.fail(
                        f"LAW 5 VIOLATION: {path.name}:{node.lineno} calls open() — "
                        "file I/O must be confined to telemetry/sources.py or io/"
                    )


@pytest.mark.architecture
def test_law5_sources_is_the_io_boundary() -> None:
    """telemetry/sources.py must be the I/O boundary — it imports csv/json."""
    sources_path = _SRC / "telemetry" / "sources.py"
    has_io = _has_import(sources_path, "csv") or _has_import(sources_path, "json")
    assert has_io, (
        "telemetry/sources.py does not import csv or json — "
        "it should be the designated I/O boundary for environmental telemetry (§3.5)"
    )


@pytest.mark.architecture
def test_law5_env_engine_has_resampler_attribute(env_engine: object) -> None:
    """EnvironmentalTelemetryEngine must own a TelemetryResampler (§3.5).

    §3.5 path: TelemetrySource → TelemetryResampler → EnvironmentalTelemetryEngine → ptm(ctx).
    The engine must hold a resampler and use it for windowed convolution, not raw I/O.
    """
    from qndt.telemetry.resampler import TelemetryResampler
    assert hasattr(env_engine, "resampler"), (
        "EnvironmentalTelemetryEngine has no 'resampler' attribute — §3.5 path broken"
    )
    assert isinstance(getattr(env_engine, "resampler"), TelemetryResampler), (
        f"env_engine.resampler is {type(getattr(env_engine, 'resampler')).__name__}, "
        "not TelemetryResampler — §3.5 path broken"
    )


@pytest.mark.architecture
def test_law5_telemetry_path_structural_wiring(env_engine: object) -> None:
    """Push a sample through Source→Resampler→Engine and verify the PTM changes.

    Structural end-to-end check of the §3.5 data path at runtime.
    """
    from qndt.telemetry.sources import TelemetrySample
    # Inject a non-trivial environment reading directly into the resampler
    for dt in range(5):
        sample = TelemetrySample(
            t=float(dt) * 0.5,
            E=np.array([25.0, 0.1, 0.0]),  # +5 °C above 20 °C ref → non-zero noise
            link_id="arch_test_link",
        )
        env_engine.ingest(sample)  # type: ignore[union-attr]

    ctx2 = _minimal_ctx(t=2.0)
    ptm_after = env_engine.ptm(ctx2)  # type: ignore[union-attr]

    # After non-trivial telemetry, PTM should deviate from identity on at least one axis
    assert ptm_after.shape == (4,)
    assert float(ptm_after[0]) == pytest.approx(1.0, abs=1e-12)
    # At least one eigenvalue should be < 1 (noise was injected)
    # If all eigenvalues are 1, the telemetry path is broken
    all_identity = all(abs(float(v) - 1.0) < 1e-12 for v in ptm_after[1:])
    assert not all_identity, (
        "EnvironmentalTelemetryEngine.ptm() still returns identity after ingesting "
        "5 non-trivial telemetry samples — §3.5 telemetry path appears broken"
    )


# ===========================================================================
# LAW 6 — GUI isolation + lambda_q boundary (§3.6)
# ===========================================================================


@pytest.mark.architecture
def test_law6_no_qt_imports_outside_gui() -> None:
    """AST: no module outside src/qndt/gui/ imports PySide6, PyQt, or qndt.gui.*.

    Scans all non-gui packages: core, physics, telemetry, control_plane, quantum, io.
    Comments referencing qndt.gui are NOT imports and do not trigger this test.
    """
    non_gui_packages = [
        _SRC / "core",
        _SRC / "physics",
        _SRC / "telemetry",
        _SRC / "control_plane",
        _SRC / "quantum",
        _SRC / "io",
    ]
    forbidden_prefixes = ("PySide6", "PyQt5", "PyQt6", "qndt.gui")

    for pkg_dir in non_gui_packages:
        for py_file in pkg_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            for imp in _direct_imports(py_file):
                for prefix in forbidden_prefixes:
                    assert not imp.startswith(prefix), (
                        f"LAW 6 VIOLATION: {py_file.relative_to(_SRC)} imports "
                        f"'{imp}' — Qt/GUI types must not appear outside gui/"
                    )


def _lambda_q_nm_mult_sites(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, expr) for every `<something>.lambda_q* * <scale>` BinOp.

    Matches expressions where one operand contains 'lambda_q' (as attribute or
    name) and the other is a numeric constant.  Uses AST value comparison to
    avoid brittleness from ast.unparse() normalisation (e.g. 1e-9 → 1e-09,
    1e9 → 1000000000.0).
    """
    tree = _parse(path)
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
            continue
        left, right = node.left, node.right

        def _has_lambda_q(n: ast.expr) -> bool:
            # Walk all sub-nodes (including n itself) for Name/Attribute with 'lambda_q'
            for sub in ast.walk(n):
                if isinstance(sub, ast.Attribute) and "lambda_q" in sub.attr:
                    return True
                if isinstance(sub, ast.Name) and "lambda_q" in sub.id:
                    return True
            return False

        def _is_scale_constant(n: ast.expr) -> bool:
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                v = abs(float(n.value))
                # nm→m: 1e-9; m→nm: 1e9 (= 1000000000.0 in ast.unparse)
                return v > 1e8 or (v < 1e-8 and v > 0)
            return False

        if (_has_lambda_q(left) and _is_scale_constant(right)) or (
            _has_lambda_q(right) and _is_scale_constant(left)
        ):
            sites.append((node.lineno, ast.unparse(node)))
    return sites


@pytest.mark.architecture
def test_law6_lambda_q_unit_conversion_sites() -> None:
    """Find and verify all lambda_q nm↔m unit conversion sites in src/qndt/.

    §3.6: 'OpContext.lambda_q is stored in SI metres — the single unit-conversion
    point between engine inputs and the nm-based physics literature.'

    Expected conversion sites:
      orchestrator.py — creates OpContext: lambda_q=link.lambda_q_nm * 1e-9  (nm→m)
      raman.py ptm() — consumes OpContext: ctx.lambda_q * 1e9                (m→nm)

    RamanProfile.beta() and raman_rate() use nm-native λ arguments internally
    (lambda_c_nm, lambda_q_nm) and convert to frequency; those conversions are
    ALSO reported here because they multiply a 'lambda_*_nm' variable by 1e-9,
    but they are clearly labelled '_nm' and are not OpContext unit boundaries.
    """
    all_src = [p for p in _SRC.rglob("*.py") if "__pycache__" not in str(p)]

    ctx_lambda_q_sites: list[tuple[str, int, str]] = []
    for py_file in all_src:
        for lineno, expr in _lambda_q_nm_mult_sites(py_file):
            rel = str(py_file.relative_to(_SRC))
            ctx_lambda_q_sites.append((rel, lineno, expr))

    found_files = {site[0] for site in ctx_lambda_q_sites}

    # Both expected files must contain a lambda_q unit conversion
    assert "core/orchestrator.py" in found_files, (
        "orchestrator.py no longer contains 'lambda_q_nm * 1e-9' — "
        "verify OpContext is still created with SI metres"
    )
    assert "physics/raman.py" in found_files, (
        "raman.py no longer contains 'ctx.lambda_q * 1e9' — "
        "verify the m→nm conversion still occurs in CoexistenceNoiseEngine.ptm()"
    )

    # Non-gui, non-raman, non-orchestrator files must NOT multiply lambda_q by 1e±9
    expected = {"core/orchestrator.py", "physics/raman.py"}
    unexpected = {
        f for f in found_files
        if not f.startswith("gui/") and f not in expected
    }
    assert unexpected == set(), (
        f"LAW 6: Unexpected lambda_q unit conversion site(s) in {unexpected}. "
        "The nm↔m boundary must only be in orchestrator.py (OpContext creation) "
        "and raman.py ptm() (OpContext consumption). "
        f"Found sites: {[s for s in ctx_lambda_q_sites if s[0] in unexpected]}"
    )


@pytest.mark.architecture
def test_law6_op_context_stores_si_metres() -> None:
    """OpContext.lambda_q passes through the SI-metres value unchanged."""
    test_m = 1550e-9
    ctx = _minimal_ctx(lambda_q=test_m)
    assert ctx.lambda_q == pytest.approx(test_m, rel=1e-15), (
        "OpContext.lambda_q modified the stored value"
    )
    # Sanity: optical wavelengths in SI metres are in the 1e-7 … 2e-6 range
    assert 1e-7 < ctx.lambda_q < 2e-6, (
        f"ctx.lambda_q = {ctx.lambda_q} is outside the SI-metres range for optical "
        "wavelengths — was it accidentally stored in nm?"
    )
