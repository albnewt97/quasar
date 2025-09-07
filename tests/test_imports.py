import importlib

def test_imports():
    assert importlib.import_module("sequence_ext")
    assert importlib.import_module("sequence_ext.scenarios.scenario1_static")
    assert importlib.import_module("sequence_ext.orchestrator.mdi_controller")
