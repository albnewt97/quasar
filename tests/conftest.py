# Ensure the repo root is on sys.path so 'sequence_ext' can be imported in tests
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
