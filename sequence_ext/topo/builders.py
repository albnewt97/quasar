from typing import Dict, Any

# TODO: Build SeQUeNCe Topology objects from YAML configs

def build_static_topology(cfg: Dict[str, Any]):
"""Return a topology object from config (stub)."""
return {"nodes": cfg.get("nodes", []), "channels": cfg.get("channels", [])}
