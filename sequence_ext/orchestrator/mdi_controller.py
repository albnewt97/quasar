from typing import Any


class MDIController:
    """Coordinator for MDI-QKD orchestration hooks.

    Responsibilities (to be implemented):
      - Drive pulse schedules for Alice/Bob.
      - Listen to BSM/detector events and push to Recorder.
      - Perform MDI sifting to derive raw bits.
      - Compute Cascade-style KPIs (throughput, latency, error rate) downstream.
    """

    def __init__(self, recorder: Any) -> None:
        self.recorder = recorder

    def on_bsm_event(self, event: dict) -> None:
        """Callback for BSM outcomes (to be wired to SeQUeNCe)."""
        self.recorder.append("bsm_event", event)

    def on_detector_event(self, event: dict) -> None:
        """Callback for detector clicks (to be wired to SeQUeNCe)."""
        self.recorder.append("detector_event", event)
