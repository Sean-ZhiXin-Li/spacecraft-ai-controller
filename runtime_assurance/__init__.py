from __future__ import annotations

from importlib import import_module

__all__ = [
    "FALLBACK_ACTION",
    "FALLBACK_PROVEN_SAFE",
    "MONITOR_ID",
    "OVERSPEED_COMPARATOR",
    "OVERSPEED_THRESHOLD",
    "PREDICTION_HORIZON_STEPS",
    "FinalVetoDecision",
    "MonitorEvaluationError",
    "OneStepPrediction",
    "evaluate_overspeed_veto",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    monitor = import_module("runtime_assurance.final_veto_monitor")
    return getattr(monitor, name)
