"""Metric registry and built-in metrics for evaluation trials."""
from __future__ import annotations

from typing import Any, Callable

from bioagenteval.models import MetricGroup, Transcript

MetricFn = Callable[[Transcript, float], Any]

_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator to register a metric function by name."""
    def decorator(fn: MetricFn) -> MetricFn:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_metric(name: str) -> MetricFn:
    """Look up a registered metric by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric: {name!r}")
    return _REGISTRY[name]


def compute_metrics(
    metric_groups: list[MetricGroup],
    transcript: Transcript,
    duration_ms: float,
) -> dict[str, Any]:
    """Compute all requested metrics from metric groups."""
    results: dict[str, Any] = {}
    for group in metric_groups:
        for metric_name in group.metrics:
            if metric_name in _REGISTRY:
                results[metric_name] = _REGISTRY[metric_name](transcript, duration_ms)
    return results


# --- Built-in metrics ---

@register_metric("n_turns")
def _n_turns(transcript: Transcript, duration_ms: float) -> int:
    """Count llm_call events."""
    return sum(1 for ev in transcript.events if ev.event_type == "llm_call")


@register_metric("n_tool_calls")
def _n_tool_calls(transcript: Transcript, duration_ms: float) -> int:
    """Count cypher_query, tool_call, and tool_use events."""
    tool_types = {"cypher_query", "tool_call", "tool_use"}
    return sum(1 for ev in transcript.events if ev.event_type in tool_types)


@register_metric("n_total_tokens")
def _n_total_tokens(transcript: Transcript, duration_ms: float) -> int:
    """Sum token counts from events."""
    total = 0
    for ev in transcript.events:
        total += ev.data.get("prompt_tokens", 0)
        total += ev.data.get("completion_tokens", 0)
    return total


@register_metric("time_to_first_token")
def _time_to_first_token(transcript: Transcript, duration_ms: float) -> float | None:
    """Delta from transcript.started_at to first response event."""
    if transcript.started_at is None:
        return None
    for ev in transcript.events:
        if ev.event_type in ("llm_response", "llm_call"):
            delta = (ev.timestamp - transcript.started_at).total_seconds() * 1000
            return delta
    return None


@register_metric("output_tokens_per_sec")
def _output_tokens_per_sec(transcript: Transcript, duration_ms: float) -> float | None:
    """Completion tokens per second."""
    if duration_ms <= 0:
        return None
    completion_tokens = sum(
        ev.data.get("completion_tokens", 0) for ev in transcript.events
    )
    if completion_tokens == 0:
        return None
    return completion_tokens / (duration_ms / 1000)


@register_metric("time_to_last_token")
def _time_to_last_token(transcript: Transcript, duration_ms: float) -> float:
    """Passthrough of duration_ms."""
    return duration_ms
