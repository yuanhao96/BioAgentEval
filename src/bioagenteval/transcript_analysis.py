"""Transcript analysis utilities for debugging agent trajectories."""
from __future__ import annotations

from collections import Counter
from typing import Any

from bioagenteval.models import Transcript


def summarize_transcript(transcript: Transcript) -> dict[str, Any]:
    """Summarize a transcript into event counts and timing info.

    Returns a dict with:
    - total_events: int
    - event_counts: dict mapping event_type to count
    - duration_ms: float or None (if started_at and finished_at set)
    - has_errors: bool (any event with "error" in data)
    """
    events = transcript.events
    counter: Counter[str] = Counter(ev.event_type for ev in events)

    duration_ms = None
    if transcript.started_at and transcript.finished_at:
        delta = transcript.finished_at - transcript.started_at
        duration_ms = delta.total_seconds() * 1000

    has_errors = any("error" in ev.data for ev in events)

    return {
        "total_events": len(events),
        "event_counts": dict(counter),
        "duration_ms": duration_ms,
        "has_errors": has_errors,
    }


def extract_tool_sequence(transcript: Transcript) -> list[str]:
    """Extract the ordered sequence of event types from a transcript.

    Returns a list of event_type strings in chronological order.
    If event_name is set, returns "event_type:event_name" for richer info.
    """
    sequence: list[str] = []
    for ev in transcript.events:
        if ev.event_name:
            sequence.append(f"{ev.event_type}:{ev.event_name}")
        else:
            sequence.append(ev.event_type)
    return sequence


def detect_retries(
    transcript: Transcript, threshold: int = 2,
) -> list[dict[str, Any]]:
    """Detect consecutive repeated events that may indicate agent looping.

    Finds runs of consecutive events with the same event_type (and same
    event_name if set) that are at least `threshold` long.

    Returns a list of dicts, each with:
    - event_type: str
    - event_name: str
    - count: int (length of the run)
    - start_index: int (index of first event in the run)
    """
    events = transcript.events
    if not events:
        return []

    retries: list[dict[str, Any]] = []
    run_start = 0
    run_count = 1

    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        if curr.event_type == prev.event_type and curr.event_name == prev.event_name:
            run_count += 1
        else:
            if run_count >= threshold:
                retries.append({
                    "event_type": events[run_start].event_type,
                    "event_name": events[run_start].event_name,
                    "count": run_count,
                    "start_index": run_start,
                })
            run_start = i
            run_count = 1

    # Check final run
    if run_count >= threshold:
        retries.append({
            "event_type": events[run_start].event_type,
            "event_name": events[run_start].event_name,
            "count": run_count,
            "start_index": run_start,
        })

    return retries
