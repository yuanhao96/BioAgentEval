"""Tests for transcript analysis utilities."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bioagenteval.models import Transcript, TranscriptEvent
from bioagenteval.transcript_analysis import (
    detect_retries,
    extract_tool_sequence,
    summarize_transcript,
)


def _make_event(event_type: str, event_name: str = "", data: dict | None = None):
    return TranscriptEvent(event_type=event_type, event_name=event_name, data=data or {})


class TestSummarizeTranscript:
    def test_basic_summary(self):
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("llm_call"),
                _make_event("tool_call"),
                _make_event("llm_call"),
            ],
        )
        result = summarize_transcript(t)

        assert result["total_events"] == 3
        assert result["event_counts"] == {"llm_call": 2, "tool_call": 1}
        assert result["has_errors"] is False

    def test_empty_transcript(self):
        t = Transcript(task_id="t1")
        result = summarize_transcript(t)

        assert result["total_events"] == 0
        assert result["event_counts"] == {}
        assert result["duration_ms"] is None

    def test_duration_computed(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(seconds=2.5)
        t = Transcript(
            task_id="t1",
            started_at=start,
            finished_at=end,
            events=[_make_event("llm_call")],
        )
        result = summarize_transcript(t)

        assert result["duration_ms"] == 2500.0

    def test_has_errors_detected(self):
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("llm_call"),
                _make_event("tool_call", data={"error": "API timeout"}),
            ],
        )
        result = summarize_transcript(t)

        assert result["has_errors"] is True


class TestExtractToolSequence:
    def test_basic_sequence(self):
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("llm_call"),
                _make_event("cypher_query"),
                _make_event("llm_call"),
            ],
        )
        result = extract_tool_sequence(t)

        assert result == ["llm_call", "cypher_query", "llm_call"]

    def test_with_event_names(self):
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("tool_call", event_name="search"),
                _make_event("tool_call", event_name="query"),
            ],
        )
        result = extract_tool_sequence(t)

        assert result == ["tool_call:search", "tool_call:query"]

    def test_empty_transcript(self):
        t = Transcript(task_id="t1")
        assert extract_tool_sequence(t) == []

    def test_mixed_named_and_unnamed(self):
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("llm_call"),
                _make_event("tool_call", event_name="search"),
            ],
        )
        result = extract_tool_sequence(t)

        assert result == ["llm_call", "tool_call:search"]


class TestDetectRetries:
    def test_no_retries(self):
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("llm_call"),
                _make_event("tool_call"),
                _make_event("llm_call"),
            ],
        )
        assert detect_retries(t) == []

    def test_detects_repeated_events(self):
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("llm_call"),
                _make_event("cypher_query"),
                _make_event("cypher_query"),
                _make_event("cypher_query"),
                _make_event("llm_call"),
            ],
        )
        result = detect_retries(t)

        assert len(result) == 1
        assert result[0]["event_type"] == "cypher_query"
        assert result[0]["count"] == 3
        assert result[0]["start_index"] == 1

    def test_custom_threshold(self):
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("llm_call"),
                _make_event("llm_call"),
            ],
        )
        # threshold=2 (default) catches it
        assert len(detect_retries(t, threshold=2)) == 1
        # threshold=3 does not
        assert len(detect_retries(t, threshold=3)) == 0

    def test_multiple_retry_groups(self):
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("llm_call"),
                _make_event("llm_call"),
                _make_event("tool_call"),
                _make_event("tool_call"),
                _make_event("tool_call"),
            ],
        )
        result = detect_retries(t)

        assert len(result) == 2
        assert result[0]["event_type"] == "llm_call"
        assert result[0]["count"] == 2
        assert result[1]["event_type"] == "tool_call"
        assert result[1]["count"] == 3

    def test_empty_transcript(self):
        t = Transcript(task_id="t1")
        assert detect_retries(t) == []

    def test_respects_event_name(self):
        """Same event_type but different event_name = not a retry."""
        t = Transcript(
            task_id="t1",
            events=[
                _make_event("tool_call", event_name="search"),
                _make_event("tool_call", event_name="query"),
            ],
        )
        assert detect_retries(t) == []
