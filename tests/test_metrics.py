"""Tests for the metrics registry and built-in metrics."""
import pytest
from datetime import datetime, timezone, timedelta

from bioagenteval.metrics import (
    _REGISTRY, compute_metrics, get_metric, register_metric,
)
from bioagenteval.models import MetricGroup, Transcript, TranscriptEvent


class TestRegistry:
    def test_builtin_metrics_registered(self):
        expected = {
            "n_turns", "n_tool_calls", "n_total_tokens",
            "time_to_first_token", "output_tokens_per_sec",
            "time_to_last_token",
        }
        assert expected.issubset(set(_REGISTRY.keys()))

    def test_get_metric_found(self):
        fn = get_metric("n_turns")
        assert callable(fn)

    def test_get_metric_not_found(self):
        with pytest.raises(KeyError, match="Unknown metric"):
            get_metric("nonexistent_metric")

    def test_register_custom_metric(self):
        @register_metric("test_custom")
        def _custom(transcript, duration_ms):
            return 42

        assert get_metric("test_custom") is _custom
        # Clean up
        del _REGISTRY["test_custom"]


class TestNTurns:
    def test_counts_llm_calls(self):
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(event_type="llm_call"),
                TranscriptEvent(event_type="llm_call"),
                TranscriptEvent(event_type="tool_call"),
            ],
        )
        fn = get_metric("n_turns")
        assert fn(transcript, 1000.0) == 2

    def test_empty_transcript(self):
        transcript = Transcript(task_id="t1")
        fn = get_metric("n_turns")
        assert fn(transcript, 0.0) == 0


class TestNToolCalls:
    def test_counts_tool_events(self):
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(event_type="cypher_query"),
                TranscriptEvent(event_type="tool_call"),
                TranscriptEvent(event_type="tool_use"),
                TranscriptEvent(event_type="llm_call"),
            ],
        )
        fn = get_metric("n_tool_calls")
        assert fn(transcript, 1000.0) == 3


class TestNTotalTokens:
    def test_sums_tokens(self):
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(
                    event_type="llm_call",
                    data={"prompt_tokens": 100, "completion_tokens": 50},
                ),
                TranscriptEvent(
                    event_type="llm_call",
                    data={"prompt_tokens": 200, "completion_tokens": 100},
                ),
            ],
        )
        fn = get_metric("n_total_tokens")
        assert fn(transcript, 1000.0) == 450

    def test_no_token_data(self):
        transcript = Transcript(
            task_id="t1",
            events=[TranscriptEvent(event_type="llm_call")],
        )
        fn = get_metric("n_total_tokens")
        assert fn(transcript, 1000.0) == 0


class TestTimeToFirstToken:
    def test_with_started_at_and_response(self):
        started = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        transcript = Transcript(
            task_id="t1",
            started_at=started,
            events=[
                TranscriptEvent(
                    event_type="llm_response",
                    timestamp=started + timedelta(milliseconds=500),
                ),
            ],
        )
        fn = get_metric("time_to_first_token")
        result = fn(transcript, 1000.0)
        assert result == pytest.approx(500.0)

    def test_no_started_at(self):
        transcript = Transcript(
            task_id="t1",
            events=[TranscriptEvent(event_type="llm_response")],
        )
        fn = get_metric("time_to_first_token")
        assert fn(transcript, 1000.0) is None

    def test_no_response_events(self):
        started = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        transcript = Transcript(
            task_id="t1",
            started_at=started,
            events=[TranscriptEvent(event_type="tool_call")],
        )
        fn = get_metric("time_to_first_token")
        assert fn(transcript, 1000.0) is None


class TestOutputTokensPerSec:
    def test_calculation(self):
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(
                    event_type="llm_call",
                    data={"completion_tokens": 100},
                ),
            ],
        )
        fn = get_metric("output_tokens_per_sec")
        # 100 tokens in 2000ms = 50 tokens/sec
        assert fn(transcript, 2000.0) == pytest.approx(50.0)

    def test_zero_duration(self):
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(
                    event_type="llm_call",
                    data={"completion_tokens": 100},
                ),
            ],
        )
        fn = get_metric("output_tokens_per_sec")
        assert fn(transcript, 0.0) is None

    def test_no_completion_tokens(self):
        transcript = Transcript(task_id="t1")
        fn = get_metric("output_tokens_per_sec")
        assert fn(transcript, 1000.0) is None


class TestTimeToLastToken:
    def test_passthrough(self):
        transcript = Transcript(task_id="t1")
        fn = get_metric("time_to_last_token")
        assert fn(transcript, 1234.5) == 1234.5


class TestComputeMetrics:
    def test_computes_requested_metrics(self):
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(event_type="llm_call"),
                TranscriptEvent(event_type="tool_call"),
            ],
        )
        groups = [
            MetricGroup(type="transcript", metrics=["n_turns", "n_tool_calls"]),
        ]
        result = compute_metrics(groups, transcript, 500.0)
        assert result["n_turns"] == 1
        assert result["n_tool_calls"] == 1

    def test_empty_groups(self):
        transcript = Transcript(task_id="t1")
        result = compute_metrics([], transcript, 0.0)
        assert result == {}

    def test_unknown_metrics_skipped(self):
        transcript = Transcript(task_id="t1")
        groups = [
            MetricGroup(type="custom", metrics=["nonexistent_metric"]),
        ]
        result = compute_metrics(groups, transcript, 0.0)
        assert "nonexistent_metric" not in result

    def test_multiple_groups(self):
        transcript = Transcript(
            task_id="t1",
            events=[TranscriptEvent(event_type="llm_call")],
        )
        groups = [
            MetricGroup(type="transcript", metrics=["n_turns"]),
            MetricGroup(type="latency", metrics=["time_to_last_token"]),
        ]
        result = compute_metrics(groups, transcript, 750.0)
        assert result["n_turns"] == 1
        assert result["time_to_last_token"] == 750.0
