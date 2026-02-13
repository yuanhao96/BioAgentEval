# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BioAgentEval** is an evaluation harness for biomedical knowledge-graph QA agents. It follows the architecture described in [Anthropic's "Demystifying Evals for AI Agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). The harness measures agent quality via multiple grader types, multi-trial pass@k/pass^k metrics, and full trajectory capture.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file or test
pytest tests/test_models.py
pytest tests/test_models.py::TestEvalResult::test_eval_result_aggregation

# CLI entry point
bioagenteval
```

## Architecture

```
src/bioagenteval/      # Main package (installed as "bioagenteval")
  __main__.py          # Click CLI group — register new commands here
  models.py            # Pydantic data models (see below)
tests/
  conftest.py          # Adds src/ to sys.path
  test_models.py       # Unit tests for all models
docs/plans/            # Design docs and eval plans
```

**Layout convention**: `src/` layout with `setuptools`. Package discovery via `[tool.setuptools.packages.find] where = ["src"]`.

### Core Data Model Hierarchy

```
EvalSuite (named group of tasks)
  └── Task (single question + expected entities + graders)
        └── EvalResult (aggregated across trials of one task)
              └── TrialResult (one attempt)
                    ├── Transcript (full trajectory)
                    │     └── TranscriptEvent (single step: tool call, Cypher query, etc.)
                    └── GradeResult (output from one grader)
```

Key design decisions:
- **Three grader types**: `code` (deterministic checks), `model` (LLM-based rubric scoring), `human` (calibration). Configured via `GraderConfig`.
- **pass@k** uses the unbiased combinatorial estimator (not naive sampling). Implemented in `EvalResult.pass_at_k()`.
- **Transcripts** capture both generic events and Neo4j-specific artifacts (Cypher queries, results) for debugging agent trajectories.
- **Tasks** carry `expected_entities` (e.g., gene names, ENSEMBL IDs) and `expected_cypher_patterns` for matching against agent outputs.

### Domain Context

The target agent answers biomedical questions over a Neo4j knowledge graph containing genes, diseases, pathways, and their relationships. Tasks test entity retrieval, Cypher generation correctness, and answer completeness. Complexity is classified as `simple` or `complex`.

## Dependencies

- `pydantic>=2.0` — All models are Pydantic v2 BaseModels
- `anthropic>=0.66.0` — For model-based grading and agent interaction
- `click>=8.0` — CLI framework
- `PyYAML>=6.0` — Task definition loading
- `pytest>=7.0`, `pytest-mock>=3.10` — Testing (dev only)

## Eval Design Principles (from Anthropic guide)

- Grade the **outcome**, not the tool-call path — don't penalize creative agent strategies.
- Every task must be **unambiguous** — two domain experts should independently agree on pass/fail.
- Start with **20–50 real failure cases**; grow the suite from there.
- Record full transcripts and **read them regularly** to catch grading bugs.
- Promote saturated evals (near 100% pass) to **regression suites**; keep active evals diagnostic.
