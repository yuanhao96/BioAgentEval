# Multi-Provider Model Grader + LLM Judge Enhancements

**Date**: 2026-02-25
**Milestone**: Multi-Provider Model Grader + LLM Judge Enhancements

## Summary

Refactor ModelGrader to support both OpenAI and Anthropic providers via an LLMClient abstraction. Add LLM judge "Unknown" escape hatch, pairwise comparison grading, and multi-judge consensus.

## Files to Create

1. `src/bioagenteval/graders/llm_client.py` — LLMClient protocol + OpenAI/Anthropic implementations + factory
2. `src/bioagenteval/graders/pairwise_grader.py` — Pairwise comparison grader
3. `src/bioagenteval/graders/consensus_grader.py` — Multi-judge consensus grader
4. `tests/test_llm_client.py` — Tests for LLMClient implementations
5. `tests/test_pairwise_grader.py` — Tests for PairwiseGrader
6. `tests/test_consensus_grader.py` — Tests for ConsensusGrader

## Files to Modify

1. `src/bioagenteval/graders/model_grader.py` — Use LLMClient, add Unknown handling
2. `src/bioagenteval/models.py` — Handle unknown grades in weighted_score()
3. `src/bioagenteval/graders/__init__.py` — Export new graders
4. `src/bioagenteval/__main__.py` — Support provider config, new grader types
5. `tests/test_model_grader.py` — Update for LLMClient, add Unknown tests
6. `tests/test_models.py` — Add tests for unknown grade handling in weighted_score

## Implementation Steps

### Step 1: Create LLMClient abstraction (`llm_client.py`)

Create `src/bioagenteval/graders/llm_client.py`:
- `LLMClient` Protocol with method: `complete(messages: list[dict], system: str, model: str, max_tokens: int) -> str`
- `OpenAILLMClient` — wraps `openai.OpenAI`, maps system prompt into messages[0], extracts `choices[0].message.content`
- `AnthropicLLMClient` — wraps `anthropic.Anthropic`, passes system as separate param, extracts `content[0].text`
- `create_llm_client(provider: str = "openai", **kwargs) -> LLMClient` factory function
- Both clients load .env from project root

### Step 2: Refactor ModelGrader to use LLMClient + add Unknown handling

Modify `src/bioagenteval/graders/model_grader.py`:
- Constructor accepts `client: LLMClient | None = None` and `provider: str = "openai"` and `model: str` (default varies by provider)
- If no client provided, create one via `create_llm_client(provider)`
- Update GRADING_PROMPT to include: `If you cannot determine a score due to insufficient information, respond with: {"score": 0.0, "passed": false, "verdict": "unknown", "reasoning": "..."}`
- Parse response: if `verdict == "unknown"`, set `details["status"] = "unknown"`
- Backward compatible: existing tests still work with default OpenAI provider

### Step 3: Update models.py — Unknown grade handling

Modify `src/bioagenteval/models.py`:
- `TrialResult.weighted_score()`: exclude grades where `details.get("status") == "unknown"` from the weighted average
- `TrialResult.weighted_passed()`: same exclusion logic
- Add `TrialResult.has_unknown_grades() -> bool` helper

### Step 4: Create PairwiseGrader (`pairwise_grader.py`)

Create `src/bioagenteval/graders/pairwise_grader.py`:
- `PairwiseGrader(BaseGrader)` — compares two agent outputs
- Constructor: `__init__(client: LLMClient | None, provider: str, model: str)`
- `grade()`: reads `config.params["reference_outcome"]` as the second output
- Prompt asks LLM to compare outcome A vs B and return: `{"preferred": "A"|"B"|"tie", "score": float, "passed": bool, "reasoning": str}`
- Score: 1.0 if preferred=A (the agent's output), 0.5 if tie, 0.0 if B
- If reference_outcome not in config.params, logs warning and returns score=0.0

### Step 5: Create ConsensusGrader (`consensus_grader.py`)

Create `src/bioagenteval/graders/consensus_grader.py`:
- `ConsensusGrader(BaseGrader)` — wraps an inner grader and runs it N times
- Constructor: `__init__(inner_grader: BaseGrader, num_judges: int = 3)`
- `grade()`: calls `inner_grader.grade()` N times, aggregates:
  - `passed`: majority vote (>50% of judges must pass)
  - `score`: average of all judge scores
  - `details`: includes individual judge results and vote counts
- Handles exceptions from individual judges gracefully (exclude from voting)

### Step 6: Update graders/__init__.py

Add exports: `PairwiseGrader`, `ConsensusGrader`, `create_llm_client`

### Step 7: Update __main__.py

- Add `--provider` option to `run` command (default: "openai", choices: ["openai", "anthropic"])
- Pass provider to ModelGrader constructor
- Register "pairwise" and "consensus" grader types in the graders dict

### Step 8: Write tests for LLMClient (`test_llm_client.py`)

- Test OpenAILLMClient.complete() with mocked OpenAI client
- Test AnthropicLLMClient.complete() with mocked Anthropic client
- Test create_llm_client factory for both providers
- Test create_llm_client with invalid provider raises ValueError

### Step 9: Update tests for ModelGrader (`test_model_grader.py`)

- Update existing tests to work with new LLMClient-based constructor
- Add test for Anthropic provider
- Add test for Unknown verdict handling
- Add test for backward compatibility (default provider=openai)

### Step 10: Write tests for PairwiseGrader (`test_pairwise_grader.py`)

- Test preference for outcome A (score=1.0)
- Test preference for outcome B (score=0.0)
- Test tie (score=0.5)
- Test missing reference_outcome in config.params
- Test API error handling

### Step 11: Write tests for ConsensusGrader (`test_consensus_grader.py`)

- Test unanimous pass (3/3)
- Test majority pass (2/3)
- Test majority fail (1/3)
- Test with inner grader exceptions
- Test vote counting in details

### Step 12: Add tests for unknown grade handling in models (`test_models.py`)

- Test weighted_score() excludes unknown grades
- Test weighted_passed() excludes unknown grades
- Test has_unknown_grades() helper
- Test all-unknown grades returns 0.0

### Step 13: Run full test suite

- Verify all existing 181 tests pass
- Verify all new tests pass
- Check no regressions

## Dependency Order

Steps 1 → 2 → 3 (sequential: each depends on prior)
Steps 4, 5 independent of each other (but depend on Step 1)
Step 6 depends on Steps 2, 4, 5
Step 7 depends on Step 6
Steps 8-12 can be written alongside their implementation steps
Step 13 is final
