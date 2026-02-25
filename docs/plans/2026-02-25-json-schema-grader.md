# Plan: Add json_schema check function to CodeGrader

**Date**: 2026-02-25
**Milestone**: Add json_schema check function to CodeGrader

## Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add `jsonschema>=4.0` to runtime dependencies |
| `src/bioagenteval/graders/code_grader.py` | Add `_check_json_schema()` function and dispatch branch |

## Implementation Steps

### Step 1: Add `jsonschema` dependency to `pyproject.toml`

Add `"jsonschema>=4.0"` to the `dependencies` list in `pyproject.toml`. Then run `pip install -e ".[dev]"` to install it.

### Step 2: Add `_check_json_schema()` function to `code_grader.py`

Add a new module-level function following the existing `_check_*` pattern:

```python
def _check_json_schema(schema: dict, outcome: str) -> tuple[float, dict[str, Any]]:
```

Unlike other `_check_*` functions that return only a float, this one returns a tuple of `(score, details)` because the json_schema check needs to return structured validation error info.

Logic:
1. Try `json.loads(outcome)` — if it fails, return `(0.0, {"parse_error": str(e)})`.
2. Try `jsonschema.validate(data, schema)` — if it passes, return `(1.0, {})`.
3. On `jsonschema.ValidationError`: return `(0.0, {"validation_errors": [str(e) for each error]})`. Use `jsonschema.Draft7Validator(schema).iter_errors(data)` to collect all errors, not just the first.
4. On `jsonschema.SchemaError`: return `(0.0, {"schema_error": str(e)})`.

### Step 3: Add dispatch branch in `CodeGrader.grade()`

Add an `elif eo.type == "json_schema"` branch in the dispatch loop. Since `_check_json_schema` returns extra details, merge those details into the `GradeResult.details` dict:

```python
elif eo.type == "json_schema":
    score, extra = _check_json_schema(eo.value, outcome)
    check_results["json_schema"] = score
    if extra:
        details.update(extra)
```

This requires introducing a `details: dict[str, Any] = {}` variable before the loop and passing it into the final `GradeResult`.

### Step 4: Add `import json` and `import jsonschema` to code_grader.py

Add the necessary imports at the top of the file.

## Dependency Order

Steps 1 → 4 → 2 → 3 (dependency must be installed before imports; imports before function; function before dispatch).

In practice, we'll do: Step 1 (add dep + install), then Steps 2-4 together in one edit to code_grader.py.

## Verification

After implementation, run:
```bash
python -c "from bioagenteval.graders.code_grader import CodeGrader; print('Import OK')"
```

Full tests will be added in the next milestone. For now, verify the import works and the function is callable.

## Rollback

- `pyproject.toml`: revert the one-line addition
- `code_grader.py`: revert the added imports, function, and dispatch branch
- No database, config, or CI changes involved
