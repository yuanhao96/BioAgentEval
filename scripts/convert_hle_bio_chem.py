"""Convert the HLE Bio/Chem HuggingFace dataset to a BioAgentEval YAML suite."""
from __future__ import annotations

from pathlib import Path

import yaml
from datasets import load_from_disk


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "hle_bio_chem"
OUT_PATH = Path(__file__).resolve().parent.parent / "tasks" / "hle_bio_chem.yaml"


def _build_rubric(answer: str, answer_type: str, rationale: str) -> str:
    """Build a model-grader rubric from the ground-truth answer and rationale."""
    rubric_parts = [
        f"Expected answer: {answer}",
        f"Answer type: {answer_type}",
        "",
        "Rationale:",
        rationale,
        "",
        "Scoring guidelines:",
        "- 1.0: The agent's response clearly contains the correct answer.",
        "- 0.5: The agent's response is partially correct or contains the right reasoning but wrong final answer.",
        "- 0.0: The agent's response is incorrect or does not address the question.",
    ]
    return "\n".join(rubric_parts)


def convert() -> None:
    ds = load_from_disk(str(DATA_PATH))
    df = ds["train"].to_pandas()

    tasks = []
    for _, row in df.iterrows():
        answer = str(row["answer"])
        answer_type = str(row["answer_type"])
        rationale = str(row["rationale"])

        graders = [
            {
                "type": "model",
                "rubric": _build_rubric(answer, answer_type, rationale),
            },
        ]

        task = {
            "id": str(row["id"]),
            "question": str(row["question"]),
            "expected_entities": [answer],
            "metadata": {
                "answer": answer,
                "answer_type": answer_type,
                "raw_subject": str(row["raw_subject"]),
                "category": str(row["category"]),
                "rationale": rationale,
                "source": "hle_bio_chem",
            },
            "graders": graders,
        }
        tasks.append(task)

    suite = {
        "name": "hle_bio_chem",
        "description": (
            "Humanity's Last Exam — Biology & Chemistry subset. "
            "149 expert-level questions spanning biology, chemistry, "
            "medicine, genetics, and related fields. "
            "Includes both multiple-choice and exact-match answer types."
        ),
        "default_num_trials": 1,
        "tasks": tasks,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        yaml.dump(suite, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    print(f"Wrote {len(tasks)} tasks to {OUT_PATH}")


if __name__ == "__main__":
    convert()
