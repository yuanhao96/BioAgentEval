import json
import logging
from pathlib import Path

import click

from bioagenteval.graders import CodeGrader, HumanGrader, ModelGrader
from bioagenteval.loader import load_suite
from bioagenteval.reporter import EvalReporter
from bioagenteval.runner import EvalRunner


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(verbose):
    """BioAgentEval — evaluation harness for biomedical KG agents."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@cli.command()
@click.argument("suite_path", type=click.Path(exists=True))
@click.option("--agent", "-a", required=True, help="Agent module path, e.g. bioagenteval.agents.baseline_qa:BaselineQAAgent")
@click.option("--output", "-o", default="eval_report.json", help="Output JSON path.")
@click.option("--skip-model-grader", is_flag=True, help="Skip model-based grading.")
def run(suite_path, agent, output, skip_model_grader):
    """Run an evaluation suite against an agent."""
    # Load suite
    suite, tasks = load_suite(suite_path)
    click.echo(f"Loaded suite '{suite.name}' with {len(tasks)} tasks")

    # Import agent class
    module_path, class_name = agent.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    agent_cls = getattr(mod, class_name)
    agent_instance = agent_cls()

    # Set up graders
    graders = {"code": CodeGrader()}
    if not skip_model_grader:
        graders["model"] = ModelGrader()
    graders["human"] = HumanGrader()

    # Run
    runner = EvalRunner(agent=agent_instance, graders=graders)
    results = runner.run_suite(tasks)

    # Report
    EvalReporter.save_report(suite.name, results, output)
    click.echo(f"Report saved to {output}")

    # Print summary
    report = EvalReporter.generate_report(suite.name, results)
    summary = report["summary"]
    click.echo(f"Tasks: {summary['total_tasks']}, Overall pass@1: {summary['overall_pass_at_1']:.2%}")


@cli.command()
@click.argument("suite_path", type=click.Path(exists=True))
def validate(suite_path):
    """Validate a task suite YAML file."""
    suite, tasks = load_suite(suite_path)
    click.echo(f"Suite: {suite.name}")
    click.echo(f"Tasks: {len(tasks)}")
    for t in tasks:
        grader_types = [g.type for g in t.graders]
        eo_types = [eo.type for eo in t.expected_output]
        tag_str = ", ".join(f"{k}={v}" for k, v in t.tags.items()) if t.tags else ""
        parts = [f"  {t.id}: {t.num_trials} trials, graders={grader_types}"]
        if eo_types:
            parts.append(f"expected_output={eo_types}")
        if tag_str:
            parts.append(f"tags=[{tag_str}]")
        click.echo(", ".join(parts))
    click.echo("Validation passed.")


if __name__ == "__main__":
    cli()
