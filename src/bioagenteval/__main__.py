import json
import logging
from pathlib import Path

import click

from bioagenteval.graders import CodeGrader, HumanGrader, ModelGrader
from bioagenteval.loader import filter_tasks_by_tags, load_suite
from bioagenteval.reporter import EvalReporter
from bioagenteval.runner import EvalRunner


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(verbose):
    """BioAgentEval — evaluation harness for biomedical KG agents."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _parse_tags(tag_strings: tuple[str, ...]) -> dict[str, str]:
    """Parse key=value tag strings into a dict."""
    tags: dict[str, str] = {}
    for s in tag_strings:
        if "=" not in s:
            raise click.BadParameter(f"Tag must be key=value, got: {s!r}")
        k, v = s.split("=", 1)
        tags[k] = v
    return tags


@cli.command()
@click.argument("suite_path", type=click.Path(exists=True))
@click.option("--agent", "-a", required=True, help="Agent module path, e.g. bioagenteval.agents.baseline_qa:BaselineQAAgent")
@click.option("--output", "-o", default="eval_report.json", help="Output JSON path.")
@click.option("--skip-model-grader", is_flag=True, help="Skip model-based grading.")
@click.option("--tags", "-t", multiple=True, help="Filter tasks by tag (key=value). Repeatable.")
def run(suite_path, agent, output, skip_model_grader, tags):
    """Run an evaluation suite against an agent."""
    # Load suite
    suite, tasks = load_suite(suite_path)
    if tags:
        tag_filters = _parse_tags(tags)
        tasks = filter_tasks_by_tags(tasks, tag_filters)
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
    EvalReporter.save_report(suite.name, results, output, eval_type=suite.eval_type)
    click.echo(f"Report saved to {output}")

    # Print summary
    report = EvalReporter.generate_report(suite.name, results, eval_type=suite.eval_type)
    summary = report["summary"]
    click.echo(f"Tasks: {summary['total_tasks']}, Overall pass@1: {summary['overall_pass_at_1']:.2%}")


@cli.command()
@click.argument("suite_path", type=click.Path(exists=True))
@click.option("--tags", "-t", multiple=True, help="Filter tasks by tag (key=value). Repeatable.")
def validate(suite_path, tags):
    """Validate a task suite YAML file."""
    suite, tasks = load_suite(suite_path)
    if tags:
        tag_filters = _parse_tags(tags)
        tasks = filter_tasks_by_tags(tasks, tag_filters)
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


@cli.command()
@click.argument("report_a", type=click.Path(exists=True))
@click.argument("report_b", type=click.Path(exists=True))
def diff(report_a, report_b):
    """Compare two evaluation reports and show per-task changes."""
    with open(report_a) as f:
        a = json.load(f)
    with open(report_b) as f:
        b = json.load(f)

    a_tasks = {r["task_id"]: r for r in a.get("results", [])}
    b_tasks = {r["task_id"]: r for r in b.get("results", [])}

    all_ids = sorted(set(a_tasks) | set(b_tasks))

    click.echo(f"Comparing: {Path(report_a).name} vs {Path(report_b).name}")
    click.echo(f"{'Task':<40} {'pass@1 A':>10} {'pass@1 B':>10} {'Delta':>10}")
    click.echo("-" * 72)

    for tid in all_ids:
        pa = a_tasks.get(tid, {}).get("pass_at_1")
        pb = b_tasks.get(tid, {}).get("pass_at_1")
        pa_str = f"{pa:.2%}" if pa is not None else "N/A"
        pb_str = f"{pb:.2%}" if pb is not None else "N/A"
        if pa is not None and pb is not None:
            delta = pb - pa
            sign = "+" if delta > 0 else ""
            delta_str = f"{sign}{delta:.2%}"
        else:
            delta_str = "-"
        click.echo(f"{tid:<40} {pa_str:>10} {pb_str:>10} {delta_str:>10}")

    # Summary
    sa = a.get("summary", {})
    sb = b.get("summary", {})
    click.echo("-" * 72)
    oa = sa.get("overall_pass_at_1", 0)
    ob = sb.get("overall_pass_at_1", 0)
    delta = ob - oa
    sign = "+" if delta > 0 else ""
    click.echo(f"{'Overall pass@1':<40} {oa:>9.2%} {ob:>10.2%} {sign}{delta:>9.2%}")


if __name__ == "__main__":
    cli()
