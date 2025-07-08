import sys
import click
from pathlib import Path
from modak import run_queue_wrapper, reset_project


@click.group(invoke_without_command=True)
@click.option("-p", "--project", "project_name", type=str, default=None)
@click.option(
    "-d",
    "--db",
    "state_file",
    type=click.Path(exists=True, file_okay=False),
    default=Path.home() / ".modak",
    show_default=True,
    help="Path to the state database file.",
)
@click.pass_context
def cli(ctx, project_name, state_file):
    if ctx.invoked_subcommand is None:
        run_queue_wrapper(state_file, project_name)


@cli.command()
@click.option("-f", "--force", is_flag=True)
@click.argument("project_name")
@click.option(
    "-d",
    "--db",
    "state_file",
    type=click.Path(exists=True, file_okay=False),
    default=Path.home() / ".modak",
    show_default=True,
    help="Path to the state database file.",
)
def rm(force, project_name, state_file):
    if not force:
        click.echo(f"Are you sure you want to delete project '{project_name}'? [y/N]")
        confirm = input().strip().lower()
        if confirm != "y":
            click.echo("Aborted.")
            sys.exit(1)
    reset_project(state_file, project_name)
    click.echo(f"Project '{project_name}' deleted.")
