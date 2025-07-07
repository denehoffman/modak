from __future__ import annotations

from pathlib import Path

import click
from modak import run_queue_wrapper


@click.command()
@click.argument(
    "state_file",
    type=click.Path(exists=True, file_okay=True),
    default=Path.home() / ".modak/state.db",
    required=False,
)
def cli(state_file: Path):
    run_queue_wrapper(state_file)
