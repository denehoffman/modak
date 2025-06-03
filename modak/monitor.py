from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import click
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.widgets import (
    Footer,
    Header,
    Static,
)

from modak import run_queue_wrapper
from modak.graph import from_state


class GraphDisplay(Static):
    def __init__(self, state_file: Path, *args, **kwargs):
        self.state_file = state_file
        self.state: str = ""
        super().__init__(*args, **kwargs)

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh_graph)

    def refresh_graph(self) -> None:
        if self.state_file.exists():
            new_state = self.state_file.read_text()
            if new_state != self.state:
                self.state = new_state
                try:
                    graph = from_state(json.loads(self.state))
                    self.update(graph)
                except:  # noqa: E722
                    pass
        else:
            self.state = ""
            self.update("")


class GraphApp(App):
    CSS = """
#main {
    padding: 2;
    border: double $accent;
    height: 100%;
    width: 100%;
    background: $background;
}

GraphDisplay {
    height: 100%;
    width: 100%;
    content-align: center middle;
}
"""

    BINDINGS: ClassVar = [
        Binding("q", "quit", "Exit"),
    ]

    def __init__(self, state_file: Path, *args, **kwargs):
        self.state_file = state_file
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(GraphDisplay(self.state_file), id="graph_container")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Modak Graph"


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "state_file",
    type=click.Path(exists=True, file_okay=True),
    default=".modak",
    required=False,
)
def queue(state_file: Path):
    run_queue_wrapper(state_file)


@cli.command()
@click.argument(
    "state_file",
    type=click.Path(exists=True, file_okay=True),
    default=".modak",
    required=False,
)
def graph(state_file: Path):
    GraphApp(Path(state_file)).run()
