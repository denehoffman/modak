from __future__ import annotations

import json
from pathlib import Path

import click
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static

from modak import run_queue_wrapper
from modak.graph import GraphRender


class RenderApp(App):
    CSS = """
    #scroller {
        overflow-x: auto;
        overflow-y: auto;
        align: center middle;
    }

    #output {
        width: auto;
        min-width: 100%;
        text-align: center
    }
    """

    def __init__(self, state_file: Path | str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_file = Path(state_file)
        self.graph_render: GraphRender | None = None

    def read_state(self) -> dict[str, dict] | None:
        state_file = Path(self.state_file)
        try:
            if state_file.exists():
                state = state_file.read_text()
                return json.loads(state)
        except Exception:
            return None

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="scroller"):
            yield Static("", id="output")

    def on_mount(self) -> None:
        self.set_interval(1.0, self.update_view)

    def update_view(self) -> None:
        container = self.query_one("#scroller", ScrollableContainer)
        visible_width, visible_height = container.size
        virtual_width, virtual_height = container.virtual_size
        horizontal_scrollbar_visible = virtual_width > visible_width
        vertical_scrollbar_visible = virtual_height > visible_height
        stat = self.query_one("#output", Static)
        state = self.read_state()
        if state is None:
            stat.update("[red]Error reading state file[/red]")
            return
        if self.graph_render is None:
            self.graph_render = GraphRender(
                state,
                visible_width - 2 * int(vertical_scrollbar_visible),
                visible_height - 2 * int(horizontal_scrollbar_visible),
            )
        else:
            self.graph_render.update(
                state,
                visible_width - 2 * int(vertical_scrollbar_visible),
                visible_height - 2 * int(horizontal_scrollbar_visible),
            )
        stat.update(Text.from_ansi(self.graph_render.render()))


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
    RenderApp(state_file).run()
