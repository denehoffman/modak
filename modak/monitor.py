from __future__ import annotations

import json
from pathlib import Path

import click

from modak import run_queue_wrapper
from modak.graph import from_state

from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.containers import ScrollableContainer
from rich.text import Text


def queue_graph(
    state_file: Path | str = Path(".modak"), width: int = 80, height: int = 24
) -> Text:
    try:
        state_file = Path(state_file)
        if state_file.exists():
            state = state_file.read_text()
            return Text.from_ansi(from_state(json.loads(state), width, height))
    except:
        return Text.from_markup("[red]Error[/]")


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

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="scroller"):
            yield Static(queue_graph(self.state_file), id="output")

    def on_mount(self) -> None:
        self.set_interval(1.0, self.update_view)

    def update_view(self) -> None:
        container = self.query_one("#scroller", ScrollableContainer)
        visible_width, visible_height = container.size
        virtual_width, virtual_height = container.virtual_size
        horizontal_scrollbar_visible = virtual_width > visible_width
        vertical_scrollbar_visible = virtual_height > visible_height
        stat = self.query_one("#output", Static)
        stat.update(
            queue_graph(
                self.state_file,
                visible_width - 2 * int(vertical_scrollbar_visible),
                visible_height - 2 * int(horizontal_scrollbar_visible),
            )
        )


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
    # with Live(queue_graph(state_file), auto_refresh=False) as live:
    #     time.sleep(0.5)
    #     text = queue_graph(state_file)
    #     live.update(text)
    #     live.refresh()
