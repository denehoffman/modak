import json
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import tzlocal
from rich.table import Table
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.events import Key
from textual.widgets import Footer, Header, Static, Tab, TabbedContent, TabPane, Tabs

from modak.graph import from_state

STATE_FILE = Path('.modak/state.json')

STATUS_ORDER = {'done': 0, 'skipped': 1, 'running': 2, 'queued': 3, 'waiting': 4, 'failed': 5, 'canceled': 6}
STATUS_COLOR = {
    'done': 'green',
    'skipped': 'cyan',
    'running': 'blue',
    'queued': 'yellow',
    'waiting': 'white',
    'failed': 'red',
    'canceled': 'magenta',
}


class QueueDisplay(Static):
    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh_table)

    def refresh_table(self) -> None:
        table = Table(title='Task Queue', expand=True, show_edge=True, header_style='bold magenta')
        table.add_column('Task')
        table.add_column('Status', style='bold')
        table.add_column('Start Time', justify='right')
        table.add_column('End Time', justify='right')

        if STATE_FILE.exists():
            with STATE_FILE.open() as f:
                state = json.load(f)
        else:
            state = {}

        def status_key(task):
            return STATUS_ORDER.get(state[task]['status'], 99)

        local_tz = tzlocal.get_localzone()

        def fmt_time(ts):
            if not ts:
                return ''
            return datetime.fromtimestamp(ts, tz=local_tz).strftime('%H:%M:%S')

        for task in sorted(state, key=status_key):
            status = state[task]['status']
            start = fmt_time(state[task].get('start_time'))
            end = fmt_time(state[task].get('end_time'))
            table.add_row(f'[bold]{task.replace("\n", "\\n")}', f'[{STATUS_COLOR[status]}]{status}', start, end)

        status_counts = {s: 0 for s in STATUS_ORDER}
        for t in state.values():
            status_counts[t['status']] += 1

        summary = ' | '.join(f'[{STATUS_COLOR[k]}]{k}: {v}' for k, v in status_counts.items() if v > 0)
        table.caption = summary
        self.update(table)


class GraphDisplay(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state: str = ''

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh_graph)

    def refresh_graph(self) -> None:
        if STATE_FILE.exists():
            new_state = STATE_FILE.read_text()
            if new_state != self.state:
                self.state = new_state
                graph = from_state(json.loads(self.state))
                self.update(graph)
        else:
            self.state = ''
            self.update('')


class StateApp(App):
    CSS = """
    #main {
        padding: 1 2;
        border: round white;
        height: 100%;
        width: 100%;
    }
    QueueDisplay {
        height: 100%;
        width: 100%;
        content-align: center middle;
    }
    """

    BINDINGS: ClassVar = [
        Binding('q', 'quit', 'Exit'),
        Binding('m', "show_tab('monitor')", 'Queue Monitor'),
        Binding('1', "show_tab('monitor')", 'Queue Monitor', show=False),
        Binding('g', "show_tab('graph')", 'Queue Graph'),
        Binding('2', "show_tab('graph')", 'Queue Graph', show=False),
        Binding('i', "show_tab('info')", 'Info'),
        Binding('3', "show_tab('info')", 'Info', show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial='monitor'):
            with TabPane('Monitor', id='monitor'):
                yield Container(QueueDisplay(), id='monitor_container')
            with TabPane('Graph', id='graph'):
                yield ScrollableContainer(GraphDisplay(), id='graph_container')
            with TabPane('Info', id='info'):
                yield Static('Modak Monitor v1.0\n\nThis is an info panel.', id='info_text')
        yield Footer()

    def action_show_tab(self, tab: str) -> None:
        self.get_child_by_type(TabbedContent).active = tab

    def on_mount(self) -> None:
        self.title = 'Modak Monitor'


if __name__ == '__main__':
    StateApp().run()
