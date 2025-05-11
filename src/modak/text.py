from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from typing import Literal, Union, override

from rich.text import Text


class BorderType(Enum):
    LIGHT = 'light'
    HEAVY = 'heavy'
    DOUBLE = 'double'


BORDER_CHARS = {
    BorderType.LIGHT: {
        'hor': '─',
        'ver': '│',
        'dl': '┐',
        'dr': '┌',
        'ul': '┘',
        'ur': '└',
        't': '┴',
        'b': '┬',
        'l': '┤',
        'r': '├',
        'x': '┼',
    },
    BorderType.HEAVY: {
        'hor': '━',
        'ver': '┃',
        'dl': '┓',
        'dr': '┏',
        'ul': '┛',
        'ur': '┗',
        't': '┻',
        'b': '┳',
        'l': '┫',
        'r': '┣',
        'x': '╋',
    },
    BorderType.DOUBLE: {
        'hor': '═',
        'ver': '║',
        'dl': '╗',
        'dr': '╔',
        'ul': '╝',
        'ur': '╚',
        't': '╩',
        'b': '╦',
        'l': '╣',
        'r': '╠',
        'x': '╬',
    },
}


@dataclass
class StyledChar:
    char: str
    style: str
    x: int
    y: int


@dataclass(frozen=True)
class Left:
    length: int


@dataclass(frozen=True)
class Right:
    length: int


@dataclass(frozen=True)
class Up:
    length: int


@dataclass(frozen=True)
class Down:
    length: int


Segment = Union[Left, Right, Up, Down]


class AbstractTextObject(ABC):
    def __init__(self, *, penalty_group: str | None = None):
        self.penalty_group = penalty_group

    @property
    @abstractmethod
    def chars(self) -> list[StyledChar]: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    @property
    @abstractmethod
    def width(self) -> int: ...


class TextObject(AbstractTextObject):
    def __init__(self):
        self._chars: list[StyledChar] = []
        self.penalty_group: str | None = None

    @property
    @override
    def chars(self) -> list[StyledChar]:
        return self._chars

    def with_penalty_group(self, group: str) -> TextObject:
        self.penalty_group = group
        return self

    @classmethod
    def from_string(cls, text: str, *, style: str = '', transparent: bool = False) -> TextObject:
        obj = cls()
        lines = text.splitlines()
        max_width = max(len(line) for line in lines)

        for y, line in enumerate(lines):
            padded = line.ljust(max_width)
            for x, char in enumerate(padded):
                if transparent and char == ' ':
                    continue
                obj.add_char(char, x, y, style=style)
        return obj

    @classmethod
    def from_path(  # noqa: PLR0912
        cls,
        segments: list[Segment],
        *,
        border_type: BorderType = BorderType.LIGHT,
        style: str = '',
        start_char: str | None = None,
        start_style: str | None = None,
        end_char: str | None = None,
        end_style: str | None = None,
    ) -> TextObject:
        obj = cls()
        x = y = 0
        visited: list[tuple[int, int]] = []

        for seg in segments:
            if isinstance(seg, Left):
                for _ in range(seg.length):
                    visited.append((x, y))
                    x -= 1
            elif isinstance(seg, Right):
                for _ in range(seg.length):
                    visited.append((x, y))
                    x += 1
            elif isinstance(seg, Up):
                for _ in range(seg.length):
                    visited.append((x, y))
                    y -= 1
            elif isinstance(seg, Down):
                for _ in range(seg.length):
                    visited.append((x, y))
                    y += 1
        visited.append((x, y))

        for i, (x, y) in enumerate(visited):
            if i == 0 and start_char:
                obj.add_char(start_char, x, y, style=start_style or style)
            elif i == len(visited) - 1 and end_char:
                obj.add_char(end_char, x, y, style=end_style or style)
            else:
                obj.add_char('#', x, y, style=style)

        obj.merge_path_intersections(border_type)
        return obj

    def merge_path_intersections(self, border_type: BorderType):
        chars = BORDER_CHARS[border_type]
        path_map: dict[tuple[int, int], set[str]] = {}

        for c in self.chars:
            if c.char in chars.values() or c.char == '#':
                x, y = c.x, c.y
                for dx, dy, direction in [(-1, 0, 'left'), (1, 0, 'right'), (0, -1, 'up'), (0, 1, 'down')]:
                    if any(
                        (c2.x, c2.y) == (x + dx, y + dy) and (c2.char in chars.values() or c2.char == '#')
                        for c2 in self.chars
                    ):
                        path_map.setdefault((x, y), set()).add(direction)

        conn_map = {
            frozenset(['left']): chars['hor'],
            frozenset(['right']): chars['hor'],
            frozenset(['left', 'right']): chars['hor'],
            frozenset(['up']): chars['ver'],
            frozenset(['down']): chars['ver'],
            frozenset(['up', 'down']): chars['ver'],
            frozenset(['down', 'right']): chars['dr'],
            frozenset(['down', 'left']): chars['dl'],
            frozenset(['up', 'right']): chars['ur'],
            frozenset(['up', 'left']): chars['ul'],
            frozenset(['left', 'right', 'up']): chars['t'],
            frozenset(['left', 'right', 'down']): chars['b'],
            frozenset(['up', 'down', 'right']): chars['r'],
            frozenset(['up', 'down', 'left']): chars['l'],
            frozenset(['up', 'down', 'left', 'right']): chars['x'],
        }

        new_chars = []
        seen = {(c.x, c.y): c for c in self.chars}
        for (x, y), dirs in path_map.items():
            new_char = conn_map.get(frozenset(dirs), chars['x'])
            style = seen.get((x, y), StyledChar('', '', x, y)).style
            new_chars.append(StyledChar(new_char, style, x, y))

        self._chars = [c for c in self.chars if (c.x, c.y) not in path_map] + new_chars

    @property
    @override
    def width(self) -> int:
        if not self.chars:
            return 0
        return max(c.x for c in self.chars) - min(c.x for c in self.chars) + 1

    @property
    @override
    def height(self) -> int:
        if not self.chars:
            return 0
        return max(c.y for c in self.chars) - min(c.y for c in self.chars) + 1

    def add_char(self, char: str, x: int, y: int, *, style: str = ''):
        self.chars.append(StyledChar(char, style, x, y))

    def __rich__(self):
        if not self.chars:
            return ''
        min_x = min(c.x for c in self.chars)
        min_y = min(c.y for c in self.chars)
        shifted = [StyledChar(c.char, c.style, c.x - min_x, c.y - min_y) for c in self.chars]
        width = max(c.x for c in shifted) + 1
        height = max(c.y for c in shifted) + 1
        grid = [[Text(' ') for _ in range(width)] for _ in range(height)]
        for c in shifted:
            grid[c.y][c.x] = Text(c.char, style=c.style)
        return Text('\n').join(Text().join(row) for row in grid)


class TextPanel(AbstractTextObject):
    def __init__(self):
        self.objects: list[tuple[AbstractTextObject, int, int]] = []

    def add_object(self, obj: AbstractTextObject, x_offset: int, y_offset: int):
        self.objects.append((obj, x_offset, y_offset))

    @property
    @override
    def chars(self) -> list[StyledChar]:
        all_chars = []
        for obj, dx, dy in self.objects:
            all_chars.extend([StyledChar(c.char, c.style, c.x + dx, c.y + dy) for c in obj.chars])
        return all_chars

    @property
    @override
    def width(self) -> int:
        if not self.objects:
            return 0
        all_x = [c.x + dx for obj, dx, _ in self.objects for c in obj.chars]
        return max(all_x) - min(all_x) + 1 if all_x else 0

    @property
    @override
    def height(self) -> int:
        if not self.objects:
            return 0
        all_y = [c.y + dy for obj, _, dy in self.objects for c in obj.chars]
        return max(all_y) - min(all_y) + 1 if all_y else 0

    def __rich__(self):
        all_chars = []
        for obj, dx, dy in self.objects:
            all_chars.extend([StyledChar(c.char, c.style, c.x + dx, c.y + dy) for c in obj.chars])
        if not all_chars:
            return ''
        min_x = min(c.x for c in all_chars)
        min_y = min(c.y for c in all_chars)
        shifted = [StyledChar(c.char, c.style, c.x - min_x, c.y - min_y) for c in all_chars]
        width = max(c.x for c in shifted) + 1
        height = max(c.y for c in shifted) + 1
        grid = [[Text(' ') for _ in range(width)] for _ in range(height)]
        for c in shifted:
            grid[c.y][c.x] = Text(c.char, style=c.style)
        return Text('\n').join(Text().join(row) for row in grid)

    def find_path(  # noqa: PLR0912, PLR0915
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        bend_penalty: int = 1,
        group_penalties: dict[str, int] | None = None,
    ) -> tuple[list[Segment], int, int, int]:
        owner_map: dict[tuple[int, int], AbstractTextObject] = {}
        for obj, dx, dy in self.objects:
            for c in obj.chars:
                pos = (c.x + dx, c.y + dy)
                owner_map[pos] = obj

        def cost(pos, prev_dir=None, new_dir=None):
            group = (owner_map[pos].penalty_group if pos in owner_map else None) if pos in owner_map else None
            base = 20
            if group_penalties and group in group_penalties:
                base = group_penalties[group]
            bend = bend_penalty if prev_dir and prev_dir != new_dir else 0
            return base + bend

        frontier = [(0, start, (0, 0))]
        came_from = {start: ((0, 0), (0, 0))}
        cost_so_far = {start: 0}

        while frontier:
            _, current, prev_dir = heapq.heappop(frontier)
            if current == end:
                break
            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (x + dx, y + dy)
                direction = (dx, dy)
                new_cost = cost_so_far[current] + cost(next_pos, prev_dir, direction)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + abs(end[0] - next_pos[0]) + abs(end[1] - next_pos[1])
                    heapq.heappush(frontier, (priority, next_pos, direction))
                    came_from[next_pos] = (current, direction)

        path = []
        cur = end
        while cur != start:
            path.append(cur)
            cur = came_from.get(cur, (None,))[0]
            if cur is None:
                msg = 'No path found'
                raise ValueError(msg)
        path.append(start)
        path.reverse()

        segments = []
        i = 0
        while i < len(path) - 1:
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            dx, dy = x1 - x0, y1 - y0
            count = 1
            while i + count < len(path):
                x2, y2 = path[i + count]
                if (x2 - x1, y2 - y1) != (dx, dy):
                    break
                count += 1
                x1, y1 = x2, y2
            if dx == 1:
                segments.append(Right(count))
            elif dx == -1:
                segments.append(Left(count))
            elif dy == 1:
                segments.append(Down(count))
            elif dy == -1:
                segments.append(Up(count))
            i += count

        x0, y0 = path[0]
        return segments, x0, y0, cost_so_far[end]

    def connect(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        border_type: BorderType = BorderType.LIGHT,
        style: str = '',
        start_char: str | None = None,
        start_style: str | None = None,
        end_char: str | None = None,
        end_style: str | None = None,
        bend_penalty: int = 1,
        group_penalties: dict[str, int] | None = None,
    ) -> TextObject:
        segments, x0, y0, _ = self.find_path(start, end, bend_penalty=bend_penalty, group_penalties=group_penalties)
        obj = TextObject.from_path(
            segments,
            border_type=border_type,
            style=style,
            start_char=start_char,
            start_style=start_style,
            end_char=end_char,
            end_style=end_style,
        )
        for c in obj.chars:
            c.x += x0
            c.y += y0
        return obj

    def connect_many(
        self,
        starts: list[tuple[int, int]],
        ends: list[tuple[int, int]],
        border_type: BorderType = BorderType.LIGHT,
        style: str = '',
        start_char: str | None = None,
        start_style: str | None = None,
        end_char: str | None = None,
        end_style: str | None = None,
        bend_penalty: int = 1,
        group_penalties: dict[str, int] | None = None,
        merge_penalty_group: str = '_mergepath',
    ) -> TextObject:
        assert len(starts) == len(ends)
        best_obj = None
        min_total_cost = float('inf')
        path_pairs = list(zip(starts, ends))

        for ordering in permutations(path_pairs):
            temp_panel = TextPanel()
            temp_panel.objects = list(self.objects)  # shallow copy
            paths = []
            total_cost = 0

            for s, e in ordering:
                segments, x0, y0, cost = temp_panel.find_path(
                    s,
                    e,
                    bend_penalty=bend_penalty,
                    group_penalties={**(group_penalties or {}), merge_penalty_group: 0},
                )
                total_cost += cost

                path_obj = TextObject.from_path(
                    segments,
                    border_type=border_type,
                    style=style,
                    start_char=start_char,
                    start_style=start_style,
                    end_char=end_char,
                    end_style=end_style,
                )
                for c in path_obj.chars:
                    c.x += x0
                    c.y += y0
                path_obj.penalty_group = merge_penalty_group
                temp_panel.add_object(path_obj, 0, 0)
                paths.append(path_obj)

            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_obj = TextObject()
                for p in paths:
                    best_obj.chars.extend(p.chars)
                best_obj.merge_path_intersections(border_type)

        assert best_obj is not None
        return best_obj


class TextBox(TextObject):
    def __init__(
        self,
        content: TextObject,
        *,
        border_style: str = '',
        border_type: BorderType = BorderType.LIGHT,
        padding: tuple[int, int, int, int] = (0, 1, 0, 1),
    ):
        super().__init__()

        top_pad, right_pad, bottom_pad, left_pad = padding
        content_min_x = min((c.x for c in content.chars), default=0)
        content_min_y = min((c.y for c in content.chars), default=0)

        # Shift and copy content into box interior
        for c in content.chars:
            self.add_char(
                c.char,
                c.x - content_min_x + 1 + left_pad,
                c.y - content_min_y + 1 + top_pad,
                style=c.style,
            )

        inner_width = content.width + left_pad + right_pad
        inner_height = content.height + top_pad + bottom_pad
        box_width = inner_width + 2
        box_height = inner_height + 2

        chars = BORDER_CHARS[border_type]

        for x in range(box_width):
            self.add_char(chars['hor'], x, 0, style=border_style)
            self.add_char(chars['hor'], x, box_height - 1, style=border_style)
        for y in range(box_height):
            self.add_char(chars['ver'], 0, y, style=border_style)
            self.add_char(chars['ver'], box_width - 1, y, style=border_style)

        self.add_char(chars['dr'], 0, 0, style=border_style)
        self.add_char(chars['dl'], box_width - 1, 0, style=border_style)
        self.add_char(chars['ur'], 0, box_height - 1, style=border_style)
        self.add_char(chars['ul'], box_width - 1, box_height - 1, style=border_style)

    @classmethod
    def from_string(
        cls,
        text: str,
        *,
        border_style: str = '',
        style: str = '',
        border_type: BorderType = BorderType.LIGHT,
        padding: tuple[int, int, int, int] = (0, 1, 0, 1),
        justify: Literal['left', 'center', 'right'] = 'center',
        transparent: bool = False,
    ) -> TextBox:
        lines = text.splitlines()
        max_line_length = max((len(line) for line in lines), default=0)
        aligned_lines = []

        for line in lines:
            if justify == 'left':
                text = line.ljust(max_line_length)
            elif justify == 'center':
                text = line.center(max_line_length)
            else:
                text = line.rjust(max_line_length)
            aligned_lines.append(text)

        padded_lines = aligned_lines or ['']
        text_obj = TextObject.from_string('\n'.join(padded_lines), style=style, transparent=transparent)

        return cls(text_obj, border_style=border_style, border_type=border_type, padding=padding)


if __name__ == '__main__':
    from rich import print
    from itertools import groupby

    panel = TextPanel()

    def bottom_center(box, x, y):
        return (x + box.width // 2, y + box.height)

    def top_center(box, x, y):
        return (x + box.width // 2, y - 1)

    # Define node rows
    rows = [
        ['A', 'B', 'C'],
        ['D', 'E', 'F', 'G'],
        ['H', 'I', 'J', 'K'],
    ]

    # Position and create boxes
    boxes = {}
    positions = {}
    x_spacing = 12
    y_spacing = 12

    for row_index, row in enumerate(rows):
        for col_index, label in enumerate(row):
            x = 4 + col_index * x_spacing
            y = 2 + row_index * y_spacing
            box = TextBox.from_string(label, border_type=BorderType.DOUBLE)
            box.penalty_group = 'box'
            boxes[label] = box
            positions[label] = (x, y)

            cx_bot = x + box.width // 2
            cy_bot = y + box.height
            cx_top = cx_bot
            cy_top = y - 1

            for dx in (-1, 1):
                panel.add_object(
                    TextObject.from_string(' ', style='on black').with_penalty_group('box'), cx_bot + dx, cy_bot
                )
                panel.add_object(
                    TextObject.from_string(' ', style='on black').with_penalty_group('box'), cx_top + dx, cy_top
                )
            panel.add_object(box, x, y)

    # Original directed edges
    edges = {
        'H': ['D', 'E', 'F'],
        'I': ['E'],
        'J': ['A', 'G'],
        'K': ['G'],
        'D': ['A', 'B'],
        'E': ['A'],
        'F': ['B'],
        'G': ['B'],
    }

    # Group edges by identical target sets
    sorted_edges = sorted(edges.items(), key=lambda x: tuple(sorted(x[1])))
    groups: list[set[str]] = []

    for _, group in groupby(sorted_edges, key=lambda x: tuple(sorted(x[1]))):
        group_sources = {source for source, _ in group}
        groups.append(group_sources)

    # Build merged paths for each group
    for group in groups:
        starts = []
        ends = []
        seen_edges = set()

        for source in group:
            sx, sy = positions[source]
            for target in edges.get(source, []):
                edge = (source, target)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                tx, ty = positions[target]
                starts.append(top_center(boxes[source], sx, sy))
                ends.append(bottom_center(boxes[target], tx, ty))

        path_obj = panel.connect_many(
            starts,
            ends,
            style='yellow',
            border_type=BorderType.LIGHT,
            bend_penalty=1,
            group_penalties={'box': 100, 'line': 60},
        )
        path_obj.penalty_group = 'line'
        panel.add_object(path_obj, 0, 0)

    print(panel)
