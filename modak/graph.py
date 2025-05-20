from __future__ import annotations

from collections import defaultdict
from itertools import chain, permutations
from typing import ClassVar, override

import numpy as np
from textdraw import BorderType, StyledChar, TextBox, TextObject, TextPanel


class TaskLike:
    def __init__(self, task_name: str, task_inputs: list[str], task_status: str):
        self.task_name = task_name
        self.task_inputs = task_inputs
        self.task_status = task_status

    def __repr__(self) -> str:
        return f"{self.task_name} ({self.task_inputs})"


def layer_tasks(tasks: list[TaskLike]) -> list[list[TaskLike]]:
    name_to_tasklike: dict[str, TaskLike] = {t.task_name: t for t in tasks}

    children: dict[TaskLike, list[TaskLike]] = defaultdict(list)
    for tasklike in name_to_tasklike.values():
        for input_task_name in tasklike.task_inputs:
            children[name_to_tasklike[input_task_name]].append(tasklike)

    memo: dict[str, int] = {}

    def longest_path_from(tasklike: TaskLike) -> int:
        name = tasklike.task_name
        if name in memo:
            return memo[name]
        if tasklike not in children or not children[tasklike]:
            memo[name] = 0
        else:
            memo[name] = 1 + max(longest_path_from(child) for child in children[tasklike])
        return memo[name]

    for tasklike in name_to_tasklike.values():
        longest_path_from(tasklike)

    depth_to_tasks: dict[int, list[TaskLike]] = defaultdict(list)
    for tasklike in name_to_tasklike.values():
        depth = memo[tasklike.task_name]
        depth_to_tasks[depth].append(tasklike)

    return [depth_to_tasks[d] for d in sorted(depth_to_tasks)]


def count_crossings(top_layer: list[TaskLike], bottom_layer: list[TaskLike]) -> int:
    matrix = np.array(
        [[bottom_task in top_task.task_inputs for bottom_task in bottom_layer] for top_task in top_layer], dtype=np.int_
    )
    p, q = matrix.shape
    count = 0
    for j in range(p - 1):
        for k in range(j + 1, p):
            for a in range(q - 1):
                for b in range(a + 1, q):
                    count += matrix[j, b] * matrix[k, a]
    return count


def count_all_crossings(layers: list[list[TaskLike]]) -> int:
    return sum(count_crossings(top_layer, bottom_layer) for top_layer, bottom_layer in zip(layers[:-1], layers[1:]))


def minimize_crossings(
    top_permutations: list[list[TaskLike]], bottom_permutations: list[list[TaskLike]]
) -> tuple[tuple[int, int], tuple[list[TaskLike], list[TaskLike]]]:
    min_crossings = count_crossings(top_permutations[0], bottom_permutations[0])
    i_top_min = 0
    i_bottom_min = 0
    for i_top, top_perm in enumerate(top_permutations):
        for i_bottom, bottom_perm in enumerate(bottom_permutations):
            crossings = count_crossings(top_perm, bottom_perm)
            if crossings < min_crossings:
                min_crossings = crossings
                i_top_min = i_top
                i_bottom_min = i_bottom
    return (i_top_min, i_bottom_min), (top_permutations[i_top_min], bottom_permutations[i_bottom_min])


def minimize_all_crossings(layers: list[list[TaskLike]], max_iters=10):
    minimized = False
    i = 0
    down = True
    best_permutations = tuple([0] * len(layers))
    past_permutations: set[tuple[int, ...]] = {best_permutations}
    best_min_crossings: int | None = None
    layer_permutations = [[list(p) for p in permutations(layer)] for layer in layers]
    while i < max_iters and not minimized:
        loop_permutations = [0] * len(layers)
        js = list(range(len(layers) - 1))
        if not down:
            js.reverse()
        for j in js:
            top_permutations = layer_permutations[j] if not down else [layer_permutations[j][loop_permutations[j]]]
            bottom_permutations = (
                layer_permutations[j + 1] if down else [layer_permutations[j + 1][loop_permutations[j + 1]]]
            )
            layer_perm, best_layers = minimize_crossings(top_permutations, bottom_permutations)
            if not down:
                loop_permutations[j] = layer_perm[0]
                layers[j] = best_layers[0]
            if down:
                loop_permutations[j + 1] = layer_perm[1]
                layers[j + 1] = best_layers[1]
        total_min_crossings = count_all_crossings(layers)
        if best_min_crossings is None or total_min_crossings <= best_min_crossings:
            best_permutations = tuple(loop_permutations)
            best_min_crossings = total_min_crossings
            if best_permutations in past_permutations:
                minimized = True
            past_permutations.add(best_permutations)
        down = not down
        i += 1


class TaskBox(TextObject):
    STYLES: ClassVar = {
        "pending": "dim",
        "queued": "yellow",
        "running": "blue",
        "done": "green",
        "failed": "red",
        "skipped": "cyan",
        "canceled": "magenta",
    }
    BORDER_TYPES: ClassVar = {
        "pending": BorderType.LIGHT,
        "queued": BorderType.LIGHT,
        "running": BorderType.HEAVY,
        "done": BorderType.HEAVY,
        "failed": BorderType.HEAVY,
        "skipped": BorderType.HEAVY,
        "canceled": BorderType.LIGHT,
    }

    def __init__(
        self, task_name: str, task_status: str, num_task_inputs: int, x: int, y: int, *, has_output: bool = True
    ):
        super().__init__(penalty_group="taskbox")
        self.num_task_inputs = num_task_inputs
        min_width = max(num_task_inputs, len(task_name))
        diff = min_width - len(task_name)
        self.box = TextBox.from_string(
            task_name,
            border_style=TaskBox.STYLES[task_status],
            style="bold",
            border_type=BorderType.DOUBLE,
            padding=(0, 1 + diff // 2, 0, 1 + diff // 2),
        )
        self.x = x
        self.y = y
        self.has_output = has_output
        self.xy_output = (x + self.box.width // 2, y - 1)
        self.xy_inputs = [
            (x + i - num_task_inputs // 2 + self.box.width // 2, y + self.box.height) for i in range(num_task_inputs)
        ]
        self.barrier = TextObject.from_string(" ")

    @property
    @override
    def chars(self) -> list[StyledChar]:
        panel = TextPanel([(self.box, self.x, self.y)])
        if self.has_output:
            panel.add_object(self.barrier, self.xy_output[0] - 1, self.xy_output[1])
            panel.add_object(self.barrier, self.xy_output[0] + 1, self.xy_output[1])
        if self.num_task_inputs > 0:
            panel.add_object(self.barrier, self.xy_inputs[0][0] - 1, self.xy_inputs[0][1])
            panel.add_object(self.barrier, self.xy_inputs[-1][0] + 1, self.xy_inputs[-1][1])
        return panel.chars


def render_task_layers(layers: list[list[TaskLike]]) -> TextPanel:
    panel = TextPanel()
    # task_positions: dict[str, tuple[int, int]] = {}
    task_boxes: dict[str, TaskBox] = {}
    task_dict: dict[str, TaskLike] = {tasklike.task_name: tasklike for tasklike in chain(*layers)}

    x_spacing = 4
    y_spacing = 10
    for layer_idx, layer in enumerate(layers):
        x_offset = 0
        x_length = 0
        x_length = sum(
            [
                max([len(task_line) for task_line in tasklike.task_name.split("\n")]) + 2 * x_spacing
                for tasklike in layer
            ]
        )
        for tasklike in layer:
            task_box = TaskBox(
                tasklike.task_name,
                tasklike.task_status,
                len(tasklike.task_inputs),
                x_offset - x_length // 2,
                2 + layer_idx * y_spacing,
            )
            task_boxes[tasklike.task_name] = task_box
            x_offset += task_box.width + x_spacing
            panel.add_object(task_box)

    fanouts: defaultdict[str, list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)

    for ilayer, layer in enumerate(layers):
        sublayers = [t for la in layers[ilayer:] for t in la]
        for tasklike in layer:
            tgt_box = task_boxes[tasklike.task_name]
            inputs = [t for t in sublayers if t.task_name in tasklike.task_inputs]
            for i, input_tasklike in enumerate(inputs):
                src_box = task_boxes[input_tasklike.task_name]
                start_x, start_y = src_box.xy_output
                end_x, end_y = tgt_box.xy_inputs[i]
                fanouts[input_tasklike.task_name].append(((start_x, start_y), (end_x, end_y)))

    for task_name, pairs in fanouts.items():
        starts, ends = zip(*pairs)
        path_obj = panel.connect_many(
            list(starts),
            list(ends),
            style=TaskBox.STYLES[task_dict[task_name].task_status],
            border_type=TaskBox.BORDER_TYPES[task_dict[task_name].task_status],
            group_penalties={"taskbox": 1000, "line": 10},
            start_char="",
            end_char="▲",
        )
        path_obj.penalty_group = "line"
        panel.add_object(path_obj)

    return panel


def from_state(state: dict[str, dict]) -> TextPanel:
    tasklikes: list[TaskLike] = []
    for task, entry in state.items():
        tasklikes.append(TaskLike(task, entry["inputs"], entry["status"]))
    layers = layer_tasks(tasklikes)
    minimize_all_crossings(layers)
    return render_task_layers(layers)
