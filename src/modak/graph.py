from __future__ import annotations

from collections import defaultdict
from itertools import chain, permutations
from typing import override

import numpy as np

from modak import Task, TaskStatus
from modak.text import TextObject, TextPanel, TextBox, BorderType


def layer_tasks(tasks: list[Task]) -> list[list[Task]]:
    name_to_task: dict[str, Task] = {}

    def collect(task: Task):
        if task.name in name_to_task:
            return
        name_to_task[task.name] = task
        for input_task in task.inputs:
            collect(input_task)

    for task in tasks:
        collect(task)

    children: dict[str, list[Task]] = defaultdict(list)
    for task in name_to_task.values():
        for input_task in task.inputs:
            children[input_task.name].append(task)

    memo: dict[str, int] = {}

    def longest_path_from(task: Task) -> int:
        if task.name in memo:
            return memo[task.name]
        if task.name not in children or not children[task.name]:
            memo[task.name] = 0
        else:
            memo[task.name] = 1 + max(longest_path_from(child) for child in children[task.name])
        return memo[task.name]

    for task in name_to_task.values():
        longest_path_from(task)

    depth_to_tasks: dict[int, list[Task]] = defaultdict(list)
    for task in name_to_task.values():
        depth = memo[task.name]
        depth_to_tasks[depth].append(task)

    return [depth_to_tasks[d] for d in sorted(depth_to_tasks)]


def count_crossings(top_layer: list[Task], bottom_layer: list[Task]) -> int:
    matrix = np.array(
        [[bottom_task in top_task.inputs for bottom_task in bottom_layer] for top_task in top_layer], dtype=np.int_
    )
    p, q = matrix.shape
    count = 0
    for j in range(p - 1):
        for k in range(j + 1, p):
            for a in range(q - 1):
                for b in range(a + 1, q):
                    count += matrix[j, b] * matrix[k, a]
    return count


def count_all_crossings(layers: list[list[Task]]) -> int:
    return sum(count_crossings(top_layer, bottom_layer) for top_layer, bottom_layer in zip(layers[:-1], layers[1:]))


def minimize_crossings(
    top_permutations: list[list[Task]], bottom_permutations: list[list[Task]]
) -> tuple[tuple[int, int], tuple[list[Task], list[Task]]]:
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


def minimize_all_crossings(layers: list[list[Task]], max_iters=10):
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


def render_task_layers(layers: list[list[Task]]) -> TextPanel:
    panel = TextPanel()
    task_positions: dict[str, tuple[int, int]] = {}
    task_boxes: dict[str, TextPanel] = {}
    task_dict: dict[str, Task] = {task.name: task for task in chain(*layers)}
    styles = {
        TaskStatus.WAITING: 'dim',
        TaskStatus.RUNNING: 'blue',
        TaskStatus.DONE: 'green',
        TaskStatus.FAILED: 'red',
        TaskStatus.SKIPPED: 'cyan',
        TaskStatus.QUEUED: 'yellow',
        TaskStatus.CANCELED: 'magenta',
    }
    linestyles = {
        TaskStatus.WAITING: BorderType.LIGHT,
        TaskStatus.RUNNING: BorderType.HEAVY,
        TaskStatus.DONE: BorderType.HEAVY,
        TaskStatus.FAILED: BorderType.HEAVY,
        TaskStatus.SKIPPED: BorderType.HEAVY,
        TaskStatus.QUEUED: BorderType.LIGHT,
        TaskStatus.CANCELED: BorderType.LIGHT,
    }

    y_spacing = 10
    for layer_idx, layer in enumerate(layers):
        x_offset = 0
        x_length = 0
        for task in layer:
            x_length += len(task.name) + 4 + 4

        for task in layer:
            task_panel = TextPanel()
            min_width = max(len(task.inputs), len(task.name))
            diff = min_width - len(task.name)
            box = TextBox.from_string(
                task.name,
                border_type=BorderType.DOUBLE,
                justify='center',
                padding=(0, 1 + diff // 2, 0, 1 + diff // 2),
                border_style=styles[task.status],
            )

            barrier = TextObject.from_string(' ')
            task_panel.add_object(box, 0, 0)
            task_panel.add_object(barrier, box.width // 2 - 1, -1)
            task_panel.add_object(barrier, box.width // 2 + 1, -1)

            num_inputs = len(task.inputs)
            if num_inputs > 0:
                input_diff = box.width - num_inputs
                task_panel.add_object(barrier, input_diff // 2 - 1, box.height)
                task_panel.add_object(barrier, (input_diff // 2 + num_inputs), box.height)

            task_panel.penalty_group = 'box'
            task_boxes[task.name] = task_panel
            x = x_offset - x_length // 2
            x_offset += box.width + 4
            y = 2 + layer_idx * y_spacing
            task_positions[task.name] = (x, y)
            panel.add_object(task_panel, x, y)

    fanouts: defaultdict[str, list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)

    for ilayer, layer in enumerate(layers):
        sublayers = [t for la in layers[ilayer:] for t in la]
        for task in layer:
            tx, ty = task_positions[task.name]
            tgt_box = task_boxes[task.name]

            inputs = [t for t in sublayers if t in task.inputs]
            num_inputs = len(inputs)

            for i, input_task in enumerate(inputs):
                sx, sy = task_positions[input_task.name]
                src_box = task_boxes[input_task.name]

                start_x = sx + src_box.width // 2
                start_y = sy - 1

                t_offset_x = i - num_inputs // 2 + tgt_box.width // 2
                end_x = tx + t_offset_x
                end_y = ty + tgt_box.height - 2

                fanouts[input_task.name].append(((start_x, start_y), (end_x, end_y)))

    for task_name, pairs in fanouts.items():
        starts, ends = zip(*pairs)
        path_obj = panel.connect_many(
            list(starts),
            list(ends),
            style=styles[task_dict[task_name].status],
            border_type=linestyles[task_dict[task_name].status],
            bend_penalty=0,
            group_penalties={'box': 1000, 'line': 60},
        )
        path_obj.penalty_group = 'line'
        panel.add_object(path_obj, 0, 0)

    return panel


def main():
    class SimpleTask(Task):
        def __init__(self, name: str, inputs: list[Task] | None = None, status: TaskStatus = TaskStatus.WAITING):
            if inputs is None:
                inputs = []
            super().__init__(name=name, inputs=inputs)
            self.status = status

        def run(self):
            pass

        @override
        def __repr__(self):
            return self.name

    m = SimpleTask('M', inputs=[], status=TaskStatus.FAILED)
    l = SimpleTask('L', inputs=[], status=TaskStatus.QUEUED)
    k = SimpleTask('K', inputs=[], status=TaskStatus.DONE)
    j = SimpleTask('J', inputs=[k, l])
    i = SimpleTask('I', inputs=[k, m], status=TaskStatus.FAILED)
    h = SimpleTask('H', inputs=[], status=TaskStatus.QUEUED)
    g = SimpleTask('G', inputs=[k], status=TaskStatus.RUNNING)
    f = SimpleTask('F', inputs=[], status=TaskStatus.QUEUED)
    e = SimpleTask('E', inputs=[g, j])
    d = SimpleTask('D', inputs=[h, i, j], status=TaskStatus.FAILED)
    c = SimpleTask('C', inputs=[g])
    b = SimpleTask('B', inputs=[c, f])
    a = SimpleTask('A', inputs=[c, d, e, f, l], status=TaskStatus.FAILED)

    layers = layer_tasks([a, b, c, d, e, f, g, h, i, j, k, l, m])

    minimize_all_crossings(layers)
    panel = render_task_layers(layers)
    from rich import print

    print(panel)


if __name__ == '__main__':
    main()
