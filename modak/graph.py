from __future__ import annotations

from collections import defaultdict
from itertools import chain, permutations
from typing import ClassVar, Self

import numpy as np
from textdraw import (
    BoundingBox,
    Box,
    Pixel,
    PixelGroup,
    Point,
    Style,
    TextPath,
    multipath,
    render,
)


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
            memo[name] = 1 + max(
                longest_path_from(child) for child in children[tasklike]
            )
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
        [
            [bottom_task in top_task.task_inputs for bottom_task in bottom_layer]
            for top_task in top_layer
        ],
        dtype=np.int_,
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
    return sum(
        count_crossings(top_layer, bottom_layer)
        for top_layer, bottom_layer in zip(layers[:-1], layers[1:])
    )


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
    return (i_top_min, i_bottom_min), (
        top_permutations[i_top_min],
        bottom_permutations[i_bottom_min],
    )


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
            top_permutations = (
                layer_permutations[j]
                if not down
                else [layer_permutations[j][loop_permutations[j]]]
            )
            bottom_permutations = (
                layer_permutations[j + 1]
                if down
                else [layer_permutations[j + 1][loop_permutations[j + 1]]]
            )
            layer_perm, best_layers = minimize_crossings(
                top_permutations, bottom_permutations
            )
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


class TaskBox:
    STYLES: ClassVar = {
        "pending": "dimmed",
        "queued": "yellow",
        "running": "blink blue",
        "done": "green",
        "failed": "red",
    }
    BORDER_TYPES: ClassVar = {
        "pending": "light",
        "queued": "light",
        "running": "heavy",
        "done": "heavy",
        "failed": "heavy",
    }

    def __init__(
        self,
        task_name: str,
        task_status: str,
        num_task_inputs: int,
        position: Point,
        *,
        has_output: bool = True,
    ):
        self.task_name = task_name
        self.task_status = task_status
        self.position = position
        self.num_task_inputs = num_task_inputs
        min_width = max(num_task_inputs, len(task_name)) + 4
        self.box = Box(
            task_name,
            width=min_width,
            height=3,
            position=position,
            style="bold on black",
            border_style=TaskBox.STYLES[task_status] + "on black",
            padding_style="on black",
            line_style="double",
            justify="center",
            padding=(0, 1, 0, 1),
        )
        self.has_output = has_output
        self.xy_output = self.box.bbox.top_center + Point(0, 1)
        self.xy_inputs = [
            self.box.bbox.bottom_center + Point(i - num_task_inputs // 2, -1)
            for i in range(num_task_inputs)
        ]
        barrier = Pixel(" ", weight=None)
        barriers = []
        if self.has_output:
            barriers.append(barrier.duplicate(self.xy_output - Point(1, 0)))
            barriers.append(barrier.duplicate(self.xy_output + Point(1, 0)))
        if self.num_task_inputs > 0:
            barriers.append(barrier.duplicate(self.xy_inputs[0] - Point(1, 0)))
            barriers.append(barrier.duplicate(self.xy_inputs[-1] + Point(1, 0)))
        self.barriers = PixelGroup(barriers)
        self.paths: list[TextPath] = []

    def update_status(self, new_status: str) -> None:
        self.task_status = new_status
        self.box.border_style = Style(TaskBox.STYLES[new_status] + "on black")
        for path in self.paths:
            path.style = Style(TaskBox.STYLES[new_status] + "on black")
            path.line_style = TaskBox.BORDER_TYPES[new_status]

    def duplicate_shifted(self, delta: Point) -> Self:
        tb = TaskBox(
            self.task_name,
            self.task_status,
            self.num_task_inputs,
            self.position + delta,
            has_output=self.has_output,
        )
        tb.paths = [p.duplicate_shifted(delta) for p in self.paths]
        return tb

    def all_objects(self) -> list[Box | TextPath]:
        return [self.box, *self.paths]


def render_task_layers(
    layers: list[list[TaskLike]], width: int, height: int
) -> dict[str, TaskBox]:
    task_boxes: dict[str, TaskBox] = {}
    task_dict: dict[str, TaskLike] = {
        tasklike.task_name: tasklike
        for tasklike in chain(*layers)
        if tasklike.task_status
        in {
            "pending",
            "queued",
            "running",
            "done",
            "failed",
        }
    }

    x_spacing = 4
    y_spacing = 8
    y_pos = 0
    for ilayer, layer in enumerate(layers):
        taskboxes_layer = []
        x_pos = 0
        for tasklike in [
            tl
            for tl in layer
            if tl.task_status in {"pending", "queued", "running", "done", "failed"}
        ]:
            box = TaskBox(
                tasklike.task_name,
                tasklike.task_status,
                len(tasklike.task_inputs),
                Point(x_pos, y_pos),
                has_output=ilayer == 0,  # TODO: check
            )
            x_pos += box.box.bbox.width + x_spacing
            taskboxes_layer.append(box)
        bbox = BoundingBox.wrap([tb.box for tb in taskboxes_layer])
        for taskbox in taskboxes_layer:
            task_boxes[taskbox.task_name] = taskbox.duplicate_shifted(
                Point(-bbox.bottom_center.x, 0)
            )
        y_pos -= y_spacing

    fanouts: defaultdict[str, list[tuple[Point, Point]]] = defaultdict(list)
    targets: defaultdict[str, list[str]] = defaultdict(list)

    for ilayer, layer in enumerate(layers):
        sublayers = [t for la in layers[ilayer:] for t in la]
        for tasklike in [
            tl
            for tl in layer
            if tl.task_status in {"pending", "queued", "running", "done", "failed"}
        ]:
            tgt_box = task_boxes.get(tasklike.task_name)
            if tgt_box is None:
                continue
            inputs = [t for t in sublayers if t.task_name in tasklike.task_inputs]
            for i, input_tasklike in enumerate(inputs):
                src_box = task_boxes[input_tasklike.task_name]
                start = src_box.xy_output
                end = tgt_box.xy_inputs[i]
                fanouts[input_tasklike.task_name].append((start, end))
                targets[input_tasklike.task_name].append(tasklike.task_name)
    bbox_all_boxes = BoundingBox.wrap([tb.box for tb in task_boxes.values()])
    all_paths = []
    for task_name, pairs in fanouts.items():
        starts, ends = zip(*pairs)
        path_objs = multipath(
            list(starts),
            list(ends),
            style=TaskBox.STYLES[task_dict[task_name].task_status] + "on black",
            line_style=TaskBox.BORDER_TYPES[task_dict[task_name].task_status],
            start_directions=["down"] * len(list(starts)),
            end_directions=["up"] * len(list(ends)),
            weight=2,
            bend_penalty=6,
            barriers=[
                *[tb.box for tb in task_boxes.values()],
                task_boxes[task_name].barriers,
                *[task_boxes[target].barriers for target in targets[task_name]],
            ],
            environment=all_paths,
            bbox=(
                bbox_all_boxes.top + 5,
                bbox_all_boxes.right + 5,
                bbox_all_boxes.bottom - 5,
                bbox_all_boxes.left - 5,
            ),
        )
        task_boxes[task_name].paths = path_objs
        all_paths.extend(path_objs)
    return task_boxes


class GraphRender:
    def __init__(self, state: dict[str, dict], width: int, height: int):
        self.tasklikes = [
            TaskLike(t, e["inputs"], e["status"]) for t, e in state.items()
        ]
        layers = layer_tasks(self.tasklikes)
        minimize_all_crossings(layers)
        self.width = width
        self.height = height
        self.task_boxes = render_task_layers(layers, width, height)
        self.full_box = self.get_box()

    def get_box(self) -> Box:
        all_objects = [
            obj for tb in self.task_boxes.values() for obj in tb.all_objects()
        ]
        bbox_all_objects = BoundingBox.wrap(all_objects)
        center_all_objects = bbox_all_objects.center
        origin = center_all_objects - Point(self.width // 2, self.height // 2)
        return Box(
            position=origin,
            width=max(self.width, bbox_all_objects.width),
            height=max(self.height, bbox_all_objects.height),
            style="on black",
            border_style="on black",
            line_style=None,
        )

    def update(self, state: dict[str, dict], width: int, height: int):
        new_tasklikes = [
            TaskLike(t, e["inputs"], e["status"]) for t, e in state.items()
        ]
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self.full_box = self.get_box()

        for tasklike in new_tasklikes:
            if self.task_boxes[tasklike.task_name].task_status != tasklike.task_status:
                self.task_boxes[tasklike.task_name].update_status(tasklike.task_status)

    def render(self) -> str:
        return render(
            [
                self.full_box,
                *[obj for tb in self.task_boxes.values() for obj in tb.all_objects()],
            ],
            default_style="on black",
        )
