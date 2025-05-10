from collections import defaultdict
from typing import override
from typing_extensions import Literal

from modak import Task
import numpy as np
from numpy.typing import NDArray
from itertools import permutations


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


def boxed_task(
    task: Task, *, padding: tuple[int, int, int, int] = (0, 1, 0, 1), terminal: bool = False
) -> tuple[list[str], tuple[list[int], list[int]], tuple[int, int]]:
    n_in = len(task.inputs)
    min_width = max(len(task.name), n_in)
    text = task.name.center(min_width)
    pad_top, pad_right, pad_bottom, pad_left = padding
    out = []
    out.append(f"┏{('┻' if not terminal else '━').center(len(text) + pad_left + pad_right, '━')}┓")
    out.extend([f"┃{' ' * (len(text) + pad_left + pad_right)}┃" for _ in range(pad_top)])
    out.append(f"┃{' ' * pad_left}{text}{' ' * pad_right}┃")
    out.extend([f"┃{' ' * (len(text) + pad_left + pad_right)}┃" for _ in range(pad_bottom)])
    out.append(f"┗{('┳' * n_in).center(len(text) + pad_left + pad_right, '━')}┛")
    return out, input_coords, output_coord


def boxed_layer(
    layer: list[Task],
    *,
    padding: tuple[int, int, int, int] = (0, 1, 0, 1),
    margin: tuple[int, int] = (1, 1),
    terminal: bool = False,
) -> list[str]:
    pad_top, _, pad_bottom, _ = padding
    margin_left, margin_right = margin
    out = []
    boxed_tasks = [boxed_task(task, padding=padding, terminal=terminal) for task in layer]
    out.extend(
        [
            f"{' ' * margin_left}{(' ' * (margin_left + margin_right)).join([bt[irow] for bt in boxed_tasks])}{' ' * margin_right}"
            for irow in range(pad_top + pad_bottom + 3)
        ]
    )
    return out


def print_layers(layers: list[list[Task]]):
    for i, layer in enumerate(layers):
        terminal = i == 0
        if not terminal:
            print(" TEXT HERE ")
        print("\n".join(boxed_layer(layer, terminal=terminal)))


def main():
    class SimpleTask(Task):
        def __init__(self, name: str, inputs: list[Task] | None = None):
            if inputs is None:
                inputs = []
            super().__init__(name=name, inputs=inputs)

        def run(self):
            pass

        @override
        def __repr__(self):
            return self.name

    # m = SimpleTask("--M--", inputs=[])
    # l = SimpleTask("--L--", inputs=[m])
    # k = SimpleTask("--K--", inputs=[])
    # j = SimpleTask("--J--", inputs=[])
    # i = SimpleTask("--I--", inputs=[])
    # h = SimpleTask("--H--", inputs=[m])
    # g = SimpleTask("--G--", inputs=[])
    # f = SimpleTask("--F--", inputs=[h, i, j])
    # e = SimpleTask("--E--", inputs=[])
    # d = SimpleTask("--D--", inputs=[k, l])
    # c = SimpleTask("--C--", inputs=[d])
    # b = SimpleTask("--B--", inputs=[m, e])
    # a = SimpleTask("--A--", inputs=[e, f, g])
    #
    # layers = layer_tasks([a, b, c, d, e, f, g, h, i, j, k, l, m])
    #
    m = SimpleTask("--M--", inputs=[])
    l = SimpleTask("--L--", inputs=[])
    k = SimpleTask("--K--", inputs=[])
    j = SimpleTask("--J--", inputs=[k, l])
    i = SimpleTask("--I--", inputs=[k, m])
    h = SimpleTask("--H--", inputs=[])
    g = SimpleTask("--G--", inputs=[k])
    f = SimpleTask("--F--", inputs=[])
    e = SimpleTask("--E--", inputs=[g, j])
    d = SimpleTask("--D--", inputs=[h, i, j])
    c = SimpleTask("--C--", inputs=[g])
    b = SimpleTask("--B--", inputs=[c, f])
    a = SimpleTask("--A--", inputs=[c, d, e, f])

    layers = layer_tasks([a, b, c, d, e, f, g, h, i, j, k, l, m])

    minimize_all_crossings(layers)
    print_layers(layers)


if __name__ == "__main__":
    main()
