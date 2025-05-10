from graphlib import TopologicalSorter
from collections import defaultdict


class Task:
    def __init__(self, name, inputs=None, status="waiting"):
        self.name = name
        self.inputs = inputs or []
        self.status = status


def assign_layers(tasks):
    # Topo sort to assign depth layers
    graph = {task.name: [dep.name for dep in task.inputs] for task in tasks}
    ts = TopologicalSorter(graph)
    order = list(ts.static_order())

    depth = {}
    for name in order:
        parents = graph[name]
        depth[name] = 0 if not parents else max(depth[p] for p in parents) + 1
    return depth


def layout_graph(tasks):
    name_map = {task.name: task for task in tasks}
    depth = assign_layers(tasks)

    layers = defaultdict(list)
    for name, d in depth.items():
        layers[d].append(name)

    max_width = max(len(layer) for layer in layers.values())
    canvas = [[" " * 12 for _ in range(max_width)] for _ in range(len(layers) * 3)]

    positions = {}

    for d, names in sorted(layers.items()):
        for i, name in enumerate(names):
            row = d * 3
            col = i
            box = f"┌─{name:^8}─┐"
            mid = f"│  {name:^8}  │"
            bot = f"└──────────┘"
            canvas[row][col] = box
            canvas[row + 1][col] = mid
            canvas[row + 2][col] = bot
            positions[name] = (row, col)

    # Draw vertical connectors
    for task in tasks:
        for inp in task.inputs:
            child_row, child_col = positions[task.name]
            parent_row, parent_col = positions[inp.name]
            # draw a vertical pipe down from parent to child
            for r in range(parent_row + 3, child_row):
                canvas[r][parent_col] = "     │      "

            # draw connector line
            if parent_col == child_col:
                canvas[child_row - 1][child_col] = "     │      "
            else:
                min_col = min(parent_col, child_col)
                max_col = max(parent_col, child_col)
                canvas[child_row - 1][min_col : max_col + 1] = [
                    "────┴────" if c == child_col else "──────────" for c in range(min_col, max_col + 1)
                ]

    for row in canvas:
        print("".join(row))


# Example DAG
a = Task("A")
b = Task("B", [a])
c = Task("C", [a])
d = Task("D", [b, c])
e = Task("E", [d])
f = Task("F", [a])
g = Task("G", [e, f])

layout_graph([a, b, c, d, e, f, g])
