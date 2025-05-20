<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <h1 align="center">modak</h1>
</p>
<p align="center">
    <img alt="GitHub Release" src="https://img.shields.io/github/v/release/denehoffman/modak?style=for-the-badge&logo=github"></a>
  <a href="https://github.com/denehoffman/modak/commits/main/" alt="Latest Commits">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/denehoffman/modak?style=for-the-badge&logo=github"></a>
  <a href="LICENSE-APACHE" alt="License">
    <img alt="GitHub License" src="https://img.shields.io/github/license/denehoffman/modak?style=for-the-badge"></a>
  <a href="https://pypi.org/project/modak/" alt="View project on PyPI">
  <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/modak?style=for-the-badge&logo=python&logoColor=yellow&labelColor=blue"></a>
</p>

`modak` is a simple-to-use, opinionated task queue system with dependency
management, resource allocation, and isolation control. Tasks are run
respecting topological dependencies, resource limits, and optional isolation.

This library only has two classes, `Task`s, which are an abstract class with a
single method to override, `run(self) -> None`, and a `TaskQueue` which manages
the execution order. Additionally, `modak` comes with a task monitor TUI which
can be invoked with the `modak` shell command.

The `TaskQueue` has been written in Rust to get past issues with parallelism
and the GIL. Instead of using a thread pool or even a multiprocessing pool,
the tasks are serialized into bytes and passed to the Rust-side manager, which
handles dispatching and execution. Each task is then run as a separate subprocess
spawned in a Rust thread. This means the only way to share state between tasks is
by writing to an output file and having a task depend on that file.

By default, `modak` scripts will create a state file called `.modak` in the
current working directory. This can be changed by setting it in the `TaskQueue`'s
initialization method. The `modak` CLI also supports an optional argument to
point to the location of the state file.

## Features

- Topological task scheduling
- Persistent state and log files
- Resource-aware execution
- Isolated task handling
- Skipping of previously completed tasks

## Installation

```shell
pip install modak
```

Or with `uv`:

```shell
pip install modak
```

## FAQ

> Q: What do you mean by "opinionated"?

A: The library is meant to do one thing (and hopefully do it well): run tasks
and write output files. Some users might want more flexibility, like writing
to a database or having a target that isn't written to at all, but that is
not a goal of this library. If you need this level of control, try [`airflow`](https://airflow.apache.org/)
or [`luigi`](https://github.com/spotify/luigi).

> Q: Why make another task manager?

A: [`luigi`](https://github.com/spotify/luigi) is nice, but I've been annoyed by
the poor type hints for task parameters. It's also very confusing for
first-time users, and has a lot of features that I don't really think people
use unless they are working with products like Spotify. I built `modak` with
research pipelines in mind, so I wanted something that was so simple to use,
you don't have to think too hard about what you're doing and can focus on
the data instead. I haven't used [airflow](https://airflow.apache.org/) much,
but it also seems like a tool intended for enterprise. My goal here is
simplicity and a minimal learning curve. There are only two classes. `luigi`
has the added annoyance of running a web server to visualize the state of the
DAG, which is very tricky to use on a remote server if you don't have the
proper permissions.

> Q: Isn't Rust a bit overkill?

A: Rust isn't as scary as it sounds. I don't actually care much about memory
safety (although I'll take it for free), I like the development experience.

> Q: Any sharp corners?

A: In development, I've found that libraries that do something when imported
need to be handled with care. Such libraries should be imported inside the
`run` method of the task. This is because the task gets serialized and sent
to the `__main__` module, but the imports from your code are run before
serialization. An example of this is the `loguru` library, which sets
up the global logger [on import](https://github.com/Delgan/loguru/blob/a69bfc451413f71b81761a238db4b5833cf0a992/loguru/__init__.py#L18).
If `loguru` is only imported outside the task, the `logger` instance will have
no sink added because [these lines](https://github.com/Delgan/loguru/blob/a69bfc451413f71b81761a238db4b5833cf0a992/loguru/__init__.py#L31-L32)
will not be run when the task is deserialized. This will not effect most code,
it's just something to be aware of.

## Examples

### A simple chain of tasks

```python
from modak import Task, TaskQueue

class PrintTask(Task):
    def run(self):
        self.logger.info(f"Running {self.name}")

t1 = PrintTask(name="task1")
t2 = PrintTask(name="task2", inputs=[t1])
t3 = PrintTask(name="task3", inputs=[t2])

queue = TaskQueue()
queue.run([t3])
```

### Fan-in, fan-out

```python
from pathlib import Path
from modak import Task, TaskQueue

class DummyTask(Task):
    def run(self):
        self.logger.info(f"Running {self.name}")
        for output in self.outputs:
            output.write_text(f"Output of {self.name}")

# Leaf tasks
a = DummyTask(name="A", outputs=[Path("a.out")])
b = DummyTask(name="B", outputs=[Path("b.out")])
c = DummyTask(name="C", outputs=[Path("c.out")])

# Fan-in: D depends on A, B, C
d = DummyTask(name="D", inputs=[a, b, c], outputs=[Path("d.out")])

# Fan-out: E and F both depend on D
e = DummyTask(name="E", inputs=[d], outputs=[Path("e.out")])
f = DummyTask(name="F", inputs=[d], outputs=[Path("f.out")])

queue = TaskQueue()
queue.run([e, f])

```

### A complex workflow

```python
from pathlib import Path
from modak import Task, TaskQueue

class SimTask(Task):
    def run(self):
        self.logger.info(f"{self.name} starting with {self.requirements}")
        for out in self.outputs:
            out.write_text(f"Generated by {self.name}")

# Raw data preprocessing
pre_a = SimTask(name="PreA", outputs=[Path("a.pre")], requirements={"cpu": 1})
pre_b = SimTask(name="PreB", outputs=[Path("b.pre")], requirements={"cpu": 1})
pre_c = SimTask(name="PreC", outputs=[Path("c.pre")], requirements={"cpu": 1})

# Feature extraction (can run in parallel)
feat1 = SimTask(name="Feature1", inputs=[pre_a], outputs=[Path("a.feat")], requirements={"cpu": 2})
feat2 = SimTask(name="Feature2", inputs=[pre_b], outputs=[Path("b.feat")], requirements={"cpu": 2})
feat3 = SimTask(name="Feature3", inputs=[pre_c], outputs=[Path("c.feat")], requirements={"cpu": 2})

# Aggregation step
aggregate = SimTask(
    name="Aggregate",
    inputs=[feat1, feat2, feat3],
    outputs=[Path("agg.out")],
    requirements={"cpu": 3}
)

# Final model training (expensive, must be isolated)
train = SimTask(
    name="TrainModel",
    inputs=[aggregate],
    outputs=[Path("model.bin")],
    isolated=True,
    requirements={"cpu": 3, "gpu": 1}
)

# Side analysis and visualization can run independently
viz = SimTask(name="Visualization", inputs=[feat1, feat2], outputs=[Path("viz.png")], requirements={"cpu": 1})
stats = SimTask(name="Stats", inputs=[feat3], outputs=[Path("stats.txt")], requirements={"cpu": 1})

queue = TaskQueue(
    workers=4,
    resources={"cpu": 4, "gpu": 1}
)

queue.run([train, viz, stats])

```

## Future Plans

I'll probably make small improvements to the TUI and add features as I find the
need. Contributions are welcome, just open an issue or pull request on GitHub
and I'll try to respond as soon as I can.
