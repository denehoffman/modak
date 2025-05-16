from __future__ import annotations

import time
from pathlib import Path

from modak import Task, TaskQueue


# Define a simple task that sleeps and writes output
class SleepTask(Task):
    def __init__(
        self,
        name: str,
        duration: float,
        inputs: list[Task] | None = None,
        outputs: list[Path] | None = None,
        isolated: bool = False,
        requirements: dict[str, int] | None = None,
    ):
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = [Path(f"./out/{name}.txt")]
        if requirements is None:
            requirements = {}
        super().__init__(name=name, inputs=inputs, outputs=outputs, isolated=isolated, requirements=requirements)
        self.duration = duration

    def run(self):
        self.log_info(f"Starting {self.name}...")
        time.sleep(self.duration)
        for output in self.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"{self.name} completed at {time.time()}")
        self.log_info(f"Completed {self.name}.")


class FailTask(SleepTask):
    def run(self):
        self.log_info(f"{self.name} is going to [red]fail[/red].")
        time.sleep(self.duration)
        msg = f"Task {self.name} failed [red]intentionally[/red]."
        raise RuntimeError(msg)


class LogTask(SleepTask):
    def run(self):
        for i in range(int(self.duration)):
            for j in range(100):
                self.log_info(f"Logging some info {j}")
                self.log_warning(f"Logging some warning {j}")
                if j % 10:
                    self.log_critical("Oh no, critical error!")
                time.sleep(1 / 100)


# Create demo task graph
a = SleepTask("Atest", 3)
b = SleepTask("B", 3, inputs=[a])
c = SleepTask("C", 3, inputs=[a])
d = SleepTask("Dtest", 3, inputs=[b, c], isolated=True)
e = SleepTask("Etester", 3, inputs=[b], isolated=True)
e1 = SleepTask("E1", 6, inputs=[e], requirements={"memory": 2})
e2 = SleepTask("E2", 6, inputs=[e], requirements={"memory": 3})
e3 = SleepTask("E3", 6, inputs=[e], requirements={"memory": 2})
e4 = SleepTask("E4", 6, inputs=[e], requirements={"memory": 2})
f = SleepTask("F\nhas\nmultiple\nlines", 3, inputs=[e, d])
g = LogTask("g", 3, inputs=[e])
h = SleepTask("h", 3, inputs=[g])

if __name__ == "__main__":
    queue = TaskQueue(max_workers=4, total_resources={"memory": 4})
    queue.run([f, h, e1, e2, e3, e4])
