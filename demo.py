from __future__ import annotations

import time
from pathlib import Path

from modak import Task, TaskQueue


# Define a simple task that sleeps and writes output
class SleepTask(Task):
    def __init__(self, name: str, duration: float, inputs: list[Task] | None = None, outputs: list[Path] | None = None):
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = [Path(f"./out/{name}.txt")]
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.duration = duration

    def run(self):
        print(f"Starting {self.name}...")
        time.sleep(self.duration)
        for output in self.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"{self.name} completed at {time.time()}")
        print(f"Completed {self.name}.")


class FailTask(SleepTask):
    def run(self):
        print(f"{self.name} is going to fail.")
        time.sleep(self.duration)
        raise RuntimeError(f"Task {self.name} failed intentionally.")


# Create demo task graph
a = SleepTask("A", 3)
b = SleepTask("B", 3, inputs=[a])
c = FailTask("C", 3, inputs=[a])  # <- will fail
d = SleepTask("D", 3, inputs=[b, c])  # <- will fail due to C
e = SleepTask("E", 3, inputs=[b])  # <- will succeed
f = SleepTask("F", 3, inputs=[e, d])  # <- will fail due to D

if __name__ == "__main__":
    queue = TaskQueue(max_workers=4)
    queue.run([f])
