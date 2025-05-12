import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from graphlib import TopologicalSorter
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
import unicodedata

from rich.logging import RichHandler

STATE_FILE = Path(".modak/state.json")


class TaskStatus(Enum):
    WAITING = "waiting"
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"


def slugify(value: str) -> str:
    """
    From <https://github.com/django/django/blob/825ddda26a14847c30522f4d1112fb506123420d/django/utils/text.py#L453>
    """
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


@dataclass
class Task(ABC):
    name: str
    _: KW_ONLY
    inputs: list["Task"] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    _status: TaskStatus = field(default=TaskStatus.WAITING)
    _status_lock: Lock = field(default_factory=lambda: Lock())
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self):
        self._logger = logging.getLogger(f"modak.{self.name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        log_path = Path(".modak") / f"{slugify(self.name)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RichHandler(rich_tracebacks=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(handler.formatter)
        self._logger.handlers.clear()
        self._logger.addHandler(file_handler)

    @abstractmethod
    def run(self):
        pass

    def capture_run(self):
        try:
            self.run()
        except Exception as e:
            self.log_exception(e)
            raise

    def __repr__(self):
        return f"<Task {self.name} ({self._status}) at {hex(id(self))}>"

    def log_debug(self, msg: str):
        self._logger.debug(msg)

    def log_info(self, msg: str):
        self._logger.info(msg)

    def log_warning(self, msg: str):
        self._logger.warning(msg)

    def log_error(self, msg: str):
        self._logger.error(msg)

    def log_critical(self, msg: str):
        self._logger.critical(msg)

    def log_exception(self, e: Exception):
        self._logger.exception(e, stack_info=True)

    def compute_output_hashes(self):
        hashes = {}
        for output in self.outputs:
            if output.exists():
                data = output.read_bytes()
                hashes[str(output)] = hashlib.md5(data).hexdigest()  # noqa: S324
        return hashes


class TaskQueue:
    def __init__(self, max_workers: int):
        self.tasks: dict[str, Task] = {}
        self.task_graph = TopologicalSorter()
        self.state: dict[str, dict] = {}
        self.max_workers = max_workers
        self.queue = Queue()
        self.threads: list[Thread] = []
        self.lock = Lock()

    def add_task(self, task: Task):
        self.tasks[task.name] = task
        self.task_graph.add(task.name, *[dep.name for dep in task.inputs])

    def load_state_file(self, tasks: list[Task]):
        if STATE_FILE.exists():
            with STATE_FILE.open() as f:
                state = json.load(f)
                self.state = {task: entry for task, entry in state.items() if task in tasks}
        else:
            self.state = {}

    def write_state_file(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with STATE_FILE.open("w") as f:
            json.dump(self.state, f, indent=2)

    def set_task_status(self, task: Task, status: TaskStatus):
        with task._status_lock:
            task._status = status

        with self.lock:
            entry = self.state.get(task.name, {})
            entry["status"] = status.value

            if "dependencies" not in entry:
                entry["dependencies"] = [inp.name for inp in task.inputs]

            now = time.time()
            if status == TaskStatus.RUNNING:
                entry["start_time"] = now
            elif status in {TaskStatus.DONE, TaskStatus.FAILED}:
                entry["end_time"] = now

            # Save current outputs
            entry["outputs"] = task.compute_output_hashes()

            # Save hashes of inputs for future validation
            if status == TaskStatus.DONE:
                input_hashes = {}
                for dep in task.inputs:
                    for output in dep.outputs:
                        if output.exists():
                            input_hashes[str(output)] = hashlib.md5(output.read_bytes()).hexdigest()  # noqa: S324
                entry["input_hashes"] = input_hashes

            self.state[task.name] = entry
            self.write_state_file()

    def all_deps_done(self, task: Task) -> bool:
        return all(dep._status in {TaskStatus.DONE, TaskStatus.SKIPPED} for dep in task.inputs)

    def any_deps_failed(self, task: Task) -> bool:
        return any(dep._status == TaskStatus.FAILED for dep in task.inputs)

    def outputs_valid(self, task: Task) -> bool:
        state_entry = self.state.get(task.name)
        if not state_entry:
            return False

        expected_outputs = state_entry.get("outputs", {})
        if task.outputs and not expected_outputs:
            return False

        current_outputs = task.compute_output_hashes()

        if set(expected_outputs) != set(current_outputs):
            return False
        for path_str, expected_hash in expected_outputs.items():
            if not Path(path_str).exists():
                return False
            actual_hash = hashlib.md5(Path(path_str).read_bytes()).hexdigest()  # noqa: S324
            if actual_hash != expected_hash:
                return False

        if task.inputs:
            expected_inputs = state_entry.get("input_hashes", {})
            for dep in task.inputs:
                for output in dep.outputs:
                    output_path = str(output)
                    if not output.exists():
                        return False
                    actual_hash = hashlib.md5(output.read_bytes()).hexdigest()  # noqa: S324
                    if output_path not in expected_inputs:
                        return False
                    if actual_hash != expected_inputs[output_path]:
                        return False

        return True

    def propagate_failure(self, task: Task):
        if task._status == TaskStatus.FAILED:
            return
        self.set_task_status(task, TaskStatus.FAILED)
        for t in self.tasks.values():
            if task in t.inputs:
                self.propagate_failure(t)

    def worker(self):
        while True:
            task = self.queue.get()
            if task is None:
                break
            self.set_task_status(task, TaskStatus.RUNNING)
            try:
                task.capture_run()
                self.set_task_status(task, TaskStatus.DONE)
            except Exception:  # noqa: BLE001
                self.propagate_failure(task)
            self.queue.task_done()

    def cancel_all(self):
        for task in self.tasks.values():
            if task._status in {TaskStatus.WAITING, TaskStatus.QUEUED, TaskStatus.RUNNING}:
                self.set_task_status(task, TaskStatus.CANCELED)

    def run(self, tasks: list[Task]):
        self.load_state_file(tasks)
        visited: set[str] = set()

        def collect(task: Task):
            if task.name in visited:
                return
            visited.add(task.name)
            for dep in task.inputs:
                collect(dep)
            self.add_task(task)
            self.set_task_status(task, TaskStatus.WAITING)

        for task in tasks:
            collect(task)

        self.task_graph.prepare()

        for _ in range(self.max_workers):
            thread = Thread(target=self.worker)
            thread.start()
            self.threads.append(thread)

        try:
            pending = set(self.task_graph.get_ready())
            while pending:
                for name in list(pending):
                    task = self.tasks[name]
                    if self.any_deps_failed(task):
                        self.propagate_failure(task)
                        self.task_graph.done(name)
                        pending.remove(name)
                    elif self.all_deps_done(task):
                        if self.outputs_valid(task):
                            if task._status != TaskStatus.DONE:
                                task.log_info(f"Task {task} was already completed. Skipping.")
                                self.set_task_status(task, TaskStatus.SKIPPED)
                        else:
                            self.set_task_status(task, TaskStatus.QUEUED)
                            self.queue.put(task)
                        self.task_graph.done(name)
                        pending.remove(name)
                pending.update(self.task_graph.get_ready())

            self.queue.join()
        except KeyboardInterrupt:
            self.cancel_all()
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except Exception:  # noqa: BLE001, PERF203, S110
                    pass
        finally:
            for _ in self.threads:
                self.queue.put(None)
            for t in self.threads:
                t.join()
