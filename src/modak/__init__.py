from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from graphlib import TopologicalSorter
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

__all__ = ["Task", "TaskQueue"]

MODAK_DIR = Path(os.getenv("MODAK_PATH") or ".modak")
STATE_FILE = MODAK_DIR / Path("state.json")


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
    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


@dataclass
class Task(ABC):
    """
    Abstract base class representing a unit of work in the task queue.

    Attributes
    ----------
    name : str
        Name of the task.
    inputs : list of Task
        Tasks that this task depends on.
    outputs : list of Path
        Files produced by this task.
    isolated : bool
        If True, the task will only run by itself.
    requirements : dict of str to int
        Resource requirements for the task.
    """

    name: str
    _: KW_ONLY
    inputs: list[Task] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    isolated: bool = False
    requirements: dict[str, int] = field(default_factory=dict)
    _status: TaskStatus = field(init=False, default=TaskStatus.WAITING)
    _status_lock: Lock = field(init=False, default_factory=lambda: Lock())
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self):
        self._logger = logging.getLogger(f"modak.{self.name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        log_path = MODAK_DIR / f"{slugify(self.name)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        self._logger.handlers.clear()
        self._logger.addHandler(file_handler)

    @abstractmethod
    def run(self):
        """
        Run the task's main logic. Must be implemented by subclasses.

        Raises
        ------
        Exception
            If the task encounters a problem during execution.
        """

    def _capture_run(self):
        try:
            self.run()
        except Exception as e:
            self.log_exception(e)
            raise

    def __repr__(self):
        return f"<Task {self.name} ({self._status}) at {hex(id(self))}>"

    def log_debug(self, msg: str):
        """
        Log a debug-level message for this task.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        self._logger.debug(msg)

    def log_info(self, msg: str):
        """
        Log an info-level message for this task.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        self._logger.info(msg)

    def log_warning(self, msg: str):
        """
        Log a warning-level message for this task.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        self._logger.warning(msg)

    def log_error(self, msg: str):
        """
        Log an error-level message for this task.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        self._logger.error(msg)

    def log_critical(self, msg: str):
        """
        Log a critical-level message for this task.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        self._logger.critical(msg)

    def log_exception(self, e: Exception):
        """
        Log an exception with traceback for this task.

        Parameters
        ----------
        e : Exception
            The exception to log.
        """
        self._logger.exception(e, stack_info=True)

    def _compute_output_hashes(self):
        hashes = {}
        for output in self.outputs:
            if output.exists():
                data = output.read_bytes()
                hashes[str(output)] = hashlib.md5(data).hexdigest()  # noqa: S324
        return hashes


class TaskQueue:
    """
    Manages and executes a set of tasks with dependencies, resources, and isolation constraints.

    Parameters
    ----------
    workers : int, optional
        Maximum number of concurrent worker threads (default is 4).
    resources : dict of str to int, optional
        Total available resources to allocate to tasks.
    """

    def __init__(self, workers: int = 4, resources: dict[str, int] | None = None):
        self.tasks: dict[str, Task] = {}
        self.task_graph = TopologicalSorter()
        self.state: dict[str, dict] = {}
        self.workers = workers
        self.queue = Queue()
        self.threads: list[Thread] = []
        self.lock = Lock()
        self.resource_lock = Lock()
        self.total_resources = resources.copy() if resources else {}
        self.available_resources = self.total_resources.copy()
        self.isolated_running = False
        self.ready_tasks: list[Task] = []

    def add_task(self, task: Task):
        """
        Add a task to the queue and update the dependency graph.

        Parameters
        ----------
        task : Task
            The task to be added.
        """
        self.tasks[task.name] = task
        self.task_graph.add(task.name, *[dep.name for dep in task.inputs])

    def _load_state_file(self, tasks: list[Task]):
        if STATE_FILE.exists():
            with STATE_FILE.open() as f:
                state = json.load(f)
                self.state = {
                    task: entry for task, entry in state.items() if task in tasks
                }
        else:
            self.state = {}

    def _write_state_file(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with STATE_FILE.open("w") as f:
            json.dump(self.state, f, indent=2)

    def _set_task_status(self, task: Task, status: TaskStatus):
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
            entry["outputs"] = task._compute_output_hashes()

            # Save hashes of inputs for future validation
            if status == TaskStatus.DONE:
                input_hashes = {}
                for dep in task.inputs:
                    for output in dep.outputs:
                        if output.exists():
                            input_hashes[str(output)] = hashlib.md5(
                                output.read_bytes()
                            ).hexdigest()  # noqa: S324
                entry["input_hashes"] = input_hashes

            self.state[task.name] = entry
            self._write_state_file()

    def _all_deps_done(self, task: Task) -> bool:
        return all(
            dep._status in {TaskStatus.DONE, TaskStatus.SKIPPED} for dep in task.inputs
        )

    def _any_deps_failed(self, task: Task) -> bool:
        return any(dep._status == TaskStatus.FAILED for dep in task.inputs)

    def _outputs_valid(self, task: Task) -> bool:  # noqa: PLR0911
        state_entry = self.state.get(task.name)
        if not state_entry:
            return False

        expected_outputs = state_entry.get("outputs", {})
        if task.outputs and not expected_outputs:
            return False

        current_outputs = task._compute_output_hashes()

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

    def _propagate_failure(self, task: Task):
        if task._status == TaskStatus.FAILED:
            return
        self._set_task_status(task, TaskStatus.FAILED)
        for t in self.tasks.values():
            if task in t.inputs:
                self._propagate_failure(t)

    def _cancel_all(self):
        for task in self.tasks.values():
            if task._status in {
                TaskStatus.WAITING,
                TaskStatus.QUEUED,
                TaskStatus.RUNNING,
            }:
                self._set_task_status(task, TaskStatus.CANCELED)

    def _can_run(self, task: Task) -> bool:
        with self.resource_lock:
            return all(
                task.requirements[k] <= self.available_resources.get(k, 0)
                for k in task.requirements
            )

    def _allocate_resources(self, task: Task):
        with self.resource_lock:
            for k, v in task.requirements.items():
                self.available_resources[k] -= v

    def _release_resources(self, task: Task):
        with self.resource_lock:
            for k, v in task.requirements.items():
                self.available_resources[k] += v

    def _validate_requirements(self):
        for task in self.tasks.values():
            for k, v in task.requirements.items():
                if v > self.total_resources.get(k, 0):
                    task.log_critical(
                        "Task requires more resources than are available."
                    )
                    self._propagate_failure(task)

    def _worker(self):
        while True:
            task = self.queue.get()
            if task is None:
                break

            self._set_task_status(task, TaskStatus.RUNNING)
            try:
                task._capture_run()
                self._set_task_status(task, TaskStatus.DONE)
            except Exception:  # noqa: BLE001
                self._propagate_failure(task)
            self._release_resources(task)
            if task.isolated:
                with self.lock:
                    self.isolated_running = False
            self.queue.task_done()

    def run(self, tasks: list[Task]):  # noqa: PLR0912, PLR0915
        """
        Execute a list of tasks, managing dependencies and resource constraints.

        Parameters
        ----------
        tasks : list of Task
            List of root tasks to execute. Dependencies will be resolved automatically
            using input/output file hashes to determine when tasks can be skipped.
        """
        self._load_state_file(tasks)
        visited: set[str] = set()

        def collect(task: Task):
            if task.name in visited:
                return
            visited.add(task.name)
            for dep in task.inputs:
                collect(dep)
            self.add_task(task)
            self._set_task_status(task, TaskStatus.WAITING)

        for task in tasks:
            collect(task)

        self.task_graph.prepare()
        self._validate_requirements()

        for _ in range(self.workers):
            thread = Thread(target=self._worker)
            thread.start()
            self.threads.append(thread)

        try:
            pending = set(self.task_graph.get_ready())
            while (
                pending
                or any(t._status == TaskStatus.QUEUED for t in self.tasks.values())
                or not self.queue.empty()
            ):
                # Main dispatcher logic
                with self.lock:
                    for task in list(self.ready_tasks):
                        if task._status != TaskStatus.QUEUED:
                            self.ready_tasks.remove(task)
                            continue
                        if task.isolated and self.isolated_running:
                            continue
                        if not self._can_run(task):
                            continue
                        self._allocate_resources(task)
                        if task.isolated:
                            self.isolated_running = True
                        self.ready_tasks.remove(task)
                        self.queue.put(task)

                # Topological scheduling
                for name in list(pending):
                    task = self.tasks[name]
                    if self._any_deps_failed(task):
                        self._propagate_failure(task)
                        self.task_graph.done(name)
                        pending.remove(name)
                    elif self._all_deps_done(task):
                        if self._outputs_valid(task):
                            if task._status != TaskStatus.DONE:
                                task.log_info(
                                    f"Task {task} was already completed. Skipping."
                                )
                                self._set_task_status(task, TaskStatus.SKIPPED)
                        else:
                            self._set_task_status(task, TaskStatus.QUEUED)
                            self.ready_tasks.append(task)
                        self.task_graph.done(name)
                        pending.remove(name)
                pending.update(self.task_graph.get_ready())

                time.sleep(0.1)

            self.queue.join()
        except KeyboardInterrupt:
            self._cancel_all()
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
