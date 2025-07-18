from __future__ import annotations

import base64
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import cloudpickle

from .modak import (
    TaskQueue,
    get_project_state,
    get_projects,
    reset_project,
    run_queue_wrapper,
)

if TYPE_CHECKING:
    from loguru import Logger


class Task(ABC):
    def __init__(
        self,
        name: str | None = None,
        *,
        inputs: list[Task] | None = None,
        outputs: list[Path] | None = None,
        resources: dict[str, int] | None = None,
        isolated: bool = False,
        log_file: Path | None = None,
        log_directory: Path | None = None,
        log_behavior: Literal["overwrite", "append"] = "overwrite",
    ):
        """
        Initialize a Task.

        Parameters
        ----------
        name : str, optional
            Name of the task. If None, a UUID (v4) will be generated.
        inputs : list of Task, optional
            List of input tasks that this task depends on.
        outputs : list of Path, optional
            List of output file paths this task produces.
        resources : dict of str to int, optional
            Dictionary specifying resource requirements (e.g., {"cpu": 2}).
        isolated : bool, optional
            Whether the task should be run in isolation.
        log_file : Path, optional
            Path representing the log file. Defaults to ``<name>.log``
        log_directory : Path, optional
            Path where the log file is stored. Defaults to the current working directory.
        """
        self._name = name or str(uuid.uuid4())
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._resources = resources or {}
        self._isolated = isolated
        self._log_path = (log_directory or Path.cwd()) / (
            log_file or f"{self._name}.log"
        )
        self._log_behavior = log_behavior

    @override
    def __hash__(self) -> int:
        return hash(self._name)

    @property
    def logger(self) -> Logger:
        """
        Get the Loguru global logger instance.

        Returns
        -------
        loguru.Logger
            Logger instance for this task.

        Notes
        -----
        This property is designed specifically for writing to the log file for this task.
        """
        from loguru import logger

        return logger

    @property
    def name(self) -> str:
        """
        Get the name of the task.

        Returns
        -------
        str
            The task name.
        """
        return self._name

    @property
    def inputs(self) -> list[Task]:
        """
        Get the list of input tasks.

        Returns
        -------
        list of Task
            Input dependencies of this task.
        """
        return self._inputs

    @property
    def outputs(self) -> list[Path]:
        """
        Get the list of output paths.

        Returns
        -------
        list of Path
            Output files generated by this task.
        """
        return self._outputs

    @property
    def resources(self) -> dict[str, int]:
        """
        Get the resource requirements for this task.

        Returns
        -------
        dict of str to int
            Resource requirements (e.g., {"cpu": 2}).
        """
        return self._resources

    @property
    def isolated(self) -> bool:
        """
        Get the isolation flag for this task.

        Returns
        -------
        bool
            True if the task should run in isolation, False otherwise.
        """
        return self._isolated

    @property
    def log_path(self) -> Path:
        """
        Get the log file path for this task.

        Returns
        -------
        Path
            Path to the log file.
        """

        return self._log_path

    @property
    def log_behavior(self) -> Literal["overwrite", "append"]:
        """
        Get the logging behavior for this task.

        Returns
        -------
        str
            Logging behavior ("overwrite" or "append").
        """
        return self._log_behavior

    @abstractmethod
    def run(self) -> None:
        """
        Execute the task.

        This method must be implemented by subclasses.
        """

    def serialize(self) -> str:
        """
        Serialize the task to a base64-encoded string.

        Returns
        -------
        str
            Base64-encoded string representing the serialized task.

        Notes
        -----
        This is mostly used internally to create a new process from a serialized task.
        """
        raw_bytes = cloudpickle.dumps(self)
        return base64.b64encode(raw_bytes).decode("utf-8")

    @classmethod
    def deserialize(cls, data: str) -> Task:
        """
        Deserialize a task from a base64-encoded string.

        Parameters
        ----------
        data : str
            Base64-encoded string representation of a serialized task.

        Returns
        -------
        Task
            Deserialized task instance.

        Notes
        -----
        This is mostly used internally to create a new process from a serialized task.
        """
        raw_bytes = base64.b64decode(data.encode("utf-8"))
        return cloudpickle.loads(raw_bytes)


__all__ = [
    "Task",
    "TaskQueue",
    "run_queue_wrapper",
    "get_projects",
    "get_project_state",
    "reset_project",
]
