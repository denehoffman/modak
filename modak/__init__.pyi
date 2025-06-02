from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

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
    ): ...
    @property
    def logger(self) -> Logger: ...
    @property
    def name(self) -> str: ...
    @property
    def inputs(self) -> list[Task]: ...
    @property
    def outputs(self) -> list[Path]: ...
    @property
    def resources(self) -> dict[str, int]: ...
    @property
    def isolated(self) -> bool: ...
    @property
    def log_path(self) -> Path: ...
    @abstractmethod
    def run(self) -> None: ...
    def serialize(self) -> str: ...
    @classmethod
    def deserialize(cls, data: str) -> Task: ...

class TaskQueue:
    def __init__(
        self,
        *,
        workers: int = 4,
        resources: dict[str, int] | None = None,
        state_file_path: Path | None = None,
        log_path: Path | None = None,
    ) -> None: ...
    def run(self, tasks: Sequence[Task]) -> None: ...

def run_queue_wrapper(state_file_path: Path) -> None: ...

__all__ = ["Task", "TaskQueue", "run_queue_wrapper"]
