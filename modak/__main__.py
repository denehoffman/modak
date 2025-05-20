import sys
import traceback

from loguru import logger

from modak import Task


def main():
    try:
        task = Task.deserialize(sys.argv[1])
    except Exception:  # noqa: BLE001
        print("Failed to deserialize task:", file=sys.stderr)  # noqa: T201
        traceback.print_exc()
        sys.exit(2)
    logger.remove()
    logger.add(
        task.log_path, level="DEBUG", enqueue=True, backtrace=True, diagnose=True
    )
    try:
        task.run()
    except Exception:  # noqa: BLE001
        logger.exception("Task execution failed:")
        traceback.print_exc()
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
