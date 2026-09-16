"""A process deadline, not a promise of quality or coverage."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .storage import atomic_json


class Deadline:
    def __init__(self, seconds: float, *, reserve: float = 5.0):
        if seconds <= 0 or not 0 <= reserve < seconds:
            raise ValueError("invalid runtime budget")
        self.end = time.monotonic() + seconds
        self.reserve = reserve

    def remaining(self) -> float:
        return max(0.0, self.end - time.monotonic() - self.reserve)

    def require(self, estimate=0.0):
        if self.remaining() <= estimate:
            raise TimeoutError("construction_budget_exhausted")


def supervise(command: list[str], out: Path, *, seconds=3600.0) -> int:
    """Kill the worker process group at the deadline; retain atomic checkpoints.

    The caller creates the output directory and lock before invoking this. Only
    the worker handles model requests, so termination cancels recursive work too.
    """
    if seconds <= 0:
        raise ValueError("deadline must be positive")
    start = time.monotonic()
    with (out / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
        try:
            code = process.wait(timeout=max(0.001, seconds - min(5.0, seconds * 0.1)))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            graph_file = out / "hypergraph.json"
            snapshot = (
                json.loads(graph_file.read_text())["snapshot_id"] if graph_file.exists() else None
            )
            progress_file = out / "progress.json"
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
            atomic_json(
                out / "status.json",
                {
                    "status": "partial",
                    "snapshot_id": snapshot,
                    "quality_passed": None,
                    "gaps": [
                        {
                            "reason": "budget_exhausted",
                            "pending": progress.get(
                                "pending", "unprocessed_video_or_identity_work"
                            ),
                        }
                    ],
                    "elapsed_seconds": time.monotonic() - start,
                },
            )
            return 124
    if code:
        atomic_json(
            out / "failure.json", {"exit_code": code, "elapsed_seconds": time.monotonic() - start}
        )
    return code


def worker_command(*args):
    return [sys.executable, "-m", "rrt_echo.cli", "_worker", *map(str, args)]
