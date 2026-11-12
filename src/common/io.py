"""Safe artifact publication and content-addressed provenance."""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any


class PipelineError(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    return fingerprint({p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.py"))})


def staging_directory(destination: Path) -> Path:
    if destination.exists():
        raise PipelineError("FAILED", f"Refusing to overwrite existing artifact directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex[:10]}")
    temporary.mkdir()
    return temporary


def run_process(command: list[str], log: Path | None = None, timeout: float = 3600) -> dict:
    """Use argument lists, check exit codes and retain complete stdout/stderr."""
    import time

    started = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(str(exc), encoding="utf-8")
        raise PipelineError("FAILED", f"Cannot execute {command[0]}: {exc}") from exc
    output = result.stdout + result.stderr
    report = {"command": command, "returncode": result.returncode, "runtime_seconds": time.perf_counter() - started, "output": output}
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(output, encoding="utf-8")
        write_json(log.with_suffix(".command.json"), {k: v for k, v in report.items() if k != "output"})
    if result.returncode:
        raise PipelineError("FAILED", f"Command failed ({result.returncode}); log={log}\n{output[-2500:]}")
    return report

