"""Load secrets/.env (preferred) without printing values. Strips CR/BOM."""

from __future__ import annotations

import os
from pathlib import Path

_CANDIDATES = (
    Path("/workspace/jev-stance/secrets/.env"),
    Path(__file__).resolve().parents[2] / "secrets" / ".env",
    Path("/workspace/jev-stance/.env"),
    Path.cwd() / ".env",
)


def load_env(*, override: bool = False) -> Path | None:
    """Parse first existing env file into os.environ. Returns path used or None."""
    for path in _CANDIDATES:
        if path.is_file():
            _parse(path, override=override)
            return path
    return None


def _parse(path: Path, *, override: bool) -> None:
    text = path.read_bytes().decode("utf-8-sig")
    for raw in text.splitlines():
        line = raw.strip().strip("\r")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\r").strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val


def env(name: str, default: str = "") -> str:
    load_env()
    return (os.environ.get(name) or default).strip().strip("\r")
