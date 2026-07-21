"""Run output directory helpers — unique folder per prediction."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _slug(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    s = re.sub(r"_+", "_", s)
    return (s or "run")[:max_len]


def make_run_dir(
    parent: str | Path | None = None,
    graphml: str | Path | None = None,
    backend: str | None = None,
    model: str | None = None,
    label: str | None = None,
    fixed: str | Path | None = None,
) -> Path:
    """
    Create a unique output directory for one prediction run.

    - If ``fixed`` is set: use that path exactly (created if needed).
    - Else: ``{parent}/{timestamp}_{graphml_stem}_{backend}/``
      default parent = ``outputs/runs``
    """
    if fixed is not None:
        out = Path(fixed)
        out.mkdir(parents=True, exist_ok=True)
        return out.resolve()

    root = Path(parent) if parent is not None else Path("outputs/runs")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
    parts = [ts]
    if graphml is not None:
        parts.append(_slug(Path(graphml).stem))
    if backend:
        parts.append(_slug(backend, 20))
    if model:
        parts.append(_slug(Path(str(model)).name, 24))
    if label:
        parts.append(_slug(label, 24))
    out = root / "_".join(parts)
    # Extremely unlikely collision; still handle it
    if out.exists():
        n = 2
        while out.with_name(out.name + f"_{n}").exists():
            n += 1
        out = out.with_name(out.name + f"_{n}")
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def discover_graphml(paths: list[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of .graphml paths."""
    found: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() == ".graphml":
            found.append(p.resolve())
        elif p.is_dir():
            found.extend(sorted(p.rglob("*.graphml")))
        elif p.is_file():
            raise ValueError(f"Not a GraphML file: {p}")
        else:
            raise FileNotFoundError(p)
    # dedupe preserve order
    seen: set[Path] = set()
    out: list[Path] = []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out
