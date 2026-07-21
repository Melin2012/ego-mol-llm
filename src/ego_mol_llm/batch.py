"""Batch prediction over many GraphML networks."""

from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ego_mol_llm.paths import discover_graphml, make_run_dir
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.report import export_report


@dataclass
class BatchItemResult:
    graphml: str
    ok: bool
    out_dir: str | None = None
    smiles: str | None = None
    name: str | None = None
    formula: str | None = None
    confidence: float | None = None
    mass_ok: bool | None = None
    source: str | None = None
    seed_mz: float | None = None
    true_seed_name: str | None = None
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def run_batch(
    inputs: list[Path],
    *,
    backend: str = "dry-run",
    model: str = "chemdfm-8b",
    out_parent: Path | None = None,
    seed_id: str | None = None,
    hide_seed_name: bool = True,
    max_neighbors: int = 25,
    include_two_hop: bool = True,
    load_in_4bit: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
    mass_tol_da: float = 0.05,
    progress: Callable[[int, int, Path], None] | None = None,
) -> tuple[list[BatchItemResult], Path]:
    """
    Run predictions on all GraphML files.

    Returns (results, batch_summary_dir).
    Each item gets its own unique run folder under out_parent.
    """
    files = discover_graphml(inputs)
    if not files:
        raise FileNotFoundError("No .graphml files found")

    parent = Path(out_parent) if out_parent else Path("outputs/runs")
    batch_root = make_run_dir(
        parent=parent,
        graphml="batch",
        backend=backend,
        model=model,
        label=f"n{len(files)}",
    )

    results: list[BatchItemResult] = []
    for i, gpath in enumerate(files, start=1):
        if progress:
            progress(i, len(files), gpath)
        run_dir = make_run_dir(
            parent=batch_root,
            graphml=gpath,
            backend=backend,
            model=model,
        )
        try:
            result = predict_from_graphml(
                graphml_path=gpath,
                backend=backend,
                model=model,
                seed_id=seed_id,
                hide_seed_name=hide_seed_name,
                max_neighbors=max_neighbors,
                include_two_hop=include_two_hop,
                load_in_4bit=load_in_4bit,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                mass_tol_da=mass_tol_da,
            )
            export_report(result, run_dir)
            d = result.to_dict()
            results.append(
                BatchItemResult(
                    graphml=str(gpath),
                    ok=True,
                    out_dir=str(run_dir),
                    smiles=d.get("smiles"),
                    name=d.get("name"),
                    formula=d.get("formula"),
                    confidence=d.get("confidence"),
                    mass_ok=d.get("mass_ok"),
                    source=d.get("source"),
                    seed_mz=d.get("seed_mz"),
                    true_seed_name=d.get("true_seed_name"),
                    detail=d,
                )
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            (run_dir / "error.txt").write_text(
                err + "\n\n" + traceback.format_exc(), encoding="utf-8"
            )
            results.append(
                BatchItemResult(
                    graphml=str(gpath),
                    ok=False,
                    out_dir=str(run_dir),
                    error=err,
                )
            )

    _write_batch_summary(batch_root, results, backend=backend, model=model)
    return results, batch_root


def _write_batch_summary(
    batch_root: Path,
    results: list[BatchItemResult],
    backend: str,
    model: str,
) -> None:
    rows = [
        {
            "graphml": r.graphml,
            "ok": r.ok,
            "out_dir": r.out_dir,
            "smiles": r.smiles,
            "name": r.name,
            "formula": r.formula,
            "confidence": r.confidence,
            "mass_ok": r.mass_ok,
            "source": r.source,
            "seed_mz": r.seed_mz,
            "true_seed_name": r.true_seed_name,
            "error": r.error,
        }
        for r in results
    ]
    (batch_root / "batch_summary.json").write_text(
        json.dumps(
            {"backend": backend, "model": model, "n": len(rows), "results": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = batch_root / "batch_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "graphml",
                "ok",
                "name",
                "smiles",
                "formula",
                "confidence",
                "mass_ok",
                "source",
                "seed_mz",
                "true_seed_name",
                "out_dir",
                "error",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    ok_n = sum(1 for r in results if r.ok)
    md = [
        "# Batch summary",
        "",
        f"- Backend: `{backend}`",
        f"- Model: `{model}`",
        f"- Total: **{len(results)}**",
        f"- OK: **{ok_n}**",
        f"- Failed: **{len(results) - ok_n}**",
        "",
        "| # | File | OK | Name | SMILES | conf | mass_ok | source |",
        "|---|------|----|------|--------|------|---------|--------|",
    ]
    for i, r in enumerate(results, 1):
        stem = Path(r.graphml).name
        smi = (r.smiles or "")[:36]
        nm = (r.name or "")[:40].replace("|", "/")
        md.append(
            f"| {i} | `{stem}` | {r.ok} | {nm} | `{smi}` | {r.confidence} | {r.mass_ok} | {r.source} |"
        )
    (batch_root / "batch_summary.md").write_text("\n".join(md), encoding="utf-8")
