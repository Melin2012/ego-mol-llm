"""Command-line interface for ego-mol-llm."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ego_mol_llm.batch import run_batch
from ego_mol_llm.paths import make_run_dir
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.report import export_report

app = typer.Typer(
    name="ego-mol-llm",
    help="Blind structure prediction from MS/MS molecular-network ego neighborhoods.",
    add_completion=False,
)
console = Console()


def _print_prediction_table(d: dict) -> None:
    table = Table(title="Prediction", show_header=True, header_style="bold magenta")
    table.add_column("Field")
    table.add_column("Value")
    for k in [
        "smiles",
        "smiles_valid",
        "name",
        "formula",
        "adduct",
        "confidence",
        "mass_ok",
        "mass_error_da",
        "source",
        "parse_mode",
        "seed_mz",
        "msms_used",
    ]:
        table.add_row(k, str(d.get(k)))
    console.print(table)
    if d.get("msms_used") and d.get("spectral"):
        spec = d["spectral"]
        console.print(
            f"[cyan]MS/MS:[/cyan] seed peaks={spec.get('seed_n_peaks')} "
            f"diag={list((spec.get('seed_diagnostics') or {}).keys())} "
            f"neighbor spectra={len(spec.get('neighbor_msms_cosine') or {})}"
        )


@app.command()
def predict(
    graphml: Path = typer.Argument(..., exists=True, help="Path to GraphML molecular network"),
    backend: str = typer.Option(
        "dry-run",
        "--backend",
        "-b",
        help="dry-run | transformers | ollama | openai",
    ),
    model: str = typer.Option(
        "chemdfm-8b",
        "--model",
        "-m",
        help="Preset or Ollama/HF model id (e.g. chemdfm-v2-14b, qwen2.5:14b)",
    ),
    seed_id: Optional[str] = typer.Option(None, "--seed-id", help="Center node id (default: auto/'0')"),
    seed_name: Optional[str] = typer.Option(None, "--seed-name", help="Substring match for seed name"),
    show_seed_name: bool = typer.Option(
        False, "--show-seed-name", help="Do NOT hide seed library name (evaluation leak)"
    ),
    max_neighbors: int = typer.Option(25, "--max-neighbors"),
    no_two_hop: bool = typer.Option(False, "--no-two-hop"),
    out: Path = typer.Option(
        Path("outputs/runs"),
        "--out",
        "-o",
        help="Parent directory; a NEW unique run folder is created under it each time",
    ),
    fixed_out: Optional[Path] = typer.Option(
        None,
        "--fixed-out",
        help="Exact output directory (no auto timestamp). Overwrites files in place.",
    ),
    temperature: float = typer.Option(0.2, "--temperature"),
    max_new_tokens: int = typer.Option(1024, "--max-new-tokens"),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    mass_tol: float = typer.Option(0.05, "--mass-tol", help="Mass match tolerance in Da"),
    mgf: Optional[List[Path]] = typer.Option(
        None,
        "--mgf",
        help="MGF file(s) with MS/MS for network nodes (NETWORK_NODE_ID or PEPMASS). Repeatable.",
    ),
    seed_mgf: Optional[Path] = typer.Option(
        None,
        "--seed-mgf",
        help="Optional MGF with the query/seed spectrum only (e.g. Ego_MSMS.mgf)",
    ),
):
    """Predict structure of an unknown center node from GraphML ego network (+ optional MGF)."""
    run_dir = make_run_dir(
        parent=out,
        graphml=graphml,
        backend=backend,
        model=model,
        fixed=fixed_out,
    )
    mgf_note = ""
    if mgf or seed_mgf:
        mgf_note = f"\nmgf={list(mgf or [])}\nseed_mgf={seed_mgf}"
    console.print(
        Panel.fit(
            f"[bold]ego-mol-llm[/bold]\nbackend={backend} model={model}\n"
            f"input={graphml}\nout={run_dir}{mgf_note}",
            border_style="cyan",
        )
    )
    result = predict_from_graphml(
        graphml_path=graphml,
        backend=backend,
        model=model,
        seed_id=seed_id,
        seed_name_contains=seed_name,
        hide_seed_name=not show_seed_name,
        max_neighbors=max_neighbors,
        include_two_hop=not no_two_hop,
        load_in_4bit=load_in_4bit,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        mass_tol_da=mass_tol,
        mgf_paths=list(mgf) if mgf else None,
        seed_mgf=seed_mgf,
    )
    paths = export_report(result, run_dir)
    d = result.to_dict()
    _print_prediction_table(d)
    console.print(f"[green]Wrote report to[/green] {run_dir}")
    for k, p in paths.items():
        console.print(f"  - {k}: {p}")


@app.command()
def batch(
    inputs: List[Path] = typer.Argument(
        ...,
        help="GraphML file(s) and/or directories containing .graphml",
    ),
    backend: str = typer.Option("dry-run", "--backend", "-b"),
    model: str = typer.Option("chemdfm-8b", "--model", "-m"),
    out: Path = typer.Option(
        Path("outputs/runs"),
        "--out",
        "-o",
        help="Parent directory for the batch folder (unique each time)",
    ),
    seed_id: Optional[str] = typer.Option(None, "--seed-id"),
    show_seed_name: bool = typer.Option(False, "--show-seed-name"),
    max_neighbors: int = typer.Option(25, "--max-neighbors"),
    no_two_hop: bool = typer.Option(False, "--no-two-hop"),
    temperature: float = typer.Option(0.2, "--temperature"),
    max_new_tokens: int = typer.Option(1024, "--max-new-tokens"),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    mass_tol: float = typer.Option(0.05, "--mass-tol"),
):
    """Batch-predict many GraphML networks; each file gets its own run folder."""
    console.print(Panel.fit(f"[bold]batch[/bold] backend={backend} model={model}", border_style="cyan"))

    def progress(i: int, n: int, path: Path) -> None:
        console.print(f"[cyan]({i}/{n})[/cyan] {path.name}")

    results, batch_root = run_batch(
        list(inputs),
        backend=backend,
        model=model,
        out_parent=out,
        seed_id=seed_id,
        hide_seed_name=not show_seed_name,
        max_neighbors=max_neighbors,
        include_two_hop=not no_two_hop,
        load_in_4bit=load_in_4bit,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        mass_tol_da=mass_tol,
        progress=progress,
    )
    ok = sum(1 for r in results if r.ok)
    table = Table(title=f"Batch results ({ok}/{len(results)} ok)")
    table.add_column("File")
    table.add_column("OK")
    table.add_column("SMILES")
    table.add_column("conf")
    table.add_column("source")
    for r in results:
        table.add_row(
            Path(r.graphml).name,
            "yes" if r.ok else "no",
            (r.smiles or r.error or "")[:40],
            str(r.confidence),
            str(r.source or ""),
        )
    console.print(table)
    console.print(f"[green]Batch summary:[/green] {batch_root}")
    console.print(f"  - {batch_root / 'batch_summary.csv'}")
    console.print(f"  - {batch_root / 'batch_summary.md'}")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share", help="Gradio public link"),
):
    """Launch a small local web UI (single + batch). Requires: pip install 'ego-mol-llm[ui]'."""
    try:
        from ego_mol_llm.ui_app import launch
    except ImportError as e:
        console.print(
            "[red]UI deps missing.[/red] Install with:\n  pip install 'ego-mol-llm[ui]'\n"
            f"({e})"
        )
        raise typer.Exit(1) from e
    console.print(f"[green]Starting UI[/green] http://{host}:{port}")
    launch(host=host, port=port, share=share)


@app.command("list-models")
def list_models():
    """List built-in model presets for the transformers backend."""
    from ego_mol_llm.backends.transformers_backend import MODEL_PRESETS

    table = Table(title="Model presets (transformers)")
    table.add_column("Preset")
    table.add_column("Hugging Face id")
    for k, v in MODEL_PRESETS.items():
        table.add_row(k, v)
    console.print(table)
    console.print(
        "\n[dim]For Ollama use model tags like chemdfm-v2-14b or qwen2.5:14b "
        "with --backend ollama.[/dim]"
    )


def main():
    app()


if __name__ == "__main__":
    main()
