"""Command-line interface for ego-mol-llm."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.report import export_report

app = typer.Typer(
    name="ego-mol-llm",
    help="Blind structure prediction from MS/MS molecular-network ego neighborhoods.",
    add_completion=False,
)
console = Console()


@app.command()
def predict(
    graphml: Path = typer.Argument(..., exists=True, help="Path to GraphML molecular network"),
    backend: str = typer.Option(
        "dry-run",
        "--backend",
        "-b",
        help="dry-run | transformers | openai (Ollama/vLLM/OpenRouter)",
    ),
    model: str = typer.Option(
        "chemdfm-8b",
        "--model",
        "-m",
        help="Preset (chemdfm-8b, chemdfm-14b, qwen2.5-7b, qwen3.5-4b) or HF/Ollama model id",
    ),
    seed_id: Optional[str] = typer.Option(None, "--seed-id", help="Center node id (default: auto/'0')"),
    seed_name: Optional[str] = typer.Option(None, "--seed-name", help="Substring match for seed name"),
    show_seed_name: bool = typer.Option(
        False, "--show-seed-name", help="Do NOT hide seed library name (evaluation leak)"
    ),
    max_neighbors: int = typer.Option(25, "--max-neighbors"),
    no_two_hop: bool = typer.Option(False, "--no-two-hop"),
    out_dir: Path = typer.Option(Path("outputs/run"), "--out", "-o"),
    temperature: float = typer.Option(0.2, "--temperature"),
    max_new_tokens: int = typer.Option(1024, "--max-new-tokens"),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    mass_tol: float = typer.Option(0.05, "--mass-tol", help="Mass match tolerance in Da"),
):
    """Predict structure of an unknown center node from GraphML ego network."""
    console.print(Panel.fit(
        f"[bold]ego-mol-llm[/bold]\nbackend={backend} model={model}\n{graphml}",
        border_style="cyan",
    ))
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
    )
    paths = export_report(result, out_dir)
    d = result.to_dict()

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
        "seed_mz",
    ]:
        table.add_row(k, str(d.get(k)))
    console.print(table)
    console.print(f"[green]Wrote report to[/green] {out_dir.resolve()}")
    for k, p in paths.items():
        console.print(f"  - {k}: {p}")


@app.command("list-models")
def list_models():
    """List built-in model presets for the transformers backend."""
    from ego_mol_llm.backends.transformers_backend import MODEL_PRESETS

    table = Table(title="Model presets")
    table.add_column("Preset")
    table.add_column("Hugging Face id")
    for k, v in MODEL_PRESETS.items():
        table.add_row(k, v)
    console.print(table)
    console.print(
        "\n[dim]ChemDFM models are chemistry-specialized open LLMs "
        "(Qwen2.5 post-trained for ChemDFM-v2 / R). "
        "Qwen3.5/Qwen2.5 presets are general instruct models.[/dim]"
    )


def main():
    app()


if __name__ == "__main__":
    main()
