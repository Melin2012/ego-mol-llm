"""Small Gradio UI for single + batch ego-mol-llm predictions."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ego_mol_llm.batch import run_batch
from ego_mol_llm.paths import make_run_dir
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.report import export_report


BACKENDS = ["dry-run", "ollama", "openai", "transformers"]
MODEL_PRESETS = [
    "chemdfm-8b",
    "chemdfm-14b",
    "chemdfm-r-14b",
    "qwen2.5-7b",
    "qwen2.5:14b",
    "chemdfm-v2-14b",
    "qwen2.5-14b",
]


def _predict_one(
    graphml_file,
    backend: str,
    model: str,
    seed_id: str,
    hide_seed: bool,
    max_neighbors: int,
    base_url: str,
    api_key: str,
    mass_tol: float,
):
    if graphml_file is None:
        return "Upload a GraphML file first.", None, None, ""

    src = Path(graphml_file if isinstance(graphml_file, str) else graphml_file.name)
    # Gradio may give a temp path — copy into a stable name for stem
    work = Path(tempfile.mkdtemp(prefix="ego_mol_")) / src.name
    shutil.copy2(src, work)

    out_dir = make_run_dir(
        parent=Path("outputs/runs"),
        graphml=work,
        backend=backend,
        model=model,
        label="ui",
    )
    result = predict_from_graphml(
        graphml_path=work,
        backend=backend,
        model=model,
        seed_id=seed_id.strip() or None,
        hide_seed_name=hide_seed,
        max_neighbors=int(max_neighbors),
        base_url=base_url.strip() or None,
        api_key=api_key.strip() or None,
        mass_tol_da=float(mass_tol),
        load_in_4bit=False,
    )
    paths = export_report(result, out_dir)
    d = result.to_dict()

    summary = "\n".join(
        [
            f"**Output:** `{out_dir}`",
            f"**SMILES:** `{d.get('smiles')}`",
            f"**Name:** {d.get('name')}",
            f"**Confidence:** {d.get('confidence')}",
            f"**Mass OK:** {d.get('mass_ok')}  ·  **Source:** {d.get('source')}",
            f"**Seed m/z:** {d.get('seed_mz')}",
            f"**Parse mode:** {d.get('parse_mode')}",
            "",
            "### Rationale",
            d.get("rationale") or "_none_",
            "",
            "### Rescue notes",
            "\n".join(f"- {n}" for n in (d.get("rescue_notes") or [])) or "_none_",
        ]
    )
    fig = str(paths["figure"]) if paths.get("figure") and Path(paths["figure"]).exists() else None
    md_path = str(paths.get("markdown", ""))
    return summary, fig, md_path, str(out_dir)


def _predict_batch(
    graphml_files,
    backend: str,
    model: str,
    hide_seed: bool,
    max_neighbors: int,
    base_url: str,
    api_key: str,
    mass_tol: float,
    progress=None,
):
    if not graphml_files:
        return "Upload one or more GraphML files.", ""

    tmp = Path(tempfile.mkdtemp(prefix="ego_mol_batch_"))
    paths: list[Path] = []
    for f in graphml_files:
        src = Path(f if isinstance(f, str) else f.name)
        dest = tmp / src.name
        # avoid overwrite collisions
        if dest.exists():
            dest = tmp / f"{src.stem}_{len(paths)}{src.suffix}"
        shutil.copy2(src, dest)
        paths.append(dest)

    def _prog(i, n, p):
        if progress is not None:
            progress(i / n, desc=f"{i}/{n} {p.name}")

    results, batch_root = run_batch(
        paths,
        backend=backend,
        model=model,
        out_parent=Path("outputs/runs"),
        hide_seed_name=hide_seed,
        max_neighbors=int(max_neighbors),
        base_url=base_url.strip() or None,
        api_key=api_key.strip() or None,
        mass_tol_da=float(mass_tol),
        load_in_4bit=False,
        progress=_prog,
    )
    ok = sum(1 for r in results if r.ok)
    lines = [
        f"**Batch folder:** `{batch_root}`",
        f"**OK:** {ok}/{len(results)}",
        "",
        "| File | OK | SMILES | conf | source |",
        "|------|----|--------|------|--------|",
    ]
    for r in results:
        lines.append(
            f"| `{Path(r.graphml).name}` | {r.ok} | `{str(r.smiles)[:32] if r.smiles else ''}` "
            f"| {r.confidence} | {r.source} |"
        )
    summary_md = batch_root / "batch_summary.md"
    return "\n".join(lines), str(summary_md if summary_md.exists() else batch_root)


def build_ui():
    import gradio as gr

    default_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    default_key = os.environ.get("OPENAI_API_KEY", "ollama")

    with gr.Blocks(title="ego-mol-llm", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # ego-mol-llm
            Blind structure prediction from MS/MS **GraphML ego networks**  
            (ChemDFM / Qwen via Ollama or local transformers)
            """
        )
        with gr.Row():
            backend = gr.Dropdown(BACKENDS, value="ollama", label="Backend")
            model = gr.Dropdown(
                MODEL_PRESETS,
                value="chemdfm-v2-14b",
                label="Model",
                allow_custom_value=True,
            )
        with gr.Row():
            base_url = gr.Textbox(value=default_url, label="API base URL (Ollama/vLLM)")
            api_key = gr.Textbox(value=default_key, label="API key", type="password")
        with gr.Row():
            max_neighbors = gr.Slider(5, 50, value=25, step=1, label="Max neighbors")
            mass_tol = gr.Number(value=0.05, label="Mass tol (Da)")
            hide_seed = gr.Checkbox(value=True, label="Blind seed name")

        with gr.Tab("Single run"):
            graphml = gr.File(label="GraphML network", file_types=[".graphml"])
            seed_id = gr.Textbox(value="0", label="Seed node id (blank = auto)")
            btn = gr.Button("Predict", variant="primary")
            summary = gr.Markdown()
            fig = gr.Image(label="Ego network", type="filepath")
            out_path = gr.Textbox(label="Run folder", interactive=False)
            md_file = gr.File(label="prediction.md")
            btn.click(
                _predict_one,
                inputs=[
                    graphml,
                    backend,
                    model,
                    seed_id,
                    hide_seed,
                    max_neighbors,
                    base_url,
                    api_key,
                    mass_tol,
                ],
                outputs=[summary, fig, md_file, out_path],
            )

        with gr.Tab("Batch run"):
            files = gr.File(
                label="One or more GraphML files",
                file_types=[".graphml"],
                file_count="multiple",
            )
            btn_b = gr.Button("Run batch", variant="primary")
            batch_summary = gr.Markdown()
            batch_file = gr.File(label="batch_summary.md")
            btn_b.click(
                _predict_batch,
                inputs=[
                    files,
                    backend,
                    model,
                    hide_seed,
                    max_neighbors,
                    base_url,
                    api_key,
                    mass_tol,
                ],
                outputs=[batch_summary, batch_file],
            )

        gr.Markdown(
            "Each run writes a **new folder** under `outputs/runs/"
            "<timestamp>_<file>_<backend>/` so nothing is overwritten."
        )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    demo = build_ui()
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch()
