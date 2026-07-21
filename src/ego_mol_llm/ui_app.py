"""Small Gradio UI for single + batch ego-mol-llm predictions."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ego_mol_llm.batch import run_batch
from ego_mol_llm.draw import clean_display_name
from ego_mol_llm.paths import make_run_dir
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.report import export_report


BACKENDS = ["dry-run", "ollama", "openai", "transformers"]
MODEL_PRESETS = [
    "chemdfm-v2-14b",
    "chemdfm-8b",
    "chemdfm-14b",
    "chemdfm-r-14b",
    "qwen2.5:14b",
    "qwen2.5-7b",
    "qwen2.5-14b",
]


def _format_model_card(d: dict, out_dir: Path) -> str:
    """Markdown card styled like model outputs (name + fields)."""
    name = clean_display_name(d.get("name")) or "_(name not provided by model)_"
    smi = d.get("smiles") or "—"
    conf = d.get("confidence")
    conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
    alts = d.get("alternatives") or []
    alt_lines = []
    for a in alts[:5]:
        if not isinstance(a, dict):
            continue
        alt_lines.append(
            f"- `{a.get('smiles', '')}`  "
            f"(conf={a.get('confidence')}) — {a.get('note') or a.get('name') or ''}"
        )

    return "\n".join(
        [
            "## Model prediction",
            "",
            f"### {name}",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Name** | {name} |",
            f"| **SMILES** | `{smi}` |",
            f"| **Formula** | `{d.get('formula') or '—'}` |",
            f"| **Adduct** | `{d.get('adduct') or d.get('matched_adduct') or '—'}` |",
            f"| **Confidence** | **{conf_s}** |",
            f"| **Mass OK** | `{d.get('mass_ok')}` (Δ {d.get('mass_error_da')} Da) |",
            f"| **Exact mass** | `{d.get('exact_mass')}` |",
            f"| **Source** | `{d.get('source')}` · parse `{d.get('parse_mode')}` |",
            f"| **Seed m/z** | `{d.get('seed_mz')}` |",
            f"| **Backend** | `{d.get('backend')}` / `{d.get('model_id')}` |",
            f"| **MS/MS used** | `{d.get('msms_used')}` |",
            "",
            "### Rationale",
            "",
            d.get("rationale") or "_none_",
            "",
            "### Rescue notes",
            "",
            "\n".join(f"- {n}" for n in (d.get("rescue_notes") or [])) or "_none_",
            "",
            "### Alternatives",
            "",
            "\n".join(alt_lines) if alt_lines else "_none_",
            "",
            f"**Run folder:** `{out_dir}`",
        ]
    )


def _copy_upload(f, dest_dir: Path) -> Path | None:
    if f is None:
        return None
    src = Path(f if isinstance(f, str) else f.name)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest


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
    mgf_file=None,
    seed_mgf_file=None,
):
    empty = ("Upload a GraphML file first.", None, None, None, "", "", "")
    if graphml_file is None:
        return empty

    tmp = Path(tempfile.mkdtemp(prefix="ego_mol_"))
    src = Path(graphml_file if isinstance(graphml_file, str) else graphml_file.name)
    work = tmp / src.name
    shutil.copy2(src, work)

    mgf_paths = []
    mgf_p = _copy_upload(mgf_file, tmp)
    if mgf_p:
        mgf_paths.append(mgf_p)
    seed_mgf_p = _copy_upload(seed_mgf_file, tmp)

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
        mgf_paths=mgf_paths or None,
        seed_mgf=seed_mgf_p,
    )
    paths = export_report(result, out_dir)
    d = result.to_dict()

    card = _format_model_card(d, out_dir)
    name = clean_display_name(d.get("name")) or ""
    smi = d.get("smiles") or ""

    structure = None
    if paths.get("structure") and Path(paths["structure"]).exists():
        structure = str(paths["structure"])
    elif paths.get("structure_mol") and Path(paths["structure_mol"]).exists():
        structure = str(paths["structure_mol"])

    ego = None
    if paths.get("ego_network") and Path(paths["ego_network"]).exists():
        ego = str(paths["ego_network"])
    elif paths.get("figure") and Path(paths["figure"]).exists():
        ego = str(paths["figure"])

    md_path = str(paths["markdown"]) if paths.get("markdown") else None

    return card, structure, ego, md_path, name, smi, str(out_dir)


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
        return "Upload one or more GraphML files.", None, ""

    tmp = Path(tempfile.mkdtemp(prefix="ego_mol_batch_"))
    paths: list[Path] = []
    for f in graphml_files:
        src = Path(f if isinstance(f, str) else f.name)
        dest = tmp / src.name
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
        "| File | OK | Name | SMILES | conf | source |",
        "|------|----|------|--------|------|--------|",
    ]
    gallery = []
    for r in results:
        pred_name = clean_display_name(r.name) if r.name else None
        if not pred_name and r.detail:
            pred_name = clean_display_name(r.detail.get("name"))
        display = pred_name or "—"
        lines.append(
            f"| `{Path(r.graphml).name}` | {r.ok} | {display} | "
            f"`{str(r.smiles)[:28] if r.smiles else ''}` | {r.confidence} | {r.source} |"
        )
        if r.out_dir:
            struct = Path(r.out_dir) / "structure.png"
            if struct.exists():
                caption = f"{display}\n{r.smiles or ''}"[:80]
                gallery.append((str(struct), caption))

    summary_md = batch_root / "batch_summary.md"
    return (
        "\n".join(lines),
        gallery if gallery else None,
        str(summary_md if summary_md.exists() else batch_root),
    )


def build_ui():
    import gradio as gr

    default_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    default_key = os.environ.get("OPENAI_API_KEY", "ollama")

    with gr.Blocks(title="ego-mol-llm", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # ego-mol-llm
            Blind structure prediction from MS/MS **GraphML ego networks**  
            ChemDFM / Qwen · structure drawing · model-style name & fields
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
            with gr.Row():
                mgf_file = gr.File(
                    label="Network MGF (optional, NETWORK_NODE_ID)",
                    file_types=[".mgf"],
                )
                seed_mgf_file = gr.File(
                    label="Seed/query MGF (optional)",
                    file_types=[".mgf"],
                )
            seed_id = gr.Textbox(value="0", label="Seed node id (blank = auto)")
            btn = gr.Button("Predict", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Predicted structure")
                    structure_img = gr.Image(
                        label="Structure (name + SMILES)",
                        type="filepath",
                        height=420,
                    )
                    pred_name = gr.Textbox(label="Predicted name", interactive=False)
                    pred_smiles = gr.Textbox(label="Predicted SMILES", interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("### Model output")
                    summary = gr.Markdown()
                    ego_img = gr.Image(label="Ego network", type="filepath", height=320)

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
                    mgf_file,
                    seed_mgf_file,
                ],
                outputs=[
                    summary,
                    structure_img,
                    ego_img,
                    md_file,
                    pred_name,
                    pred_smiles,
                    out_path,
                ],
            )

        with gr.Tab("Batch run"):
            files = gr.File(
                label="One or more GraphML files",
                file_types=[".graphml"],
                file_count="multiple",
            )
            btn_b = gr.Button("Run batch", variant="primary")
            batch_summary = gr.Markdown()
            batch_gallery = gr.Gallery(
                label="Predicted structures",
                columns=3,
                height=400,
                object_fit="contain",
            )
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
                outputs=[batch_summary, batch_gallery, batch_file],
            )

        gr.Markdown(
            "Each run writes a **new folder** under `outputs/runs/…` including "
            "`structure.png` (drawn molecule + name), `prediction.json`, and ego plot.  \n"
            "Install RDKit for structure drawing: `pip install rdkit`"
        )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    demo = build_ui()
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch()
