"""User-friendly Gradio UI for ego-mol-llm predictions."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from ego_mol_llm.batch import run_batch
from ego_mol_llm.draw import clean_display_name
from ego_mol_llm.paths import make_run_dir
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.report import export_report


BACKENDS = ["ollama", "dry-run", "openai", "transformers"]
MODEL_PRESETS = [
    "chemdfm-v2-14b",
    "chemdfm-8b",
    "chemdfm-14b",
    "chemdfm-r-14b",
    "qwen2.5:14b",
    "qwen2.5-7b",
    "qwen2.5-14b",
]

CUSTOM_CSS = """
.ego-hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0ea5e9 140%);
  color: white !important;
  padding: 1.25rem 1.5rem;
  border-radius: 14px;
  margin-bottom: 0.75rem;
}
.ego-hero h1 { color: white !important; margin: 0 0 0.35rem 0 !important; font-size: 1.65rem !important; }
.ego-hero p { color: #cbd5e1 !important; margin: 0.15rem 0 !important; font-size: 0.95rem !important; }
.ego-status {
  font-size: 1.15rem;
  padding: 0.85rem 1.1rem;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  margin: 0.5rem 0 1rem 0;
}
.ego-stars { font-size: 1.55rem; letter-spacing: 0.12em; color: #f59e0b; }
.ego-muted { color: #64748b; font-size: 0.92rem; }
.ego-badge {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  margin-right: 0.35rem;
}
.ego-ok { background: #dcfce7; color: #166534; }
.ego-warn { background: #fef3c7; color: #92400e; }
.ego-bad { background: #fee2e2; color: #991b1b; }
.ego-info { background: #e0f2fe; color: #075985; }
.ego-section-title { font-weight: 700; color: #0f172a; margin-top: 0.5rem; }
"""


def confidence_stars(conf: float | None, max_stars: int = 5) -> str:
    """Map 0–1 confidence to filled/empty stars."""
    if conf is None:
        return "☆☆☆☆☆  ·  no confidence"
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return "☆☆☆☆☆  ·  no confidence"
    c = max(0.0, min(1.0, c))
    filled = int(round(c * max_stars))
    filled = max(0, min(max_stars, filled))
    return "★" * filled + "☆" * (max_stars - filled)


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m} min {s} s"
    h, m = divmod(m, 60)
    return f"{h} h {m} min"


def _status_badges(d: dict) -> str:
    badges = []
    source = d.get("source") or "—"
    if source == "abstain":
        badges.append('<span class="ego-badge ego-warn">abstain</span>')
    elif source == "neighbor_rescue":
        badges.append('<span class="ego-badge ego-info">neighbor rescue</span>')
    elif source == "model":
        badges.append('<span class="ego-badge ego-ok">model</span>')
    else:
        badges.append(f'<span class="ego-badge ego-info">{source}</span>')

    if d.get("mass_ok") is True:
        badges.append('<span class="ego-badge ego-ok">mass OK</span>')
    elif d.get("mass_ok") is False:
        badges.append('<span class="ego-badge ego-bad">mass fail</span>')
    else:
        badges.append('<span class="ego-badge ego-warn">mass n/a</span>')

    if d.get("msms_used"):
        badges.append('<span class="ego-badge ego-ok">MS/MS on</span>')
    else:
        badges.append('<span class="ego-badge ego-warn">network only</span>')

    if d.get("smiles"):
        badges.append('<span class="ego-badge ego-ok">structure</span>')
    else:
        badges.append('<span class="ego-badge ego-warn">no SMILES</span>')
    return " ".join(badges)


def _format_status_bar(d: dict, elapsed_s: float) -> str:
    conf = d.get("confidence")
    stars = confidence_stars(conf if isinstance(conf, (int, float)) else None)
    conf_s = f"{float(conf):.0%}" if isinstance(conf, (int, float)) else "—"
    name = clean_display_name(d.get("name")) or ("No annotation" if not d.get("smiles") else "Unnamed structure")
    return f"""
<div class="ego-status">
  <div style="display:flex; flex-wrap:wrap; gap:1rem; align-items:center; justify-content:space-between;">
    <div>
      <div class="ego-stars">{stars}</div>
      <div class="ego-muted">Confidence <b>{conf_s}</b> · {name}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.35rem; font-weight:700; color:#0f172a;">⏱ {format_duration(elapsed_s)}</div>
      <div class="ego-muted">time consumed</div>
    </div>
  </div>
  <div style="margin-top:0.65rem;">{_status_badges(d)}</div>
</div>
"""


def _format_model_card(d: dict, out_dir: Path, elapsed_s: float) -> str:
    name = clean_display_name(d.get("name")) or "_(name not provided)_"
    smi = d.get("smiles") or "—"
    conf = d.get("confidence")
    conf_s = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "—"
    stars = confidence_stars(conf if isinstance(conf, (int, float)) else None)
    alts = d.get("alternatives") or []
    alt_lines = []
    for a in alts[:5]:
        if not isinstance(a, dict):
            continue
        ac = a.get("confidence")
        ac_s = f"{float(ac):.2f}" if isinstance(ac, (int, float)) else str(ac)
        alt_lines.append(
            f"- `{a.get('smiles', '')}`  "
            f"({confidence_stars(float(ac) if isinstance(ac, (int, float)) else None)} · {ac_s}) — "
            f"{a.get('note') or a.get('name') or ''}"
        )

    spec = d.get("spectral") or {}
    diag = spec.get("seed_diagnostics") or {}
    diag_s = ", ".join(f"`{k}`" for k in list(diag.keys())[:8]) or "—"

    return "\n".join(
        [
            f"### {name}",
            "",
            f"**{stars}** &nbsp; confidence **{conf_s}** &nbsp;·&nbsp; ⏱ **{format_duration(elapsed_s)}**",
            "",
            "| | |",
            "|:--|:--|",
            f"| **Name** | {name} |",
            f"| **SMILES** | `{smi}` |",
            f"| **Formula** | `{d.get('formula') or '—'}` |",
            f"| **Adduct** | `{d.get('adduct') or d.get('matched_adduct') or '—'}` |",
            f"| **Mass OK** | `{d.get('mass_ok')}` &nbsp; (Δ {d.get('mass_error_da')} Da) |",
            f"| **Exact mass** | `{d.get('exact_mass') or '—'}` |",
            f"| **Source** | `{d.get('source')}` · parse `{d.get('parse_mode')}` |",
            f"| **Seed m/z** | `{d.get('seed_mz')}` |",
            f"| **Model** | `{d.get('backend')}` / `{d.get('model_id')}` |",
            f"| **MS/MS** | `{'yes' if d.get('msms_used') else 'no'}` · diagnostics: {diag_s} |",
            f"| **Neighbors** | `{d.get('n_neighbors')}` · near-isobars `{d.get('n_near_isobars')}` |",
            "",
            "#### Why this call?",
            "",
            d.get("rationale") or "_No rationale returned._",
            "",
            "#### Rescue / pipeline notes",
            "",
            "\n".join(f"- {n}" for n in (d.get("rescue_notes") or [])) or "_none_",
            "",
            "#### Alternatives",
            "",
            "\n".join(alt_lines) if alt_lines else "_none_",
            "",
            f"📁 **Run folder:** `{out_dir}`",
        ]
    )


def _copy_upload(f, dest_dir: Path) -> Path | None:
    if f is None:
        return None
    src = Path(f if isinstance(f, str) else f.name)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest


def _empty_single(msg: str):
    return (
        f'<div class="ego-status"><span class="ego-badge ego-warn">waiting</span> {msg}</div>',
        msg,
        None,
        None,
        None,
        "",
        "",
        "",
        "—",
    )


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
    progress=None,
):
    def _p(frac, desc=""):
        if progress is not None:
            try:
                progress(frac, desc=desc)
            except Exception:
                pass

    if graphml_file is None:
        return _empty_single("Upload a **GraphML** network to begin.")

    t0 = time.perf_counter()
    _p(0.05, "Preparing files…")

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

    _p(0.15, "Running prediction (network + MS/MS if provided)…")
    try:
        result = predict_from_graphml(
            graphml_path=work,
            backend=backend,
            model=model,
            seed_id=(seed_id or "").strip() or None,
            hide_seed_name=hide_seed,
            max_neighbors=int(max_neighbors),
            base_url=(base_url or "").strip() or None,
            api_key=(api_key or "").strip() or None,
            mass_tol_da=float(mass_tol),
            load_in_4bit=False,
            mgf_paths=mgf_paths or None,
            seed_mgf=seed_mgf_p,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        err = (
            f'<div class="ego-status"><span class="ego-badge ego-bad">error</span> '
            f"{type(e).__name__}: {e}<br/><span class=\"ego-muted\">⏱ {format_duration(elapsed)}</span></div>"
        )
        return err, f"**Error:** `{e}`", None, None, None, "", "", "", format_duration(elapsed)

    _p(0.85, "Writing report & drawing structure…")
    paths = export_report(result, out_dir)
    d = result.to_dict()
    elapsed = time.perf_counter() - t0
    d["elapsed_seconds"] = elapsed

    status = _format_status_bar(d, elapsed)
    card = _format_model_card(d, out_dir, elapsed)
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
    _p(1.0, "Done")

    return (
        status,
        card,
        structure,
        ego,
        md_path,
        name,
        smi,
        str(out_dir),
        format_duration(elapsed),
    )


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
        empty = (
            '<div class="ego-status"><span class="ego-badge ego-warn">waiting</span> '
            "Upload one or more GraphML files.</div>"
        )
        return empty, "Upload one or more GraphML files.", None, "", "—"

    t0 = time.perf_counter()
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
            try:
                progress(i / max(n, 1), desc=f"{i}/{n} {Path(p).name}")
            except Exception:
                pass

    results, batch_root = run_batch(
        paths,
        backend=backend,
        model=model,
        out_parent=Path("outputs/runs"),
        hide_seed_name=hide_seed,
        max_neighbors=int(max_neighbors),
        base_url=(base_url or "").strip() or None,
        api_key=(api_key or "").strip() or None,
        mass_tol_da=float(mass_tol),
        load_in_4bit=False,
        progress=_prog,
    )
    elapsed = time.perf_counter() - t0
    ok = sum(1 for r in results if r.ok)
    confs = [float(r.confidence) for r in results if r.ok and isinstance(r.confidence, (int, float))]
    avg_conf = sum(confs) / len(confs) if confs else None
    stars = confidence_stars(avg_conf)

    status = f"""
<div class="ego-status">
  <div style="display:flex; flex-wrap:wrap; gap:1rem; justify-content:space-between; align-items:center;">
    <div>
      <div class="ego-stars">{stars}</div>
      <div class="ego-muted">Average confidence {f'{avg_conf:.0%}' if avg_conf is not None else '—'} · {ok}/{len(results)} succeeded</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.35rem; font-weight:700;">⏱ {format_duration(elapsed)}</div>
      <div class="ego-muted">batch time</div>
    </div>
  </div>
  <div style="margin-top:0.5rem;">
    <span class="ego-badge ego-ok">{ok} ok</span>
    <span class="ego-badge ego-bad">{len(results)-ok} failed</span>
    <span class="ego-badge ego-info">{len(results)} total</span>
  </div>
</div>
"""

    lines = [
        f"### Batch complete · ⏱ {format_duration(elapsed)}",
        "",
        f"**Folder:** `{batch_root}`",
        "",
        f"| # | File | OK | Rating | Name | SMILES | conf | source |",
        f"|--:|------|:--:|:------:|------|--------|-----:|--------|",
    ]
    gallery = []
    for i, r in enumerate(results, 1):
        pred_name = clean_display_name(r.name) if r.name else None
        if not pred_name and r.detail:
            pred_name = clean_display_name(r.detail.get("name"))
        display = pred_name or "—"
        conf = r.confidence
        conf_f = float(conf) if isinstance(conf, (int, float)) else None
        star = confidence_stars(conf_f)
        conf_s = f"{conf_f:.2f}" if conf_f is not None else "—"
        lines.append(
            f"| {i} | `{Path(r.graphml).name}` | {'✅' if r.ok else '❌'} | {star} | {display} | "
            f"`{str(r.smiles)[:28] if r.smiles else '—'}` | {conf_s} | {r.source or '—'} |"
        )
        if r.out_dir:
            struct = Path(r.out_dir) / "structure.png"
            if struct.exists():
                caption = f"{star} {display}\n{r.smiles or ''}"[:90]
                gallery.append((str(struct), caption))

    summary_md = batch_root / "batch_summary.md"
    return (
        status,
        "\n".join(lines),
        gallery if gallery else None,
        str(summary_md if summary_md.exists() else batch_root),
        format_duration(elapsed),
    )


def build_ui():
    import gradio as gr

    default_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    default_key = os.environ.get("OPENAI_API_KEY", "ollama")

    with gr.Blocks(title="ego-mol-llm · structure from networks") as demo:
        gr.HTML(
            """
            <div class="ego-hero">
              <h1>🧬 ego-mol-llm</h1>
              <p>Blind structure prediction from MS/MS molecular-network ego neighborhoods</p>
              <p>Upload GraphML · optional MGF spectra · ChemDFM / Qwen · stars = confidence · timer = runtime</p>
            </div>
            """
        )

        with gr.Accordion("⚙️ Model & connection", open=True):
            with gr.Row():
                backend = gr.Dropdown(
                    BACKENDS,
                    value="ollama",
                    label="Backend",
                    info="ollama = local ChemDFM/Qwen · dry-run = offline demo",
                )
                model = gr.Dropdown(
                    MODEL_PRESETS,
                    value="chemdfm-v2-14b",
                    label="Model",
                    allow_custom_value=True,
                    info="Ollama tag or Hugging Face preset",
                )
            with gr.Row():
                base_url = gr.Textbox(
                    value=default_url,
                    label="API base URL",
                    info="Default Ollama OpenAI-compatible endpoint",
                )
                api_key = gr.Textbox(
                    value=default_key,
                    label="API key",
                    type="password",
                    info="Use ollama for local Ollama",
                )
            with gr.Row():
                max_neighbors = gr.Slider(
                    5, 50, value=25, step=1, label="Max neighbors in ego",
                )
                mass_tol = gr.Number(
                    value=0.05,
                    label="Mass tolerance (Da)",
                    info="For adduct / multimer checks",
                )
                hide_seed = gr.Checkbox(
                    value=True,
                    label="Blind mode (hide seed library name)",
                    info="Recommended for fair evaluation",
                )

        with gr.Tab("🔬 Single prediction"):
            gr.Markdown(
                "### Step 1 — Files\n"
                "Start with **GraphML**. Add MGFs for better accuracy when available."
            )
            graphml = gr.File(
                label="① GraphML network (required)",
                file_types=[".graphml"],
            )
            with gr.Row():
                mgf_file = gr.File(
                    label="② Network MGF — all nodes (optional)",
                    file_types=[".mgf"],
                )
                seed_mgf_file = gr.File(
                    label="③ Seed / query MGF (optional)",
                    file_types=[".mgf"],
                )
            seed_id = gr.Textbox(
                value="0",
                label="Center node id",
                info="Usually 0 for HNSW query exports",
            )

            with gr.Row():
                btn = gr.Button("🚀 Predict structure", variant="primary", scale=2)
                elapsed_box = gr.Textbox(
                    label="Time consumed",
                    value="—",
                    interactive=False,
                    scale=1,
                )

            status_html = gr.HTML(
                '<div class="ego-status"><span class="ego-badge ego-info">ready</span> '
                "Upload a GraphML file, then click <b>Predict structure</b>.</div>"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Structure")
                    structure_img = gr.Image(
                        label="Drawn structure",
                        type="filepath",
                        height=400,
                    )
                    pred_name = gr.Textbox(label="Predicted name", interactive=False)
                    pred_smiles = gr.Textbox(label="Predicted SMILES", interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("### Details")
                    summary = gr.Markdown(
                        "_Results will appear here: stars, timing, formula, rationale._"
                    )
                    ego_img = gr.Image(
                        label="Ego network map",
                        type="filepath",
                        height=300,
                    )

            with gr.Accordion("📁 Outputs", open=False):
                out_path = gr.Textbox(label="Run folder", interactive=False)
                md_file = gr.File(label="Download prediction.md")

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
                    status_html,
                    summary,
                    structure_img,
                    ego_img,
                    md_file,
                    pred_name,
                    pred_smiles,
                    out_path,
                    elapsed_box,
                ],
            )

        with gr.Tab("📦 Batch"):
            gr.Markdown(
                "### Run many GraphML files\n"
                "Each file gets its own timestamped folder. Summary table includes **stars** and **batch time**."
            )
            files = gr.File(
                label="GraphML files",
                file_types=[".graphml"],
                file_count="multiple",
            )
            with gr.Row():
                btn_b = gr.Button("🚀 Run batch", variant="primary", scale=2)
                batch_elapsed = gr.Textbox(
                    label="Batch time",
                    value="—",
                    interactive=False,
                    scale=1,
                )
            batch_status = gr.HTML(
                '<div class="ego-status"><span class="ego-badge ego-info">ready</span> '
                "Select multiple GraphML files to batch-annotate.</div>"
            )
            batch_summary = gr.Markdown()
            batch_gallery = gr.Gallery(
                label="Predicted structures",
                columns=3,
                height=420,
                object_fit="contain",
            )
            batch_file = gr.File(label="Download batch_summary.md")
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
                outputs=[
                    batch_status,
                    batch_summary,
                    batch_gallery,
                    batch_file,
                    batch_elapsed,
                ],
            )

        gr.Markdown(
            """
---
#### Tips
| Star rating | Meaning (approx.) |
|:-----------:|-------------------|
| ★★★★★ | ≥ ~90% confidence |
| ★★★★☆ | ~70–90% |
| ★★★☆☆ | ~50–70% |
| ★★☆☆☆ or less | low / abstain |

- **MS/MS MGF** improves accuracy (diagnostic ions, neighbor spectral cosine).
- Each run writes a **new** folder under `outputs/runs/…` (`structure.png`, `prediction.json`, `spectral.json`).
- Structure drawing needs RDKit: `pip install rdkit`
            """
        )

    # Gradio 6: attach css/theme on launch via demo attributes when possible
    demo.css = CUSTOM_CSS
    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    import gradio as gr

    demo = build_ui()
    # Gradio 6 prefers theme/css on launch
    try:
        demo.launch(
            server_name=host,
            server_port=port,
            share=share,
            theme=gr.themes.Soft(
                primary_hue="sky",
                secondary_hue="slate",
                neutral_hue="slate",
            ),
            css=CUSTOM_CSS,
        )
    except TypeError:
        demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch()
