"""Reporting and ego-network figure export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ego_mol_llm.predict import PredictionResult


def write_json(result: PredictionResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = result.to_dict()
    payload["model_raw"] = result.model_raw
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_markdown(result: PredictionResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = result.to_dict()
    lines = [
        "# ego-mol-llm prediction report",
        "",
        f"- **Backend**: `{d.get('backend')}` / `{d.get('model_id')}`",
        f"- **Seed id**: `{d.get('seed_id')}`",
        f"- **Seed m/z**: `{d.get('seed_mz')}`",
        f"- **Hidden true name** (evaluation only): `{d.get('true_seed_name')}`",
        "",
        "## Prediction",
        "",
        f"- **SMILES**: `{d.get('smiles')}`",
        f"- **Valid SMILES**: `{d.get('smiles_valid')}`",
        f"- **Name**: {d.get('name')}",
        f"- **Formula**: `{d.get('formula')}`",
        f"- **Adduct**: `{d.get('adduct')}`",
        f"- **Matched adduct**: `{d.get('matched_adduct')}`",
        f"- **Confidence**: `{d.get('confidence')}`",
        f"- **Exact mass**: `{d.get('exact_mass')}`",
        f"- **Mass error (Da)**: `{d.get('mass_error_da')}`",
        f"- **Mass OK**: `{d.get('mass_ok')}`",
        f"- **Parse mode**: `{d.get('parse_mode')}`",
        f"- **Source**: `{d.get('source')}`",
        f"- **Near-isobars**: `{d.get('n_near_isobars')}`",
        "",
        "## Rationale",
        "",
        d.get("rationale") or "_none_",
        "",
        "## Rescue notes",
        "",
        "```",
        "\n".join(d.get("rescue_notes") or []) or "(none)",
        "```",
        "",
        "## Neighborhood class hints",
        "",
        "```json",
        json.dumps(d.get("class_hints") or {}, indent=2),
        "```",
        "",
        "## Alternatives",
        "",
        "```json",
        json.dumps(d.get("alternatives") or [], indent=2),
        "```",
        "",
        "## Parse errors",
        "",
        "```",
        "\n".join(d.get("parse_errors") or []) or "(none)",
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_ego_network(result: PredictionResult, path: str | Path) -> Path | None:
    """Save a simple ego radial plot. Returns None if matplotlib missing."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ego = result.ego
    neigh = ego.top_neighbors
    n = len(neigh)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0f1419")
    ax.set_facecolor("#0f1419")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    title_smiles = result.prediction.canonical_smiles or result.prediction.smiles or "?"
    ax.set_title(
        f"Ego network · seed m/z={ego.seed_mz}\npred: {title_smiles}",
        color="white",
        fontsize=11,
        pad=12,
    )
    ax.scatter([0], [0], s=900, c="#38bdf8", edgecolors="white", zorder=5)
    ax.text(0, 0, "?", ha="center", va="center", fontsize=18, color="#0f172a", fontweight="bold")

    if n == 0:
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    for i, ev in enumerate(neigh):
        ang = angles[i]
        x, y = np.cos(ang), np.sin(ang)
        cos = max(ev.cosine, 0.01)
        ax.plot([0, x], [0, y], color="#94a3b8", alpha=min(1.0, cos), lw=0.5 + 3 * cos, zorder=1)
        color = "#4ade80" if ev.node.is_annotated else "#64748b"
        ax.scatter([x], [y], s=120 + 400 * cos, c=color, edgecolors="white", linewidths=0.5, zorder=4)
        label = (ev.node.name or "NO_MATCH")[:28]
        lx, ly = 1.15 * x, 1.15 * y
        ha = "left" if lx > 0.08 else ("right" if lx < -0.08 else "center")
        ax.text(lx, ly, f"{label}\ncos {cos:.2f}", fontsize=6, color="#e2e8f0", ha=ha, va="center")

    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def export_report(result: PredictionResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": write_json(result, out / "prediction.json"),
        "markdown": write_markdown(result, out / "prediction.md"),
    }
    fig = plot_ego_network(result, out / "ego_network.png")
    if fig:
        paths["figure"] = fig
    # also dump prompt for reproducibility
    (out / "prompt.txt").write_text(
        "\n\n".join(f"## {m['role']}\n{m['content']}" for m in result.messages),
        encoding="utf-8",
    )
    paths["prompt"] = out / "prompt.txt"
    (out / "model_raw.txt").write_text(result.model_raw, encoding="utf-8")
    paths["model_raw"] = out / "model_raw.txt"
    return paths
