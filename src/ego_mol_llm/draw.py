"""Draw 2D molecular structures from SMILES (RDKit)."""

from __future__ import annotations

import re
from pathlib import Path


def clean_display_name(name: str | None, max_len: int = 120) -> str | None:
    """Shorten library-style names for UI display."""
    if not name:
        return None
    n = name.strip()
    for cut in [
        " CollisionEnergy:",
        " (predicted",
        " with delta m/z",
        " [M+",
        " [M-",
        " M+H",
        " M-H",
        " [IIN-based",
    ]:
        if cut in n:
            n = n.split(cut)[0]
    if n.startswith("Suspect related to "):
        n = n.replace("Suspect related to ", "related to ", 1)
    if n.startswith("Spectral Match to "):
        n = n.replace("Spectral Match to ", "match: ", 1)
    if n.startswith("Massbank:"):
        n = re.sub(r"^Massbank:\S+\s*", "", n)
    n = n.strip(" _-|")
    if len(n) > max_len:
        n = n[: max_len - 1] + "…"
    return n or None


def draw_smiles(
    smiles: str | None,
    path: str | Path,
    *,
    legend: str | None = None,
    size: tuple[int, int] = (500, 400),
) -> Path | None:
    """
    Render SMILES to a PNG file. Returns path on success, None if unavailable.
    Requires RDKit.
    """
    if not smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return _draw_placeholder(path, smiles, legend, missing_rdkit=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _draw_placeholder(path, smiles, legend, invalid=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Prefer modern drawer for cleaner output
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.clearBackground = True
        drawer.DrawMolecule(mol, legend=legend or "")
        drawer.FinishDrawing()
        path.write_bytes(drawer.GetDrawingText())
        return path
    except Exception:
        # Fallback PIL-based drawer
        try:
            img = Draw.MolToImage(mol, size=size, legend=legend or "")
            img.save(str(path))
            return path
        except Exception:
            return _draw_placeholder(path, smiles, legend, invalid=True)


def _draw_placeholder(
    path: str | Path,
    smiles: str,
    legend: str | None,
    *,
    missing_rdkit: bool = False,
    invalid: bool = False,
) -> Path | None:
    """Text placeholder when RDKit cannot draw."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    title = legend or "Predicted structure"
    if missing_rdkit:
        msg = "Install RDKit to draw structures:\npip install rdkit"
    elif invalid:
        msg = "Invalid / undrawable SMILES"
    else:
        msg = "Structure unavailable"
    ax.text(0.5, 0.7, title, ha="center", va="center", fontsize=12, fontweight="bold", wrap=True)
    ax.text(0.5, 0.45, msg, ha="center", va="center", fontsize=11, color="#b91c1c")
    ax.text(0.5, 0.2, smiles[:80], ha="center", va="center", fontsize=8, family="monospace", color="#334155")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_prediction_card(
    smiles: str | None,
    name: str | None,
    out_path: str | Path,
    *,
    formula: str | None = None,
    adduct: str | None = None,
    confidence: float | None = None,
    mz: float | None = None,
) -> Path | None:
    """
    Structure drawing with name banner (model-output style card).
    """
    display_name = clean_display_name(name) or "Predicted structure"
    path = Path(out_path)
    # First draw mol only
    mol_path = path.with_name(path.stem + "_mol_only.png")
    drawn = draw_smiles(smiles, mol_path, legend=None, size=(520, 380))

    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        return drawn

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 7.0), facecolor="white")
    # Title block
    ax_t = fig.add_axes([0.06, 0.78, 0.88, 0.18])
    ax_t.axis("off")
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.add_patch(
        plt.Rectangle((0, 0), 1, 1, facecolor="#0f172a", transform=ax_t.transAxes, clip_on=False)
    )
    ax_t.text(
        0.5,
        0.62,
        display_name,
        ha="center",
        va="center",
        color="white",
        fontsize=13,
        fontweight="bold",
        wrap=True,
    )
    meta_bits = []
    if formula:
        meta_bits.append(str(formula))
    if adduct:
        meta_bits.append(str(adduct))
    if confidence is not None:
        meta_bits.append(f"conf {confidence:.2f}" if isinstance(confidence, float) else f"conf {confidence}")
    if mz is not None:
        meta_bits.append(f"m/z {mz:.4f}" if isinstance(mz, float) else f"m/z {mz}")
    ax_t.text(
        0.5,
        0.22,
        " · ".join(meta_bits) if meta_bits else "ego-mol-llm prediction",
        ha="center",
        va="center",
        color="#94a3b8",
        fontsize=10,
    )

    ax = fig.add_axes([0.08, 0.08, 0.84, 0.68])
    ax.axis("off")
    if drawn and Path(drawn).exists():
        img = mpimg.imread(str(drawn))
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, smiles or "No SMILES", ha="center", va="center", fontsize=11)

    # SMILES footer
    ax.text(
        0.5,
        -0.04,
        (smiles or "")[:90],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        family="monospace",
        color="#475569",
    )

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # cleanup intermediate
    try:
        if mol_path.exists() and mol_path != path:
            mol_path.unlink()
    except OSError:
        pass
    return path
