from pathlib import Path

from ego_mol_llm.draw import clean_display_name, draw_prediction_card, draw_smiles


def test_clean_display_name():
    n = clean_display_name(
        "1-Methyl-hydantoin_AROMEC18COLGATE001214 CollisionEnergy:102040 M-H"
    )
    assert n is not None
    assert "CollisionEnergy" not in n
    assert "hydantoin" in n.lower() or "Methyl" in n


def test_draw_smiles_png(tmp_path: Path):
    out = tmp_path / "mol.png"
    path = draw_smiles("CCO", out, legend="ethanol")
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 100


def test_draw_prediction_card(tmp_path: Path):
    out = tmp_path / "card.png"
    path = draw_prediction_card(
        "CN1CC(=O)NC1=O",
        "1-methylhydantoin-like / dioxy-creatinine",
        out,
        formula="C4H6N2O2",
        adduct="[M+H]+",
        confidence=0.9,
        mz=115.05,
    )
    assert path is not None
    assert path.exists()
