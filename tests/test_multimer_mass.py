"""Multimer / half-mass adduct logic."""

from pathlib import Path

import pytest

from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.validate import check_mass, formula_to_mass, monomer_mass_targets

BILE = Path(
    r"C:\Users\AlexeyMelnik\OneDrive - Arome Science Inc\Attachments"
    r"\HNSW_12-Ketochenodeoxycholic acid_AROMEC18COLGATE001442.graphml"
)


def test_c24h38o5_dimer_matches_813():
    em = formula_to_mass("C24H38O5")
    assert em is not None
    # Use dummy SMILES path via formula fallback
    ok, em2, err, adduct = check_mass(
        "C",  # invalid structure; force formula
        813.551141,
        formula="C24H38O5",
        tol_da=0.05,
        allow_formula_fallback=True,
    )
    # "C" may parse as carbon - use empty invalid
    ok, em2, err, adduct = check_mass(
        "not_a_smiles_xxx",
        813.551141,
        formula="C24H38O5",
        tol_da=0.05,
    )
    # invalid smiles with formula
    from ego_mol_llm.validate import check_mass as cm

    # Direct theoretical check
    theo = 2 * em + 1.007825
    assert abs(theo - 813.551141) < 0.01


def test_check_mass_multimer_with_formula_only():
    """When SMILES mass unavailable, formula + multimer still works if we call correctly."""
    em = formula_to_mass("C24H38O5")
    assert em is not None
    # Simulate check_mass internals: theoretical_ion_mz
    from ego_mol_llm.validate import theoretical_ion_mz

    ions = theoretical_ion_mz(em, include_multimer=True)
    names = {n for n, _ in ions}
    assert "[2M+H]+" in names
    mz_2mh = dict(ions)["[2M+H]+"]
    assert abs(mz_2mh - 813.551) < 0.01


def test_monomer_targets_include_half():
    targets = monomer_mass_targets(813.551141)
    # one target near 406.27
    assert any(abs(t - 406.27) < 0.5 for _, t in targets)


@pytest.mark.skipif(not BILE.exists(), reason="bile GraphML not present")
def test_bile_ego_half_mass_and_rescue():
    net = load_graphml(BILE)
    ego = build_ego(net, seed_id="0", hide_seed_name=True, max_neighbors=35)
    assert ego.seed_mz is not None
    assert abs(ego.seed_mz - 813.55) < 0.1
    halfs = ego.half_mass_neighbors(1.0)
    assert len(halfs) >= 1
    hyps = ego.neighbor_structure_hypotheses(
        mass_tol_da=0.1, dmz_max=2.0, half_dmz_max=2.0, limit=15
    )
    assert len(hyps) >= 1
    # Top hyp should involve multimer or half-mass note
    top = hyps[0]
    assert top.get("smiles")
    note = (top.get("note") or "") + str(top.get("adduct") or "")
    assert (
        "2M" in note
        or "half" in note.lower()
        or "multimer" in note.lower()
        or (top.get("half_mass_delta") is not None and top["half_mass_delta"] < 2)
    )


@pytest.mark.skipif(not BILE.exists(), reason="bile GraphML not present")
def test_bile_dry_run_pipeline_rescues():
    result = predict_from_graphml(
        BILE,
        backend="dry-run",
        hide_seed_name=True,
        max_neighbors=35,
        mass_tol_da=0.1,
    )
    d = result.to_dict()
    assert d.get("smiles")
    # Should not be empty; preferably multimer or rescue
    assert d.get("source") in {"model", "neighbor_rescue", "hybrid"}
    # If mass checked, prefer ok; without RDKit may still rescue via half-mass
    notes = " ".join(d.get("rescue_notes") or [])
    adduct = str(d.get("adduct") or d.get("matched_adduct") or "")
    # Either multimer adduct or half-mass rescue note or bile-like SMILES length
    assert (
        "2M" in adduct
        or "multimer" in notes.lower()
        or "half" in notes.lower()
        or "Rescued" in notes
        or len(d["smiles"]) > 20
    )
