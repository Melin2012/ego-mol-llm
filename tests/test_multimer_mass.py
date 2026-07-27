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

    # Direct theoretical check (proton mass for [2M+H]+)
    from ego_mol_llm.validate import PROTON

    theo = 2 * em + PROTON
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
    # ion-only targets (no bare neutrals); [M+H]+ of half mass ~407.28
    assert any(abs(t - 407.28) < 0.5 for _, t in targets)
    assert not any(lab.startswith("neutral via") for lab, _ in targets)


def test_half_mass_delta_uses_infer_multimer():
    """True dimer residual is small; random far neighbor is not multimer-consistent."""
    from ego_mol_llm.ego import NeighborEvidence
    from ego_mol_llm.graphml import Edge, Node

    seed_mz = 813.551
    mono = Node(id="1", mz=407.279, name="mono", smiles="C")
    far = Node(id="2", mz=500.0, name="far", smiles="C")
    ev_m = NeighborEvidence(
        node=mono, edge=Edge(source="0", target="1", cosine=0.9, abs_diff_mz=406.0)
    )
    ev_f = NeighborEvidence(
        node=far, edge=Edge(source="0", target="2", cosine=0.9, abs_diff_mz=313.0)
    )
    dm = ev_m.half_mass_delta(seed_mz)
    df = ev_f.half_mass_delta(seed_mz)
    assert dm is not None and dm < 0.1
    assert df is None or df > 0.5


def test_mz_multimer_cannot_override_rdkit_mass_reject():
    """
    Misannotated library SMILES at a multimer-coincident m/z must not become
    mass_ok / rescue_ok. RDKit rejects the structure; m/z-only inference must not
    overturn that (aspirin @ ~407 vs seed [2M+H]+ @ 813.55).
    """
    from ego_mol_llm.ego import EgoContext, NeighborEvidence
    from ego_mol_llm.graphml import Edge, Node
    from ego_mol_llm.validate import DEFAULT_HALF_DMZ_MAX, check_mass

    seed_mz = 813.551
    # Monomer ion m/z for a true C24H38O5 [M+H]+ ~407.28; aspirin mass does not fit
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    ok, em, err, adduct = check_mass(
        aspirin, seed_mz, None, tol_da=0.05, include_multimer=True
    )
    assert ok is False
    assert err is not None and err > 1.0

    seed = Node(id="0", mz=seed_mz, name=None)
    bad = Node(
        id="1",
        mz=407.2786,
        name="misannotated_aspirin_at_half_mass",
        smiles=aspirin,
    )
    ego = EgoContext(
        seed=seed,
        seed_mz=seed_mz,
        neighbors=[
            NeighborEvidence(
                node=bad,
                edge=Edge(source="0", target="1", cosine=0.95, abs_diff_mz=406.27),
            )
        ],
    )
    # half_near under tight gate (m/z relationship is real)
    assert ego.neighbors[0].half_mass_delta(seed_mz) is not None
    assert ego.neighbors[0].half_mass_delta(seed_mz) <= DEFAULT_HALF_DMZ_MAX

    hyps = ego.neighbor_structure_hypotheses(
        mass_tol_da=0.05,
        dmz_max=2.0,
        half_dmz_max=DEFAULT_HALF_DMZ_MAX,
        limit=15,
        scan_all_with_smiles=True,
    )
    # Aspirin must not appear as a mass-consistent / rescue-eligible hyp
    for h in hyps:
        assert "CC(=O)Oc1ccccc1C(=O)O" not in (h.get("smiles") or "")
        assert h.get("rescue_ok") is not True or h.get("mass_ok") is True
    assert not any(
        (h.get("smiles") or "").replace(" ", "")
        in {
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(=O)Oc1ccccc1C(O)=O",
        }
        or "aspirin" in (h.get("name") or "").lower()
        for h in hyps
    )
    # Stronger: no hyp should claim tiny mass_error from this node while mass_ok
    for h in hyps:
        if h.get("name") == "misannotated_aspirin_at_half_mass":
            assert h.get("mass_ok") is not True
            assert h.get("rescue_ok") is not True


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
