"""Parser + mass-first rescue tests (no GPU required)."""

from pathlib import Path

import pytest

from ego_mol_llm.predict import refine_with_neighborhood
from ego_mol_llm.ego import EgoContext, NeighborEvidence, build_ego
from ego_mol_llm.graphml import Edge, Node, load_graphml
from ego_mol_llm.validate import parse_model_output

CHEMDFM_STYLE = """
Based on the MS/MS molecular network, the best prediction for the unknown center node is:

smiles: CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](N)CC(N)=O)C(=O)O
iupac_or_common_name: some peptide
formula: C22H31N3O5
adduct: [M-H]-
"""

GRAPHML_MTCA = Path(
    r"C:\Users\AlexeyMelnik\Downloads\HNSW_1-Methyl-1,2,3,4-tetrahydro-beta-carboline-3-carboxylic acid_AROMEC18COLGATE001635.graphml"
)


def test_parse_chemdfm_key_value_without_json():
    p = parse_model_output(CHEMDFM_STYLE, precursor_mz=115.05, mass_tol_da=0.05)
    assert p.parse_mode == "key_value"
    assert p.smiles is not None
    assert "CC(C)" in p.smiles
    assert p.formula == "C22H31N3O5"
    assert p.adduct == "[M-H]-"
    # peptide mass should NOT fit m/z 115
    if p.exact_mass is not None:
        assert p.mass_ok is False


def test_parse_json_still_works():
    text = """
```json
{
  "smiles": "CN1CC(=O)NC1=O",
  "iupac_or_common_name": "1-methylhydantoin-like",
  "formula": "C4H6N2O2",
  "adduct": "[M+H]+",
  "confidence": 0.8,
  "rationale": "near-isobar",
  "alternatives": []
}
```
"""
    p = parse_model_output(text, precursor_mz=115.05, mass_tol_da=0.05)
    assert p.parse_mode == "json"
    assert p.smiles_valid is True
    # C4H6N2O2 ~ 114.04; [M+H]+ ~ 115.05
    if p.exact_mass is not None:
        assert p.mass_ok is True


def test_neighbor_rescue_replaces_mass_inconsistent_model():
    seed = Node(id="0", mz=115.05, name=None)
    good = Node(
        id="1",
        mz=115.05,
        name="Dioxy-creatinine",
        smiles="CN1CC(=O)NC1=O",
    )
    edge = Edge(source="0", target="1", cosine=0.915, abs_diff_mz=0.0001)
    ego = EgoContext(
        seed=seed,
        seed_mz=115.05,
        neighbors=[NeighborEvidence(node=good, edge=edge)],
    )
    bad = parse_model_output(CHEMDFM_STYLE, precursor_mz=115.05, mass_tol_da=0.05)
    refined, notes = refine_with_neighborhood(bad, ego, mass_tol_da=0.05)
    assert any("Rejected" in n or "Rescued" in n for n in notes)
    assert refined.canonical_smiles is not None or refined.smiles is not None
    # Should now be the hydantoin-like neighbor, not the peptide
    smi = refined.canonical_smiles or refined.smiles or ""
    assert "CC(C)C" not in smi
    assert refined.source in {"neighbor_rescue", "hybrid"}


def test_weak_benzamidine_not_rescued_at_120():
    """Regression: ChemDFM empty + weak cos benzamidine [M]+ must not win."""
    seed = Node(id="0", mz=120.0807, name=None)
    weak = Node(
        id="1",
        mz=121.09,
        name="Benzamidine [M+H]",
        smiles="N=C(N)c1ccccc1",
    )
    edge = Edge(source="0", target="1", cosine=0.55, abs_diff_mz=1.009)
    ego = EgoContext(
        seed=seed,
        seed_mz=120.0807,
        neighbors=[NeighborEvidence(node=weak, edge=edge)],
    )
    empty = parse_model_output("", precursor_mz=120.0807, mass_tol_da=0.05)
    refined, notes = refine_with_neighborhood(empty, ego, mass_tol_da=0.05)
    smi = refined.canonical_smiles or refined.smiles or ""
    assert "N=C(N)" not in smi
    assert refined.source in {"abstain", "model"} or refined.smiles is None


@pytest.mark.skipif(not GRAPHML_MTCA.exists(), reason="MTCA GraphML missing")
def test_mtca_ego_ranks_near_isobars():
    net = load_graphml(GRAPHML_MTCA)
    ego = build_ego(net, seed_id="0", hide_seed_name=True)
    isobars = ego.near_isobars(0.5)
    assert len(isobars) >= 1
    # top evidence neighbor should have reasonably high score
    top = ego.top_neighbors[0]
    assert top.evidence_score(ego.seed_mz) > 0.5
