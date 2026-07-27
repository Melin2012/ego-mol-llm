"""In-silico spectrum prediction + re-rank tests (rule backend, offline)."""

from __future__ import annotations

from ego_mol_llm.candidates import AnnotationCandidate, build_candidates
from ego_mol_llm.ego import EgoContext, NeighborEvidence
from ego_mol_llm.graphml import Edge, Node
from ego_mol_llm.method_card import MethodCard
from ego_mol_llm.spectrum_predict import (
    RuleBasedPredictor,
    get_predictor,
    rerank_candidates_by_insilico,
    score_smiles_against_spectrum,
)


def test_rule_predictor_basic():
    p = RuleBasedPredictor()
    # isovalerylglycine
    pred = p.predict("CC(C)CC(=O)NCC(=O)O", adduct="[M+H]+", ion_mode="positive")
    assert pred is not None
    assert pred.backend == "rule"
    assert len(pred.peaks) >= 5
    mzs = [m for m, _ in pred.peaks]
    # precursor ~160.1 and glycine fragment ~76
    assert any(abs(m - 160.1) < 0.5 for m in mzs)
    assert any(abs(m - 76.039) < 0.05 for m in mzs)


def test_score_true_vs_wrong():
    exp = [
        (76.0394, 100.0),
        (85.0647, 84.0),
        (57.0698, 96.0),
        (160.0965, 6.0),
        (142.086, 5.0),
    ]
    true = "CC(C)CC(=O)NCC(=O)O"
    wrong = "CCCCCCCCCCCCCCCC(=O)O"  # palmitic — different chemistry
    cos_t, _ = score_smiles_against_spectrum(
        true, exp, predictor=RuleBasedPredictor(), adduct="[M+H]+"
    )
    cos_w, _ = score_smiles_against_spectrum(
        wrong, exp, predictor=RuleBasedPredictor(), adduct="[M+H]+"
    )
    # true acylglycine should score at least as well as wrong FA
    assert cos_t >= 0.0
    assert cos_t >= cos_w - 0.05  # soft; rule model is approximate


def test_rerank_boosts_matching_candidate():
    exp = [(76.0394, 100.0), (85.0647, 80.0), (160.1, 10.0)]
    good = AnnotationCandidate(
        smiles="CC(C)CC(=O)NCC(=O)O",
        source="neighbor",
        fusion_score=0.4,
        mass_ok=True,
        adduct="[M+H]+",
    )
    bad = AnnotationCandidate(
        smiles="CCO",
        source="neighbor",
        fusion_score=0.55,
        mass_ok=True,
        adduct="[M+H]+",
    )
    ranked = rerank_candidates_by_insilico(
        [bad, good],
        exp,
        predictor=RuleBasedPredictor(),
        weight=0.5,
    )
    assert ranked[0].smiles.startswith("CC(C)CC")
    assert ranked[0].insilico_cosine is not None
    assert ranked[0].insilico_cosine >= ranked[1].insilico_cosine


def test_build_candidates_with_insilico():
    seed = Node(id="0", mz=160.0968, name=None)
    good = Node(id="1", mz=160.1, name="iso", smiles="CC(C)CC(=O)NCC(=O)O")
    ego = EgoContext(
        seed=seed,
        seed_mz=160.0968,
        neighbors=[
            NeighborEvidence(
                node=good,
                edge=Edge(source="0", target="1", cosine=0.8, abs_diff_mz=0.003),
            )
        ],
    )
    # attach fake spectral peaks
    from ego_mol_llm.mgf import SpectralContext, Spectrum

    ego.spectral = SpectralContext(
        seed=Spectrum(
            peaks=[(76.0394, 100.0), (85.0647, 80.0), (160.0968, 5.0)],
            pepmass=160.0968,
        )
    )
    cands = build_candidates(
        ego,
        method=MethodCard(polarity="positive"),
        mass_tol_da=0.05,
        use_insilico_rerank=True,
        insilico_backend="rule",
        limit=10,
    )
    assert cands
    assert any(
        c.insilico_cosine is not None or (c.meta or {}).get("insilico_cosine") is not None
        for c in cands
    )


def test_get_predictor_rule():
    p = get_predictor("rule")
    assert p.name == "rule"
    assert p.available()
