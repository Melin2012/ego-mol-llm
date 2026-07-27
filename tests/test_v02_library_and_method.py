"""v0.2: library index, RT priors, hybrid candidates."""

from __future__ import annotations

from pathlib import Path

from ego_mol_llm.candidates import build_candidates
from ego_mol_llm.ego import EgoContext, NeighborEvidence
from ego_mol_llm.graphml import Edge, Node
from ego_mol_llm.library_search import LibraryRecord, SpectralLibraryIndex
from ego_mol_llm.method_card import MethodCard, rt_compatibility
from ego_mol_llm.mgf import build_spectral_context


def test_rt_compatibility_relative():
    # seed near neighbors → high
    s = rt_compatibility(100.0, [95.0, 105.0, 110.0], card=MethodCard())
    assert s >= 0.7
    # far from neighbors → lower
    s2 = rt_compatibility(100.0, [500.0, 520.0], card=MethodCard())
    assert s2 < s


def test_method_adduct_prior_pos():
    m = MethodCard(polarity="positive")
    assert m.adduct_prior_ok("[M+H]+")
    assert not m.adduct_prior_ok("[M-H]-")


def test_resolve_method_polarity_prefers_seed():
    from ego_mol_llm.method_card import resolve_method_polarity

    # Spectrum IONMODE wins over study-level default "positive"
    assert (
        resolve_method_polarity(
            seed_ion_mode="negative",
            method=MethodCard(polarity="positive"),
        )
        == "negative"
    )
    assert (
        resolve_method_polarity(
            seed_ion_mode="positive",
            method=MethodCard(polarity="both"),
        )
        == "positive"
    )


def test_library_search_filters_opposite_polarity_by_adduct():
    from ego_mol_llm.library_search import LibraryRecord, SpectralLibraryIndex

    peaks = [(45.0, 100.0), (31.0, 40.0)]
    pos = LibraryRecord(
        pepmass=151.04,
        name="pos_hit",
        smiles="CC(=O)c1ccc(O)cc1O",
        precursor_type="[M+H]+",
        ion_mode=None,  # force adduct-based polarity
        top_peaks=peaks,
    )
    neg = LibraryRecord(
        pepmass=151.04,
        name="neg_hit",
        smiles="CC(=O)c1ccc(O)cc1O",
        precursor_type="[M-H]-",
        ion_mode=None,
        top_peaks=peaks,
    )
    key = int(round(151.04 / 0.01))
    idx = SpectralLibraryIndex(bins={key: [pos, neg]}, n_records=2)
    pos_hits = idx.search(peaks, 151.04, ion_mode="positive", min_score=0.1, precursor_tol_da=0.05)
    neg_hits = idx.search(peaks, 151.04, ion_mode="negative", min_score=0.1, precursor_tol_da=0.05)
    assert all(h.record.name == "pos_hit" for h in pos_hits)
    assert all(h.record.name == "neg_hit" for h in neg_hits)


def test_library_index_search_roundtrip(tmp_path: Path):
    # tiny fake library: ethanol-like peaks
    peaks = [(45.0, 100.0), (31.0, 40.0), (29.0, 20.0)]
    rec = LibraryRecord(
        pepmass=47.049,
        name="ethanol_proxy",
        smiles="CCO",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        precursor_type="[M+H]+",
        ion_mode="positive",
        top_peaks=peaks,
        source_library="test",
    )
    idx = SpectralLibraryIndex(bins={int(round(47.049 / 0.01)): [rec]}, n_records=1)
    hits = idx.search(peaks, 47.049, top_k=3, precursor_tol_da=0.05, min_score=0.2)
    assert hits
    assert hits[0].record.name == "ethanol_proxy"
    assert hits[0].match_score > 0.5
    p = tmp_path / "t.pkl"
    idx.save(p)
    idx2 = SpectralLibraryIndex.load(p)
    assert idx2.n_records == 1


def test_build_candidates_includes_neighbor():
    seed = Node(id="0", mz=115.05, name=None)
    good = Node(id="1", mz=115.05, name="hydantoin", smiles="CN1CC(=O)NC1=O")
    ego = EgoContext(
        seed=seed,
        seed_mz=115.05,
        neighbors=[
            NeighborEvidence(
                node=good,
                edge=Edge(source="0", target="1", cosine=0.92, abs_diff_mz=0.0001),
            )
        ],
    )
    cands = build_candidates(ego, mass_tol_da=0.05, limit=10)
    assert any(c.smiles and "N1CC" in (c.smiles or "") for c in cands) or any(
        c.source == "neighbor" for c in cands
    )


def test_spectral_context_parses_rt(tmp_path: Path):
    mgf = tmp_path / "s.mgf"
    mgf.write_text(
        """BEGIN IONS
NETWORK_NODE_ID=0
PEPMASS=200.1
RTINSECONDS=123.4
IONMODE=Positive
MSV_LIB=MSV0000
100.0 10
200.1 100
END IONS
BEGIN IONS
NETWORK_NODE_ID=1
PEPMASS=200.1
RTINSECONDS=130.0
MSV_LIB=MSV0000
100.0 12
200.1 90
END IONS
""",
        encoding="utf-8",
    )
    ctx = build_spectral_context(
        seed_id="0",
        seed_mz=200.1,
        neighbor_ids=["1"],
        mgf_paths=[mgf],
    )
    assert ctx.seed is not None
    assert ctx.seed_rt is not None and abs(ctx.seed_rt - 123.4) < 0.1
    assert "1" in ctx.neighbor_rt
    assert ctx.neighbor_meta.get("1", {}).get("msv_lib") == "MSV0000"
