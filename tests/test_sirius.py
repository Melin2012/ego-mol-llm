"""SIRIUS MS export + summary parsing (no CLI required)."""

from __future__ import annotations

from pathlib import Path

from ego_mol_llm.candidates import build_candidates
from ego_mol_llm.ego import EgoContext, NeighborEvidence
from ego_mol_llm.graphml import Edge, Node
from ego_mol_llm.method_card import MethodCard
from ego_mol_llm.sirius import (
    SiriusHit,
    build_sirius_command,
    hits_to_prompt_block,
    parse_structure_tsv,
    write_sirius_ms,
)


def test_write_sirius_ms(tmp_path: Path):
    p = write_sirius_ms(
        tmp_path / "x.ms",
        compound_id="TEST001",
        precursor_mz=160.0968,
        peaks=[(76.0394, 100.0), (57.0698, 90.0), (85.0647, 80.0)],
        ion_mode="positive",
    )
    text = p.read_text(encoding="utf-8")
    assert ">compound TEST001" in text
    assert ">parentmass 160.096800" in text
    assert ">ionization [M+H]+" in text
    assert ">ms2" in text
    assert "76.039400" in text


def test_write_sirius_ms_negative(tmp_path: Path):
    p = write_sirius_ms(
        tmp_path / "n.ms",
        compound_id="NEG",
        precursor_mz=151.04,
        peaks=[(107.05, 50.0)],
        ion_mode="negative",
    )
    assert "[M-H]-" in p.read_text(encoding="utf-8")


def test_parse_structure_tsv(tmp_path: Path):
    tsv = tmp_path / "structure_identifications.tsv"
    tsv.write_text(
        "rank\tmolecularFormula\tadduct\tsmiles\tInChIKey\tname\tCSI:FingerIDScore\tConfidenceScore\n"
        "1\tC7H13NO3\t[M+H]+\tCC(C)CC(=O)NCC(=O)O\tZRQXMKMBBMNNQC-UHFFFAOYSA-N\t"
        "N-Isovaleroylglycine\t-45.2\t0.82\n"
        "2\tC7H13NO3\t[M+H]+\tCCC(C)C(=O)NCC(=O)O\tXXXX\talt\t-80.0\t0.11\n",
        encoding="utf-8",
    )
    hits = parse_structure_tsv(tsv, top_k=5)
    assert len(hits) == 2
    assert hits[0].smiles.startswith("CC(C)CC")
    assert hits[0].formula == "C7H13NO3"
    assert hits[0].source == "csi_fingerid"
    assert hits[0].confidence == 0.82


def test_build_candidates_includes_sirius():
    seed = Node(id="0", mz=160.0968, name=None)
    ego = EgoContext(seed=seed, seed_mz=160.0968, neighbors=[])
    hits = [
        SiriusHit(
            rank=1,
            smiles="CC(C)CC(=O)NCC(=O)O",
            formula="C7H13NO3",
            adduct="[M+H]+",
            csi_score=-40.0,
            confidence=0.9,
            name="isovalerylglycine",
        )
    ]
    cands = build_candidates(
        ego,
        sirius_hits=hits,
        method=MethodCard(polarity="positive"),
        mass_tol_da=0.05,
        limit=10,
    )
    assert any(c.source == "sirius" and c.mass_ok is True for c in cands)
    assert cands[0].source == "sirius"


def test_prompt_block():
    lines = hits_to_prompt_block(
        [
            SiriusHit(
                rank=1,
                smiles="CCO",
                formula="C2H6O",
                adduct="[M+H]+",
                confidence=0.5,
            )
        ]
    )
    assert any("SIRIUS" in x for x in lines)
    assert any("CCO" in x for x in lines)


def test_build_command(tmp_path: Path):
    cmd = build_sirius_command(
        sirius_bin=Path("sirius"),
        ms_path=tmp_path / "a.ms",
        project_dir=tmp_path / "p.sirius",
        summaries_dir=tmp_path / "sum",
        profile="orbitrap",
    )
    assert cmd[0] == "sirius"
    assert "formula" in cmd
    assert "structure" in cmd
    assert "write-summaries" in cmd
