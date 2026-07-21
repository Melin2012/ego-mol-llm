from pathlib import Path

import pytest

from ego_mol_llm.mgf import (
    build_spectral_context,
    cosine_peaks,
    diagnostic_ions,
    parse_mgf,
)
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.prompts import build_messages
from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml

PCA = Path(r"C:\Users\AlexeyMelnik\Downloads\PCA_anonimized")
GRAPHML = PCA / "Bile_Acid_HNSW_test_1.graphml"
EGO_MGF = PCA / "Ego_MSMS.mgf"
ALL_MGF = PCA / "subgraph_molecules_with_node_ids.mgf"

pytestmark = pytest.mark.skipif(
    not GRAPHML.exists() or not EGO_MGF.exists(),
    reason="PCA_anonimized dataset not present",
)


def test_parse_ego_mgf():
    specs = parse_mgf(EGO_MGF)
    assert len(specs) == 1
    assert specs[0].pepmass is not None
    assert abs(specs[0].pepmass - 556.363) < 0.01
    assert len(specs[0].peaks) > 20
    diag = diagnostic_ions(specs[0].peaks)
    assert "Phe_immonium" in diag or "Phe_related_166" in diag


def test_parse_network_mgf_and_cosine():
    ego = parse_mgf(EGO_MGF)[0]
    all_sp = parse_mgf(ALL_MGF)
    assert len(all_sp) > 100
    # self-cosine
    c = cosine_peaks(ego.peaks, ego.peaks)
    assert c > 0.99


def test_spectral_context_and_prompt():
    net = load_graphml(GRAPHML)
    ego = build_ego(net, seed_id="0", hide_seed_name=True, max_neighbors=25)
    ego.spectral = build_spectral_context(
        seed_id="0",
        seed_mz=ego.seed_mz,
        neighbor_ids=[ev.node.id for ev in ego.neighbors],
        mgf_paths=[ALL_MGF],
        seed_mgf=EGO_MGF,
    )
    assert ego.spectral.seed is not None
    assert len(ego.spectral.neighbor_msms_cosine) >= 5
    msgs = build_messages(ego)
    user = msgs[1]["content"]
    assert "QUERY MS/MS" in user
    assert "top peaks" in user
    assert "msms_cos=" in user or "MS/MS-SIMILAR" in user


def test_predict_with_mgf_dry_run():
    result = predict_from_graphml(
        GRAPHML,
        backend="dry-run",
        seed_id="0",
        hide_seed_name=True,
        mgf_paths=[ALL_MGF],
        seed_mgf=EGO_MGF,
    )
    d = result.to_dict()
    assert d["msms_used"] is True
    assert d["spectral"] is not None
    assert d["spectral"]["seed_n_peaks"] > 20
    # prompt should have been built with MS/MS
    assert any("MS/MS" in m["content"] for m in result.messages)
