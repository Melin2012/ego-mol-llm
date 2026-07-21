from pathlib import Path

import pytest

from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.prompts import build_messages
from ego_mol_llm.validate import parse_model_output

GRAPHML = Path(
    r"C:\Users\AlexeyMelnik\Downloads\HNSW_1-Methyl-1,2,3,4-tetrahydro-beta-carboline-3-carboxylic acid_AROMEC18COLGATE001635.graphml"
)

pytestmark = pytest.mark.skipif(not GRAPHML.exists(), reason="example GraphML not present")


def test_load_graphml_counts():
    net = load_graphml(GRAPHML)
    assert len(net.nodes) > 100
    assert len(net.edges) > 100
    seed = net.find_seed(seed_id="0")
    assert seed.mz is not None
    assert abs(seed.mz - 229.098) < 0.01


def test_blind_ego_hides_name():
    net = load_graphml(GRAPHML)
    ego = build_ego(net, seed_id="0", hide_seed_name=True)
    assert ego.seed.name is None
    assert ego.seed_mz is not None
    assert len(ego.neighbors) > 0
    assert ego.meta.get("true_seed_name")


def test_prompt_contains_neighbors_not_seed_name():
    net = load_graphml(GRAPHML)
    ego = build_ego(net, seed_id="0", hide_seed_name=True)
    messages = build_messages(ego)
    user = messages[1]["content"]
    assert "229" in user
    # true name should not appear when blinded
    assert "AROMEC18COLGATE001635" not in user
    assert "edge_cos=" in user or "cosine=" in user


def test_dry_run_predict_mtca_like():
    result = predict_from_graphml(GRAPHML, backend="dry-run", hide_seed_name=True)
    d = result.to_dict()
    assert d["confidence"] is not None and d["confidence"] >= 0.5
    assert d["smiles"] is not None
    # formula if present
    if d["formula"]:
        assert "C13" in d["formula"] or "C" in d["formula"]


def test_parse_json_block():
    text = """
Some reasoning...
```json
{
  "smiles": "CCO",
  "iupac_or_common_name": "ethanol",
  "formula": "C2H6O",
  "adduct": "[M+H]+",
  "confidence": 0.4,
  "rationale": "test",
  "alternatives": []
}
```
"""
    p = parse_model_output(text, precursor_mz=47.049)
    assert p.smiles == "CCO"
    assert p.confidence == 0.4
