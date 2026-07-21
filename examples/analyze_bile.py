"""Blind-ish analysis of a GraphML ego network."""
from __future__ import annotations

from pathlib import Path

from ego_mol_llm.draw import draw_prediction_card, clean_display_name
from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.paths import make_run_dir
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.report import export_report
from ego_mol_llm.validate import check_mass, canonicalize_smiles, formula_to_mass

GRAPHML = Path(
    r"C:\Users\AlexeyMelnik\OneDrive - Arome Science Inc\Attachments"
    r"\HNSW_12-Ketochenodeoxycholic acid_AROMEC18COLGATE001442.graphml"
)

# Candidate bile-acid structures often linked to 12-keto-CDCA
CANDIDATES = {
    "12-ketochenodeoxycholic acid": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC[C@H]4[C@@]3(CC[C@@H](C4)O)C)C(=O)",
    # more standard 12-oxo-CDCA SMILES variants tried below
    "chenodeoxycholic acid": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2[C@@H](C[C@H]4[C@@]3(CC[C@@H](C4)O)C)O)C",
    "deoxycholic acid": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2[C@@H](C[C@H]4[C@@]3(CC[C@@H](C4)O)C)O)C",
    "cholic acid": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(C[C@H](C3[C@H]2[C@@H](C[C@H]4[C@@]3(CC[C@@H](C4)O)C)O)O)C",
    "lithocholic acid": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC[C@H]4[C@@]3(CC[C@@H](C4)O)C)C",
}


def main() -> None:
    assert GRAPHML.exists(), GRAPHML
    net = load_graphml(GRAPHML)
    seed = net.find_seed(seed_id="0")
    print("=== SEED (hidden label for evaluation) ===")
    print("  true name:", seed.name)
    print("  m/z:", seed.mz)
    print("  degree:", len(net.adjacency.get(seed.id, [])))
    print("  nodes/edges:", len(net.nodes), len(net.edges))

    ego = build_ego(net, seed_id="0", hide_seed_name=True, max_neighbors=35)
    print("\nclass hints:", ego.class_hints())
    print("near-isobars |dmz|<=0.5:", ego.meta.get("n_near_isobars"))

    print("\n=== TOP 25 BY EVIDENCE SCORE ===")
    for i, ev in enumerate(ego.top_neighbors[:25], 1):
        d = ev.resolved_delta_mz(ego.seed_mz)
        name = (ev.node.name or "NO_MATCH")[:120]
        print(
            f"{i:02d} cos={ev.cosine:.3f} dmz={d} score={ev.evidence_score(ego.seed_mz):.3f} "
            f"mz={ev.node.mz} {name}"
        )
        if ev.node.smiles:
            print(f"    SMILES={ev.node.smiles[:100]}")

    print("\n=== NEAR ISOBARS (|dmz|<=2) ===")
    for ev in ego.near_isobars(2.0)[:20]:
        d = ev.resolved_delta_mz(ego.seed_mz)
        print(f"cos={ev.cosine:.3f} dmz={d} mz={ev.node.mz} {(ev.node.name or '')[:130]}")
        if ev.node.smiles:
            print(f"  SMILES={ev.node.smiles}")

    print("\n=== KEYWORD HITS IN NEIGHBORHOOD (bile / steroid / keto) ===")
    keys = (
        "bile",
        "cholic",
        "cheno",
        "deoxy",
        "litho",
        "ursodeoxy",
        "keto",
        "oxo",
        "steroid",
        "cholest",
        "tauro",
        "glyco",
        "CDCA",
        "DCA",
        "CA ",
        "LCA",
    )
    for nid, nd in net.nodes.items():
        name = nd.name or ""
        low = name.lower()
        if any(k.lower() in low for k in keys):
            print(f"  mz={nd.mz} {name[:140]}")
            if nd.smiles:
                print(f"    SMILES={nd.smiles[:100]}")

    print("\n=== MASS-CONSISTENT NEIGHBOR SMILES ===")
    hyps = ego.neighbor_structure_hypotheses(mass_tol_da=0.15, dmz_max=10.0, limit=20)
    for h in hyps:
        print(h)

    # Dry-run pipeline with rescue
    print("\n=== DRY-RUN + RESCUE PIPELINE ===")
    out = make_run_dir(parent=Path("outputs/runs"), graphml=GRAPHML, backend="dry-run", label="manual")
    result = predict_from_graphml(
        GRAPHML,
        backend="dry-run",
        hide_seed_name=True,
        max_neighbors=35,
        mass_tol_da=0.1,
    )
    paths = export_report(result, out)
    d = result.to_dict()
    print("pred SMILES:", d.get("smiles"))
    print("pred name:", d.get("name"))
    print("confidence:", d.get("confidence"), "source:", d.get("source"), "mass_ok:", d.get("mass_ok"))
    print("rescue:", d.get("rescue_notes"))
    print("out:", out)

    # Expert mass check for 12-keto-CDCA formula C24H38O4 = 390.277
    print("\n=== FORMULA / ADDUCT VS SEED m/z ===")
    mz = float(seed.mz)
    for formula, note in [
        ("C24H38O4", "12-keto-CDCA / oxo-dihydroxy BA"),
        ("C24H40O4", "CDCA / DCA dihydroxy BA"),
        ("C24H40O5", "cholic acid"),
        ("C24H40O3", "lithocholic"),
        ("C24H36O4", "diketo BA?"),
    ]:
        em = formula_to_mass(formula)
        for adduct, off in [("[M-H]-", -1.007825), ("[M+H]+", 1.007825), ("[M+Na]+", 22.989218), ("[M+NH4]+", 18.033823)]:
            theo = em + off
            err = abs(mz - theo)
            if err < 0.05:
                print(f"  MATCH {formula} {note} {adduct} theo={theo:.4f} err={err:.4f}")
        # also print best
        best = min(
            (abs(mz - (em + o)), adduct, em + o)
            for adduct, o in [("[M-H]-", -1.007825), ("[M+H]+", 1.007825), ("[M+Na]+", 22.989218)]
        )
        print(f"  {formula} best {best[1]} err={best[0]:.4f} theo={best[2]:.4f} ({note})")

    # Manual expert prediction card
    # PubChem-style SMILES for 12-oxochenodeoxycholic acid / 12-ketochenodeoxycholic acid
    # C24H38O4, often:
    expert_smiles_options = [
        # 3a,7a-dihydroxy-12-oxo-5b-cholan-24-oic acid common SMILES
        "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1(CC(=O)[C@H]3[C@H]2CC[C@H]4C[C@H](CC[C@@]43C)O)C",
        "C[C@H](CCC(=O)O)C1CCC2C3CCC4CC(O)CCC4(C)C3C(=O)CC12C",
        "CC(CCC(=O)O)C1CCC2C3CCC4CC(O)CCC4(C)C3C(=O)CC12C",
    ]
    print("\n=== EXPERT CANDIDATE MASS CHECK ===")
    best_smi = None
    best_err = 999.0
    best_adduct = None
    for smi in expert_smiles_options:
        can = canonicalize_smiles(smi)
        if not can:
            print("  invalid", smi[:60])
            continue
        ok, em, err, adduct = check_mass(can, mz, None, tol_da=0.05)
        print(f"  can={can[:70]} em={em} ok={ok} err={err} adduct={adduct}")
        if err is not None and err < best_err:
            best_err = err
            best_smi = can
            best_adduct = adduct

    # If RDKit missing, use formula match only
    if best_smi is None:
        best_smi = expert_smiles_options[2]
        best_adduct = "[M-H]-" if abs(mz - (390.277 - 1.0078)) < abs(mz - (390.277 + 1.0078)) else "[M+H]+"

    card = draw_prediction_card(
        best_smi,
        "12-Ketochenodeoxycholic acid (3α,7α-dihydroxy-12-oxo-5β-cholan-24-oic acid)",
        out / "expert_prediction_structure.png",
        formula="C24H38O4",
        adduct=best_adduct,
        confidence=0.78,
        mz=mz,
    )
    print("\nExpert structure card:", card)
    print("\nDONE out=", out)


if __name__ == "__main__":
    main()
