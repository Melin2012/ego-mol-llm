"""Run dry-run pipeline + print ego evidence for one GraphML."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ego_mol_llm.draw import clean_display_name, draw_prediction_card
from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.paths import make_run_dir
from ego_mol_llm.predict import predict_from_graphml
from ego_mol_llm.report import export_report
from ego_mol_llm.validate import formula_to_mass


def main(path: str) -> None:
    g = Path(path)
    assert g.exists(), g
    net = load_graphml(g)
    seed = net.find_seed(seed_id="0")
    print("=== SEED (eval) ===")
    print("true name:", seed.name)
    print("m/z:", seed.mz, "degree:", len(net.adjacency.get(seed.id, [])))
    print("nodes/edges:", len(net.nodes), len(net.edges))

    ego = build_ego(net, seed_id="0", hide_seed_name=True, max_neighbors=30)
    print("class hints:", ego.class_hints())
    print("near-isobars:", ego.meta.get("n_near_isobars"))
    print("half-mass n:", ego.meta.get("n_half_mass_neighbors"))

    print("\n=== TOP 15 EVIDENCE ===")
    for i, ev in enumerate(ego.top_neighbors[:15], 1):
        d = ev.resolved_delta_mz(ego.seed_mz)
        hd = ev.half_mass_delta(ego.seed_mz)
        print(
            f"{i:02d} cos={ev.cosine:.3f} dmz={d} half={hd} "
            f"score={ev.evidence_score(ego.seed_mz):.3f} mz={ev.node.mz}"
        )
        print(f"    {(ev.node.name or 'NO_MATCH')[:130]}")
        if ev.node.smiles:
            print(f"    SMILES={ev.node.smiles[:100]}")

    print("\n=== NEAR ISOBARS ===")
    for ev in ego.near_isobars(1.0)[:12]:
        print(
            f"cos={ev.cosine:.3f} dmz={ev.resolved_delta_mz(ego.seed_mz)} "
            f"mz={ev.node.mz} {(ev.node.name or '')[:120]}"
        )
        if ev.node.smiles:
            print(f"  SMILES={ev.node.smiles}")

    print("\n=== HYPS ===")
    hyps = ego.neighbor_structure_hypotheses(mass_tol_da=0.1, limit=10)
    for h in hyps:
        print(
            f"adduct={h.get('adduct')} err={h.get('mass_error_da')} "
            f"half={h.get('half_mass_delta')} conf={h.get('confidence')}"
        )
        print(f"  name={(h.get('name') or '')[:100]}")
        print(f"  smi={h.get('smiles')}")

    if seed.mz is not None:
        print("\n=== FORMULA VS m/z ===")
        mz = float(seed.mz)
        for formula in [
            "C8H11NO",
            "C8H9NO",
            "C7H9NO",
            "C8H11N",
            "C9H13NO",
            "C8H10N2",
        ]:
            em = formula_to_mass(formula)
            if em is None:
                continue
            for adduct, theo in [
                ("[M-H]-", em - 1.007825),
                ("[M+H]+", em + 1.007825),
                ("[M+Na]+", em + 22.989218),
                ("[2M+H]+", 2 * em + 1.007825),
                ("[2M-H]-", 2 * em - 1.007825),
            ]:
                err = abs(mz - theo)
                if err < 0.05:
                    print(f"  MATCH {formula} {adduct} theo={theo:.4f} err={err:.4f}")

    out = make_run_dir(parent=Path("outputs/runs"), graphml=g, backend="dry-run", label="pred")
    result = predict_from_graphml(
        g, backend="dry-run", hide_seed_name=True, max_neighbors=30, mass_tol_da=0.1
    )
    export_report(result, out)
    d = result.to_dict()
    print("\n=== PIPELINE RESULT ===")
    print(json.dumps({k: d.get(k) for k in [
        "smiles", "name", "formula", "adduct", "matched_adduct",
        "confidence", "mass_ok", "mass_error_da", "source", "seed_mz",
        "rescue_notes",
    ]}, indent=2))
    print("out:", out)

    smi = d.get("smiles")
    if smi:
        draw_prediction_card(
            smi,
            clean_display_name(d.get("name")) or "prediction",
            out / "structure.png",
            formula=d.get("formula"),
            adduct=d.get("adduct") or d.get("matched_adduct"),
            confidence=d.get("confidence"),
            mz=d.get("seed_mz"),
        )


if __name__ == "__main__":
    main(sys.argv[1])
