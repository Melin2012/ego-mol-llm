import json
from pathlib import Path

from ego_mol_llm.draw import draw_prediction_card

smi = (
    "O[C@H]1CC[C@@]2(C)[C@H](C[C@H](O)[C@]([C@H]3[C@@]4(C)"
    "[C@@H]([C@H](C)CCC(O)=O)CC3)([H])[C@@H]2CC4=O)C1"
)
out = Path("outputs/runs/expert_12keto_cdca")
out.mkdir(parents=True, exist_ok=True)
p = draw_prediction_card(
    smi,
    "12-Ketochenodeoxycholic acid (3α,7α-dihydroxy-12-oxo-5β-cholan-24-oic acid)",
    out / "structure.png",
    formula="C24H38O5",
    adduct="[2M+H]+",
    confidence=0.86,
    mz=813.551,
)
pred = {
    "smiles": smi,
    "name": "12-Ketochenodeoxycholic acid",
    "iupac_or_common_name": "3alpha,7alpha-dihydroxy-12-oxo-5beta-cholan-24-oic acid",
    "formula": "C24H38O5",
    "adduct": "[2M+H]+",
    "monomer_mz_calc": 407.280,
    "seed_mz": 813.551,
    "confidence": 0.86,
    "rationale": (
        "Seed m/z 813.551 matches [2M+H]+ of C24H38O5 (err ~0.0005 Da). "
        "Network is a pure oxo-bile-acid family: many high-cosine neighbors at m/z ~407 "
        "annotated as 3a,7a-dihydroxy-12-oxo-5b-cholanic acid / 12-oxo-CDCA scaffold "
        "(cosine 0.80-0.93). Half-mass 406.27 is the monomer of that scaffold. "
        "No near-isobar at 813 because library hits are monomers — ChemDFM fails if it "
        "proposes monomer SMILES against dimer precursor mass. "
        "3a,7a-diol-12-one is the definition of 12-ketochenodeoxycholic acid."
    ),
    "alternatives": [
        {
            "name": "7-Ketodeoxycholic acid",
            "formula": "C24H38O5",
            "confidence": 0.08,
            "note": "same formula; keto at C7 not C12 (also in network)",
        },
        {
            "name": "3-Hydroxy-7,12-diketocholanoic acid",
            "confidence": 0.04,
            "note": "related oxo-BA family member in network",
        },
        {
            "name": "Cholic acid",
            "formula": "C24H40O5",
            "confidence": 0.02,
            "note": "[2M+H]+ would be ~817.58, not 813.55",
        },
    ],
}
(out / "prediction.json").write_text(json.dumps(pred, indent=2), encoding="utf-8")
(out / "prediction.md").write_text(
    "\n".join(
        [
            "# Expert prediction: 12-Ketochenodeoxycholic acid",
            "",
            pred["rationale"],
            "",
            f"- **Name:** {pred['name']}",
            f"- **IUPAC-style:** {pred['iupac_or_common_name']}",
            f"- **Formula:** `{pred['formula']}`",
            f"- **Adduct:** `{pred['adduct']}` (monomer [M+H]+ ~ 407.28)",
            f"- **SMILES:** `{smi}`",
            f"- **Confidence:** {pred['confidence']}",
            f"- **Structure:** `{p}`",
        ]
    ),
    encoding="utf-8",
)
print("wrote", out.resolve())
print("structure", p)
