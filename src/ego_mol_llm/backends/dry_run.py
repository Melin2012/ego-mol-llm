"""Deterministic offline backend for tests and CI (no model download)."""

from __future__ import annotations

import json
import re

from ego_mol_llm.backends.base import GenerationConfig, LLMBackend


class DryRunBackend(LLMBackend):
    """Heuristic mock that looks for strong library evidence in the prompt."""

    name = "dry-run"

    def generate(self, messages: list[dict[str, str]], config: GenerationConfig | None = None) -> str:
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        # If CAS 5470-37-1 or tetrahydroharmane appears with high cosine, emit MTCA
        conf = 0.35
        smiles = None
        name = None
        formula = None
        rationale = "Dry-run backend: no neural model loaded."

        if re.search(r"5470-37-1|tetrahydroharmane|Tetrahydroharmane", user, re.I):
            smiles = "CC1NC(Cc2c1[nH]c1ccccc21)C(=O)O"
            name = "1-methyl-1,2,3,4-tetrahydro-beta-carboline-3-carboxylic acid"
            formula = "C13H14N2O2"
            conf = 0.82
            rationale = (
                "Dry-run heuristic: high-cosine neighbors include CAS 5470-37-1 / "
                "tetrahydroharmane-3-carboxylic acid at m/z ~229 and related indole/Trp "
                "context, consistent with MTCA [M-H]-."
            )
        elif re.search(r"C13H14N2O2", user):
            smiles = "CC1NC(Cc2c1[nH]c1ccccc21)C(=O)O"
            name = "putative C13H14N2O2 tetrahydro-beta-carboline carboxylic acid"
            formula = "C13H14N2O2"
            conf = 0.55
            rationale = "Dry-run heuristic: formula C13H14N2O2 appears among neighbors."

        payload = {
            "smiles": smiles,
            "iupac_or_common_name": name,
            "formula": formula,
            "adduct": "[M-H]-",
            "confidence": conf,
            "rationale": rationale,
            "alternatives": [],
        }
        return (
            "DRY-RUN reasoning complete.\n\n```json\n"
            + json.dumps(payload, indent=2)
            + "\n```"
        )
