# SIRIUS-first fragment explanation for ego networks

## Idea

Same role as **CFM-ID–first**, but peak chemistry is **SIRIUS subformula / fragmentation chemistry** (not CFM fragment SMILES):

1. **Neighbors with known SMILES + MS/MS**  
   - RDKit parent formula from SMILES  
   - `sirius decomp --parent FORMULA` on experimental peaks → peak → **fragment formula**
2. **Blind seed**  
   - peaks that match neighbor m/z **inherit** that fragment formula  
   - if `sirius_hits/<id>.json` exists, also decomp seed peaks under the top CSI/SIRIUS formula  
3. **LLM prompt** gets a structured peak → formula block before proposing SMILES  

```text
ego MGFs (seed + neighbors) + optional sirius_hits
        │
        ▼
 SIRIUS decomp on neighbors (structure known → parent formula)
        │
        ▼
 transfer shared peaks → seed fragment FORMULA map
        │
        ▼
 prompt = network + NIST + rule explain + SIRIUS fragment map
        │
        ▼
 LLM (Grok) structure proposal → mass gate → product hybrid
```

## Commands

```powershell
cd C:\Users\AlexeyMelnik\Desktop\Codes\ego-mol-llm
$env:PYTHONPATH = ".\src"
$env:SIRIUS_BIN = "C:\Program Files\sirius\sirius.bat"
$pack = "C:\Users\AlexeyMelnik\Desktop\Blind_holdout40b_ego_msms"

# Prefer: full CSI pack already run (sirius_hits/) for seed parent formula
python scripts/precompute_sirius_network_explain.py `
  --pack $pack `
  --sirius-bin $env:SIRIUS_BIN `
  --max-neighbors 6 `
  --force `
  --inject-prompts
```

Outputs:

- `pack/sirius_explain/<SPECTRUMID>.json`
- `jobs/` + `prompts/` updated when `--inject-prompts` (product_version `0.4-sirius-first`)

## vs CFM-first

| | CFM-first | SIRIUS-first |
|--|-----------|--------------|
| Peak label | fragment SMILES | fragment **formula** |
| Needs | Docker + CFM image | Local SIRIUS CLI (`decomp`) |
| Login | no | no (decomp only) |
| Best for | substructure SMILES clues | elemental / loss reasoning |

You can keep **both** blocks in the prompt (CFM + SIRIUS).

## Related

- `docs/CFM_NETWORK_FIRST.md`
- `docs/SIRIUS.md` — CSI structure IDs (product fusion)
- `msms_explain.py` — always-on rule losses  
