# CFM-ID–first fragment explanation for ego networks

## Idea

Run **CFM-ID before the LLM**, on the **ego network** (not only final candidate re-rank):

1. **Neighbors with known SMILES + MS/MS**  
   - `cfm-annotate`: experimental peaks → fragment ion SMILES  
   - `cfm-predict`: structure↔spectrum cosine  
2. **Blind seed** (unknown structure)  
   - peaks that **match neighbor experimental m/z** inherit that neighbor’s CFM fragment label  
   - = “this seed peak is likely the same fragment chemistry as in annotated neighbor X”  
3. **LLM prompt** gets a structured block of peak → fragment chemistry  
   - plus offline rule MS/MS explain (losses, diagnostics)  
   - plus NIST / method / polarity  

The model no longer has to invent peak meanings from raw m/z lists alone.

```text
ego MGFs (seed + neighbors)
        │
        ▼
 CFM-ID annotate neighbors (structure known)
        │
        ▼
 transfer shared peaks → seed fragment map
        │
        ▼
 prompt = network + NIST + rule explain + CFM fragment map
        │
        ▼
 LLM structure proposal → mass gate → hybrid / CFM re-rank
```

## Why this is better than CFM re-rank only

| Approach | When it helps |
|----------|----------------|
| CFM re-rank candidates | After LLM already proposed SMILES |
| **CFM-first network explain** | **Before** LLM — shapes the hypothesis space |

## Commands

```powershell
cd C:\Users\AlexeyMelnik\Desktop\Codes\ego-mol-llm
$env:PYTHONPATH = ".\src"
$pack = "C:\Users\AlexeyMelnik\Desktop\Blind_holdout40b_ego_msms"

# Start Docker Desktop first; image: wishartlab/cfmid:latest
# Precompute for full pack and inject into jobs/prompts
python scripts/precompute_cfm_network_explain.py `
  --pack $pack `
  --max-neighbors 6 `
  --force `
  --inject-prompts

# Smoke 2 samples only
python scripts/precompute_cfm_network_explain.py --pack $pack --limit 2 --force --inject-prompts

# No Docker: rule fallback
python scripts/precompute_cfm_network_explain.py --pack $pack --no-cfm --inject-prompts
```

Outputs:

- `pack/cfm_explain/<SPECTRUMID>.json` — full structured explanation  
- `jobs/` + `prompts/` updated when `--inject-prompts`  

## Prompt block (example)

```text
=== CFM-ID / IN-SILICO FRAGMENT EXPLANATION (precomputed; offline) ===
  backend = cfmid
  • Seed peaks with CFM-transferred fragment SMILES: 13/40
  SEED experimental peaks → fragment chemistry (from network):
    m/z 76.0394 → [NH2+]=CC(O)O [transferred] shared with neighbor …
  Neighbor structure annotations (CFM-ID):
    • NAME: SMILES=... annotated=12/40 predict_cos=0.71
        peak 76.0394 → [NH2+]=CC(O)O
```

## Cost

- ~few seconds per neighbor (Docker). With 6 neighbors × 40 seeds ≈ **several minutes–tens of minutes**.  
- Cache-friendly if you re-run annotate on same SMILES (future: disk cache by SMILES hash).  
- **Optional** — pack works without it; this is the “fragment-aware” LLM path.

## Related

- `docs/INSILICO_RERANK.md` — CFM re-rank after candidates exist  
- `docs/SIRIUS.md` — optional CSI (not required)  
- `msms_explain.py` — always-on rule-based losses / diagnostics  
