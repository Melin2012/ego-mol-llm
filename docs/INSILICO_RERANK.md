# In-silico spectrum re-ranking (offline)

Score **candidate SMILES** by how well their **predicted MS/MS** matches the
**experimental** seed spectrum. Complements (does not replace) NIST reverse
search and offline peak explanation.

## Backends

| Backend | Needs | Speed | Quality |
|---------|--------|-------|---------|
| **`rule`** (default) | RDKit only | ms / candidate | Modest; good for isomer filter |
| **`cfmid`** | Docker + `wishartlab/cfmid` | ~1–10 s / candidate (cached) | Strong (CFM-ID 4) |
| **`iceberg`** | `ms_pred` + checkpoints | GPU preferred | Strong when installed |
| **`auto`** | — | CFM-ID if Docker up, else rule | — |

**SIRIUS is separate** (optional, login, slow on large molecules). This path is
designed to run offline without it.

## Product integration

```python
from ego_mol_llm.candidates import select_product_annotation

pred, notes, cands = select_product_annotation(
    ego,
    model_pred,
    library_hits=library_hits,
    use_insilico_rerank=True,
    insilico_backend="rule",   # or "cfmid" / "auto"
    insilico_weight=0.35,
)
# cands[i].insilico_cosine, fusion_score updated
```

Fusion:

```text
fusion ← (1 − w) · fusion_old + w · cosine(predicted, experimental)
```

Default `w = 0.35`. Model still kept when mass-OK (conservative product policy).

## Pack script (holdout-40)

```powershell
cd C:\Users\AlexeyMelnik\Desktop\Codes\ego-mol-llm
$env:PYTHONPATH = ".\src"
$pack = "C:\Users\AlexeyMelnik\Desktop\Blind_holdout40_ego_msms"

# Always-available rule model
python scripts/apply_insilico_rerank_to_pack.py `
  --pack $pack `
  --preds-in predictions `
  --preds-out predictions_opus_insilico_rule `
  --backend rule `
  --model-label opus-insilico-rule

# CFM-ID when Docker Desktop is running
python scripts/apply_insilico_rerank_to_pack.py `
  --pack $pack `
  --preds-in predictions `
  --preds-out predictions_opus_insilico_cfmid `
  --backend cfmid
```

## Enable CFM-ID

1. Start **Docker Desktop**
2. Pull once: `docker pull wishartlab/cfmid:latest`
3. Use `--backend cfmid` (results cached under `%USERPROFILE%\.cache\ego_mol_cfmid`)

## ICEBERG

Install Coley lab [ms-pred / ICEBERG](https://github.com/coleygroup/ms-pred) and
checkpoints separately. Backend is scaffolded (`IcebergPredictor`); wire local
checkpoint paths when ready — until then `available()` is false and auto falls
back to rule/CFM-ID.
