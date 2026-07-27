# SIRIUS + CSI:FingerID in ego-mol-llm

Independent **spectral structure** evidence for annotation propagation, alongside ego neighbors and NIST reverse search.

## License (read this)

| Component | Status |
|-----------|--------|
| SIRIUS client (CLI/GUI) | Open source (**AGPL**) |
| CSI:FingerID / CANOPUS web services | Free for **academic / non-commercial**; **commercial → [Bright Giant](https://bright-giant.com)** |

Do not ship CSI web-service results as a commercial product without a proper license.

## Install + login

1. Download SIRIUS 5 or 6 from [sirius.bio](https://bio.informatik.uni-jena.de/software/sirius/) / [GitHub releases](https://github.com/sirius-ms/sirius/releases).
2. Put the CLI on `PATH`, or set:

```powershell
$env:SIRIUS_BIN = "C:\path\to\sirius.bat"
```

3. Login once (required for structure / CSI:FingerID):

```powershell
sirius login
# or: & $env:SIRIUS_BIN login
```

## Run on holdout pack

```powershell
cd C:\Users\AlexeyMelnik\Desktop\Codes\ego-mol-llm
$env:PYTHONPATH = ".\src"
$pack = "C:\Users\AlexeyMelnik\Desktop\Blind_holdout40_ego_msms"

# 1) Write .ms + run SIRIUS (all 40). Use --limit 2 to smoke-test.
python scripts/run_sirius_on_pack.py --pack $pack --profile orbitrap

# 2) Product fusion (NIST + SIRIUS + neighbors) on pure model preds
python scripts/apply_sirius_product_to_pack.py `
  --pack $pack `
  --preds-in predictions `
  --preds-out predictions_opus_sirius_product `
  --model-label opus-sirius-product-v03
```

Outputs:

- `pack/sirius_work/<ID>/` — `.ms`, project space, local `hits.json`
- `pack/sirius_hits/<ID>.json` — parsed CSI/structure ranks for fusion
- `pack/sirius_run_summary.json`

## Dry-run / offline parse

```powershell
# Only write .ms and print CLI (no SIRIUS install needed)
python scripts/run_sirius_on_pack.py --pack $pack --dry-run

# Re-parse existing project trees after a manual GUI/CLI run
python scripts/run_sirius_on_pack.py --pack $pack --parse-existing-only --force
```

## How fusion uses SIRIUS

1. Mass gate (polarity-aware method card)
2. CSI confidence / score → fusion component
3. If model SMILES is mass-OK → **keep model** (SIRIUS listed as hybrid alternative)
4. If model empty / mass-fail → may select `source=sirius_csi` when it wins fusion

## In-code API

```python
from ego_mol_llm.predict import predict_from_graphml

result = predict_from_graphml(
    graphml_path,
    backend="openai",
    model="...",
    seed_mgf=seed_mgf,
    mgf_paths=[network_mgf],
    use_sirius=True,
    sirius_bin=r"C:\path\to\sirius.bat",
    sirius_work_dir="outputs/sirius/run1",
)
```

Or pass precomputed hits:

```python
predict_ego(..., sirius_hits=list_of_SiriusHit_or_dicts)
```
