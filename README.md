# ego-mol-llm

**Blind structure prediction of unknown metabolites from MS/MS molecular-network ego neighborhoods**, using open chemistry LLMs (ChemDFM on Qwen2.5, optional Qwen3.5 / Qwen2.5 instruct).

> Publication-oriented, open-source toolkit: GraphML in → SMILES + confidence + reproducible report out.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)

---

## Why this exists

Spectral molecular networks (GNPS / HNSW-style GraphML) place an **unknown feature** next to library hits by MS/MS cosine similarity. Human analysts use that ego neighborhood to guess structure — this package automates that step with **open post-trained chemistry LLMs**, without leaking the seed’s library name into the prompt (blind evaluation mode).

**Not** a de novo MS/MS → SMILES model from raw peaks.  
**Is** annotation *propagation + chemical reasoning* over network context (neighbors, Δm/z, formulas, SMILES).

---

## Supported models

| Preset | Hugging Face id | Notes |
|--------|-----------------|-------|
| `chemdfm-8b` (**default local**) | [`OpenDFM/ChemDFM-v1.5-8B`](https://huggingface.co/OpenDFM/ChemDFM-v1.5-8B) | Chemistry LLM; fits RTX 4060 8GB better |
| `chemdfm-14b` | [`OpenDFM/ChemDFM-v2.0-14B`](https://huggingface.co/OpenDFM/ChemDFM-v2.0-14B) | **Qwen2.5-14B** chemistry post-trained |
| `chemdfm-r-14b` | [`OpenDFM/ChemDFM-R-14B`](https://huggingface.co/OpenDFM/ChemDFM-R-14B) | Chemistry *reasoning* LLM |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | General open instruct |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | Small general Qwen 3.5 |

> There is currently **no official “Qwen3.5-Chemistry”** checkpoint. The closest open **post-trained chemistry** models in the Qwen lineage are **ChemDFM-v2 / ChemDFM-R** (built on **Qwen2.5**). You can still run **Qwen3.5** as a general backend via the same interface.

**Model licenses differ from this repo** (Apache-2.0). ChemDFM is typically AGPL — respect upstream licenses when redistributing weights.

---

## Install

```bash
# core (GraphML, prompts, dry-run, reports)
pip install -e .

# local GPU inference (Linux recommended for bitsandbytes 4-bit)
pip install -e ".[local]"

# OpenAI-compatible APIs (Ollama, vLLM, OpenRouter, …)
pip install -e ".[api]"

# dev
pip install -e ".[dev]"
```

### Hardware notes (RTX 4060 8GB)

| Setup | Recommendation |
|-------|----------------|
| Windows + 8GB VRAM | Prefer **Ollama / vLLM API** backend, or `chemdfm-8b` / `qwen3.5-4b` without 4-bit if bnb unavailable |
| Linux + 8GB VRAM | `chemdfm-8b` or `qwen2.5-7b` with `--4bit` |
| ≥16–24GB VRAM | `chemdfm-14b` / `chemdfm-r-14b` |

---

## Quick start

### 1) Dry-run (no model download — CI / demos)

```bash
ego-mol-llm predict path/to/network.graphml -b dry-run -o outputs/demo
```

### 2) Local ChemDFM (transformers)

```bash
ego-mol-llm predict path/to/network.graphml \
  -b transformers -m chemdfm-8b \
  --4bit -o outputs/chemdfm
```

### 3) Ollama / OpenAI-compatible server

```bash
# example: Ollama serving a Qwen model
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama

ego-mol-llm predict path/to/network.graphml \
  -b ollama -m qwen2.5:14b \
  -o outputs/ollama
```

### Python API

```python
from ego_mol_llm import predict_from_graphml
from ego_mol_llm.report import export_report

result = predict_from_graphml(
    "network.graphml",
    backend="dry-run",          # or "transformers" / "openai"
    model="chemdfm-8b",
    hide_seed_name=True,        # blind mode
)
print(result.to_dict())
export_report(result, "outputs/run")
```

---

## Method (short)

1. **Parse** GraphML (nodes: PEPMASS, name, SMILES, community; edges: cosine, abs Δm/z).
2. **Select seed** (default node `0` or CLI flags); **hide seed name/SMILES** for blind prediction.
3. **Build ego context**: top-N direct neighbors + optional 2-hop annotated nodes.
4. **Prompt** chemistry LLM with structured neighborhood evidence + strict JSON schema.
5. **Validate** SMILES (RDKit) and optional precursor mass / adduct consistency.
6. **Export** `prediction.json`, `prediction.md`, `ego_network.png`, `prompt.txt`, `model_raw.txt`.

See [`paper/METHODS.md`](paper/METHODS.md) for a manuscript-ready methods draft.

---

## CLI

```bash
ego-mol-llm list-models

ego-mol-llm predict NETWORK.graphml \
  --backend transformers \
  --model chemdfm-14b \
  --seed-id 0 \
  --max-neighbors 25 \
  --out outputs/run
```

| Flag | Meaning |
|------|---------|
| `--backend` | `dry-run` \| `transformers` \| `openai` / `ollama` |
| `--model` | Preset name or full model id |
| `--seed-id` | Center node id |
| `--show-seed-name` | Disable blinding (evaluation leak) |
| `--4bit / --no-4bit` | bitsandbytes 4-bit load |
| `--base-url` | OpenAI-compatible endpoint |

---

## Repository layout

```
ego-mol-llm/
  src/ego_mol_llm/
    graphml.py          # GraphML parser
    ego.py              # ego neighborhood builder
    prompts.py          # ChemDFM/Qwen chat templates
    validate.py         # JSON/SMILES/mass checks
    predict.py          # end-to-end API
    report.py           # JSON/MD/figure export
    cli.py
    backends/           # dry-run, transformers, OpenAI-compatible
  tests/
  examples/
  paper/METHODS.md
```

---

## Reproducibility & publication checklist

- [x] Blind seed mode (name/SMILES redacted)
- [x] Full prompt + raw model dump saved per run
- [x] Structured JSON schema for predictions
- [x] SMILES validation + mass sanity check
- [x] Dry-run backend for unit tests without GPU
- [ ] Benchmark set of GraphML ego networks with ground-truth structures
- [ ] Comparison table: ChemDFM-8B / 14B / R vs Qwen2.5 vs Qwen3.5
- [ ] Human baseline (analyst ego annotation)

---

## Citation

If you use this software, please cite this repository and the chemistry LLM you ran:

```bibtex
@software{ego_mol_llm_2026,
  title  = {ego-mol-llm: Blind structure prediction from molecular-network ego neighborhoods},
  year   = {2026},
  url    = {https://github.com/arome-science/ego-mol-llm}
}
```

ChemDFM:

```bibtex
@article{zhao2025developing,
  title   = {Developing ChemDFM as a large language foundation model for chemistry},
  author  = {Zhao, Zihan and others},
  journal = {Cell Reports Physical Science},
  volume  = {6},
  number  = {4},
  year    = {2025}
}
```

---

## Disclaimer

Model outputs can be **wrong**. Always validate with accurate mass, RT, standards, and domain expertise before any biological or product decision.

---

## License

Code: **Apache-2.0** (see [`LICENSE`](LICENSE)).  
Model weights: **upstream licenses apply**.
