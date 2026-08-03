# Run DeepSeek-V4-Flash free-form on MSG HNSW **test64**

## Target pack
```
C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_test64_nist_neigh_handout
```
- 64 jobs: `jobs\MSGFULL*.json` (system + user prompts, neighbor NIST already in)
- Output: `predictions_deepseek_v4_flash\`

## Hardware reality (this machine)

| Resource | Have (approx) | Need for usable Flash |
|----------|---------------|------------------------|
| System RAM | 128 GB total | **≥110 GB free** for 3-bit; ≥92 GB for 1-bit |
| GPU | RTX 4060 **8 GB** | Only a few layers; model is mostly **CPU/RAM** |
| Disk free (C:) | **~80 GB** | **≥110 GB** for 3-bit GGUF; ≥95 GB for 2-bit |

**Blockers today**
1. **Disk** — not enough free space for recommended `UD-IQ3_XXS` (~103 GB). Even 1-bit (~82 GB) is tight.
2. **RAM free** — often ~70–80 GB free with desktop apps; need to close browsers/Office/Claude for a 3-bit load.

**Verdict:** Possible after freeing **~40+ GB disk** and closing heavy apps; then use Unsloth GGUF + llama.cpp (or Unsloth Studio). Not an Ollama one-liner yet (not in `ollama list` as a small model).

---

## Recommended quant for 128 GB RAM

| Quant | Size | Memory (RAM+VRAM) | Quality for free-form |
|-------|------|-------------------|------------------------|
| **UD-IQ3_XXS** | **~103 GB** | **~110 GB** | Best balance (Unsloth default for 128 GB) |
| UD-IQ2_M | ~91 GB | ~102 GB | OK if disk/RAM tight |
| UD-IQ1_S | ~82 GB | ~92 GB | Last resort / smoke test |

Do **not** start with Q4 (~155 GB) — will not fit.

---

## Path A — Unsloth Studio (easiest UI)

```powershell
# Install (once)
irm https://unsloth.ai/install.ps1 | iex

# Launch
unsloth studio -H 0.0.0.0 -p 8888
```

1. Open http://127.0.0.1:8888  
2. Download **DeepSeek-V4-Flash-0731** → quant **UD-IQ3_XXS** (after disk free)  
3. Settings: temp **1.0**, top_p **1.0**, thinking **High** or **off** for speed  
4. For batch free-form, prefer Path B (OpenAI-compatible server + our runner)

---

## Path B — llama.cpp server + our pack runner (batch)

### 1) Free space
Need **≥110 GB free** on the drive that will hold the GGUF (suggest `D:\` or `C:\Users\AlexeyMelnik\Models\`).

### 2) Download quant
```powershell
pip install -U "huggingface_hub[cli]"
# Example: 3-bit to Models (adjust drive if needed)
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF `
  --local-dir C:\Users\AlexeyMelnik\Models\DeepSeek-V4-Flash-0731-GGUF `
  --include "*UD-IQ3_XXS*"
```

If disk is still tight:
```powershell
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF `
  --local-dir C:\Users\AlexeyMelnik\Models\DeepSeek-V4-Flash-0731-GGUF `
  --include "*UD-IQ2_M*"
```

### 3) Build/run llama-server (latest llama.cpp with deepseek4 arch)
```powershell
# After building llama.cpp with CUDA (optional, limited gain on 8GB):
#   cmake -B build -DGGML_CUDA=ON
#   cmake --build build --config Release --target llama-server

# Start OpenAI-compatible server (example — adjust model path + threads)
.\llama-server.exe `
  -m C:\Users\AlexeyMelnik\Models\DeepSeek-V4-Flash-0731-GGUF\...\*-00001-of-*.gguf `
  --port 8080 `
  --ctx-size 32768 `
  -ngl 8 `
  --threads 16 `
  --temp 1.0 `
  --top-p 1.0
```

`-ngl 8` = few GPU layers only (4060 8 GB). Increase carefully until OOM.

### 4) Run test64 free-form (1 sample smoke test first)
```powershell
cd C:\Users\AlexeyMelnik\Desktop\Codes\ego-mol-llm

# Smoke: 1 spectrum
python scripts\run_jobs_prompt_ollama.py `
  --pack C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_test64_nist_neigh_handout `
  --jobs-subdir jobs `
  --out-subdir predictions_deepseek_v4_flash `
  --model deepseek-v4-flash `
  --backend openai `
  --base-url http://127.0.0.1:8080/v1 `
  --api-key none `
  --temperature 1.0 `
  --max-new-tokens 4096 `
  --limit 1

# Full 64 after smoke OK
python scripts\run_jobs_prompt_ollama.py `
  --pack C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_test64_nist_neigh_handout `
  --jobs-subdir jobs `
  --out-subdir predictions_deepseek_v4_flash `
  --model deepseek-v4-flash `
  --backend openai `
  --base-url http://127.0.0.1:8080/v1 `
  --api-key none `
  --temperature 1.0 `
  --max-new-tokens 4096
```

### 5) Score like Grok
```powershell
# After full run, point scorer at predictions_deepseek_v4_flash
# (or adapt _score_msg_test64_grok_handout.py paths)
```

---

## Path C — DeepSeek **API** (recommended if local is too slow)

Same runner, point at DeepSeek OpenAI-compatible endpoint + API key. Cost for 64 long prompts is far lower than multi-day local CPU MoE runs, and quality is full Flash (not 1–3 bit).

---

## Expected runtime (local 3-bit, rough)

| Step | Time |
|------|------|
| Download 103 GB | hours (depends on HF) |
| Load into RAM | several minutes |
| 1 long prompt (~25–35k chars user) | often **many minutes** (CPU MoE) |
| Full 64 | **hours to days** if not interrupted |

For a fair “can it run?” test: **limit 1**, then **limit 3**, only then full 64.

---

## Checklist before download

- [ ] Free **≥110 GB** disk (or use another drive)  
- [ ] Close Chrome / Word / Claude desktop / other LLMs  
- [ ] Confirm `ollama ps` empty  
- [ ] Smoke test 1 job  
- [ ] Only then batch 64  
