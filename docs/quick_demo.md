# AI Labs — Quick Demo and Architecture Cheatsheet

_Own your AI: private, fast, predictable‑cost LLMs for SMEs—ChatGPT‑style on your hardware, with your data._

Audience: AI Engineering • MLOps • DevOps • SMB Owners

## Introduction

AI Labs is a local LLM inference server that:
- Exposes OpenAI-compatible endpoints (chat completions, embeddings, similarities) for easy integration.
- Ships with a polished CLI and interactive chat UI for rapid experimentation.
- Adds a simple finance RAG over `data/all_transactions.csv` for instant, exact answers to transaction queries.

Stack choices and why
- PyTorch + Hugging Face Transformers: widest model coverage, mature generation APIs, BF16/FP16 on NVIDIA, device_map="auto" for simple multi-GPU/CPU placement, built-in streamers.
- FastAPI + Uvicorn: async-by-default, great developer ergonomics, native SSE streaming, automatic OpenAPI schema.
- Sentence-Transformers: pragmatic embeddings with cosine similarity; fast to stand up without a vector DB.
- Rich + Typer: delightful DX for terminals (structured CLI, progress bars, live streaming panels) with minimal overhead.
- Pandas (RAG): lightweight analytics over CSV for a realistic “hybrid intelligence” demo without extra infra.
- uv + Torch nightly wheels: reproducible, fast installs; access to latest CUDA kernels while keeping the option to pin stable.

Design principles
- OpenAI compatibility first to reduce integration friction.
- Single-process model cache with optional preload for fast first-token latency.
- Clear config precedence: CLI > env vars > TOML defaults.
- Streaming via SSE for perceived performance and better UX.
- GPU-first with sensible CPU fallback; dtype auto-resolution to BF16/FP16 where possible.

## Hardware Requirements

- OS and Runtime: Linux recommended; Python 3.12+. Works on macOS (CPU) and Windows WSL2 as well.
- CPU-only: Supported for development and tests; expect slow generation. Use tiny models (e.g., `sshleifer/tiny-gpt2`).
- NVIDIA GPU (recommended):
  - VRAM: ~14–16 GB for 7B models at FP16/BF16; multi-GPU sharding supported via `device_map="auto"`.
  - BF16 support: Ampere or newer (RTX 30/40, A100/A10, H100). Otherwise FP16 is used.
  - Drivers/CUDA: Recent CUDA-capable driver. Project defaults target cu130 nightly wheels (see `pyproject.toml`); you can switch to stable wheels if preferred.
- Disk: 10–20 GB per 7B model; mount/cache Hugging Face downloads for faster cold starts (see Docker section).
- Network: Internet required on first run to pull models/tokenizers from Hugging Face Hub.
- Docker GPU: Install `nvidia-container-toolkit` and run with `--gpus all` to pass through GPUs.

Note on quantization: `labs.toml` includes 4-bit/8-bit settings, but quantized loading isn’t wired in the loader yet; add BitsAndBytes to reduce VRAM needs if required.

Why Transformers instead of Ollama (brief)
- Flexibility/control: load any HF repo; full access to tokenization, decoding, chat templates, and logits; mix generation + embeddings in the same runtime.
- Extensibility/performance: add LoRA/PEFT, custom RAG, safety filters; choose dtype (bf16/fp16/fp32), device_map="auto" and multi-GPU sharding.
- Ops/observability: pure Python FastAPI service with OpenAI-compatible APIs, easy metrics/logging/testing; no extra daemon or tool-specific model format. Ollama is great for quick local setups with prebuilt quantized models, but less customizable.

## SMB Use Cases and Benefits

Practical scenarios for small/medium businesses:
- Finance and operations: Instant spend analysis, largest expenses, category/month/merchant breakdowns (via built-in RAG); budget monitoring and anomaly spotting.
- Customer support and sales: Draft replies, classify intents/tickets, summarize conversations; internal FAQ assistant powered by your knowledge base.
- Document automation: Summarize and extract fields from invoices, receipts, contracts; convert docs to embeddings for semantic search.
- Knowledge discovery: Company wiki and policy search with RAG; answer “how do we…?” securely from internal docs.
- Engineering productivity: Code and shell suggestions for internal tooling; refactor/summarize PRs and logs.
- Compliance and governance: On-premise redaction and PII detection before data leaves your environment.

Business benefits:
- Data residency and privacy: Keep sensitive data on your hardware; no third-party API exposure.
- Cost control: Predictable infra costs vs. variable per-token API bills; reuse existing GPUs.
- Customization: Tailor prompts, RAG pipelines, and adapters to your domain; mix generation + embeddings in one service.
- Latency and reliability: Low-latency, offline-tolerant operation; fewer external dependencies.
- Integration friendly: OpenAI-compatible APIs and Python SDK patterns slot into existing apps and MLOps stacks.
- Vendor neutrality: Hugging Face ecosystem + containers avoid lock-in; easy to swap models or scale horizontally.

---

## TL;DR Demo

```bash
# Install deps (Python 3.12+, uv installed)
uv sync

# Interactive chat (with rich UI)
uv run labs-gen --interactive

# One-shot prompt (streaming)
uv run labs-gen --prompt "Summarize CUDA BF16 in one paragraph" --stream

# Transaction intelligence (RAG over data/all_transactions.csv)
uv run labs-gen --prompt "What was my largest expense?"

# Show config/GPU
uv run labs-gen --show-config
uv run labs-gen --show-gpu
```

API
```bash
# Start API (LAN by default)
LABS_HOST=0.0.0.0 LABS_PORT=8000 uv run labs-api

# Health and models
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models

# Chat completion (non-streaming)
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role":"user","content":"Explain BF16 briefly"}],
    "max_tokens": 64
  }'

# Streaming (SSE)
curl -N -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role":"user","content":"Tell me a GPU joke"}],
    "max_tokens": 64,
    "stream": true
  }'

# Embeddings
curl -s -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"hello world","model":"google/embeddinggemma-300m"}'
```

---

## Configuration (Ops quick view)

Precedence: CLI > env vars > `labs.toml` > defaults.

Common env vars
- LABS_MODEL (e.g., Qwen/Qwen2.5-7B-Instruct)
- LABS_MAX_NEW_TOKENS, LABS_TEMPERATURE, LABS_TOP_P, LABS_TOP_K, LABS_DO_SAMPLE, LABS_REPETITION_PENALTY
- LABS_DEVICE_MAP=auto, LABS_TORCH_DTYPE=bf16|fp16|fp32, LABS_TRUST_REMOTE_CODE=false
- LABS_HOST, LABS_PORT
- LABS_EMBEDDING_MODEL, LABS_EMBEDDING_DEVICE=auto
- LABS_PRELOAD_ON_START=true (API preloads model on startup)

Switching models
```bash
export LABS_MODEL="Qwen/Qwen2.5-7B-Instruct"  # or set in .env
uv run labs-gen --prompt "Hello"
```

---

## Deployment (DevOps/MLOps)

Docker
```bash
docker build -t ai-labs:latest .
# Persist HF cache to speed up warm starts
docker run --rm -p 8000:8000 \
  -e LABS_HOST=0.0.0.0 -e LABS_PORT=8000 \
  -e LABS_MODEL=Qwen/Qwen2.5-7B-Instruct \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  ai-labs:latest
```

Production uvicorn (bypass reload in script)
```bash
uv run uvicorn labs.api:app --host 0.0.0.0 --port 8000 --workers 1
```

Health/readiness
- `/health`, `/v1/health` → {"status":"ok"}
- `/ready` → {"loaded": true|false} (model cache status)

Scaling guidance
- One process per GPU recommended; `device_map="auto"` supports sharding.
- Horizontal scale behind a load balancer; models download from HF Hub (mount cache).
- Model cache is in-process only; no cross-process sharing.

Security notes
- CORS is wide-open by default for compatibility; restrict in production.
- `trust_remote_code=false` by default; enable only for trusted repos.

---

## Key Components That Make It Work

- `labs/generate.py`
  - `GenerationConfig`: decoding, device, dtype, chat-template flags.
  - `HFGenerator`: loads model/tokenizer, resolves dtype (bf16 on supported GPUs, fp16 fallback), builds inputs (raw vs chat template), supports `generate()` and `stream_generate()` via `TextIteratorStreamer`.
  - Transaction RAG short-circuit: detects finance questions and answers via RAG before LLM.

- `labs/api.py`
  - FastAPI app exposing OpenAI-compatible endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/similarities`).
  - SSE streaming chunks with `delta.content` and final `[DONE]` sentinel.
  - In-process model cache `_GenCache` keyed by `(model_name, trust_remote_code)` and optional preload on startup.

- `labs/config.py`
  - Loads `.env`, merges `labs.toml`, then applies env vars; returns `GenerationConfig`.
  - Proper parsing of torch dtypes and booleans.

- `labs/model_loader.py`
  - Centralized `AutoTokenizer`/`AutoModelForCausalLM` load; sets pad token if missing; passes `device_map`, `dtype`, `trust_remote_code`.

- `labs/embeddings.py`
  - Sentence-Transformers wrapper with env-derived model/device; used by `/v1/embeddings` and `/v1/similarities`.

- `labs/rag_qa.py`
  - Pandas-based answers over `data/all_transactions.csv` (totals, largest expense, category/month/merchant, summary).

- `labs/ui.py` + `labs/interactive.py`
  - Rich UI (banners, progress, streaming panels), interactive mode with history at `~/.labs/conversations` and session stats.

- `tests/test_smoke.py`
  - CPU-friendly smoke tests using `sshleifer/tiny-gpt2` for both non-streaming and streaming paths.

---

## Demo Script (5–10 minutes)

1) Show config and GPU
```bash
uv run labs-gen --show-config
uv run labs-gen --show-gpu
```

2) Interactive chat with thinking separation and save
```bash
uv run labs-gen --interactive
# Ask two general questions; then /save and /list
```

3) RAG accuracy demo
```bash
uv run labs-gen --prompt "What was my largest expense?"
uv run labs-gen --prompt "Give me a financial summary"
```

4) API streaming
```bash
uv run labs-api &
curl -N -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Write a haiku about GPUs"}],"stream":true,"max_tokens":64}'
```

5) Embeddings + similarity
```bash
curl -s -X POST http://localhost:8000/v1/similarities \
  -H "Content-Type: application/json" \
  -d '{"sentences":["cat","feline","quantum computer"]}'
```

---

## Notes and Next Steps
- Quantization flags exist in `labs.toml`, but quantized loading isn’t wired in `model_loader.py` yet; integrate BitsAndBytes if needed.
- Add structured logging/metrics for observability (Prometheus middleware, request timing, token counts).
- Consider request limits and auth for multi-tenant deployments.
