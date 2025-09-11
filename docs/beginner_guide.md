# Labs Codebase and Stack — Beginner AI Engineer Guide

This guide explains how this repository works, the tech stack it uses, and how data flows through the system, with pointers to the most important code constructs.

Contents
- What you can do with this repo
- Tech stack overview
- Project layout and key modules
- End-to-end flow (CLI and REST API)
- Configuration (labs.toml, .env, CLI, precedence)
- GPU, dtype, and quantization basics
- Streaming generation
- Running, testing, and troubleshooting
- Security and operational considerations
- Next steps and ideas

What you can do with this repo
- Generate text using instruction-tuned LLMs from Hugging Face, leveraging your NVIDIA GPU.
- Use a CLI to run single-shot or streaming generation.
- Host a REST API (FastAPI) offering both non-streaming and Server-Sent Events (SSE) streaming endpoints.
- Configure the model and parameters via a TOML config and environment variables (.env), with clear precedence rules.
- Optionally enable 4-bit/8-bit quantization to reduce VRAM usage.

Tech stack overview
- Python 3.12+ with uv for dependency management and virtualenv.
- PyTorch (nightly wheels by default) for tensor operations and GPU acceleration.
- Transformers (Hugging Face) for tokenizer/model loading and generation APIs.
- FastAPI + Uvicorn for a minimal HTTP server (REST API).
- Optional bitsandbytes for 4-bit/8-bit quantized inference (VRAM savings).
- python-dotenv for .env support (developer-friendly configuration).
- pytest for smoke tests.

Project layout and key modules
- Core generator and configuration:
  - [labs/generate.py](../labs/generate.py)
    - Generator config: [GenerationConfig](../labs/generate.py:10)
    - Generator class: [HFGenerator](../labs/generate.py:47)
      - Model setup: [HFGenerator.__init__()](../labs/generate.py:53)
      - Dtype selection: [HFGenerator._resolve_dtype()](../labs/generate.py:88)
      - Quantization config (optional): [HFGenerator._build_quantization_config()](../labs/generate.py:101)
      - Input building: [HFGenerator._build_inputs()](../labs/generate.py:137)
      - Device alignment: [HFGenerator._maybe_move_inputs_to_model_device()](../labs/generate.py:171)
      - Non-streaming generation: [HFGenerator.generate()](../labs/generate.py:196)
      - Streaming generation: [HFGenerator.stream_generate()](../labs/generate.py:235)
  - [labs/config.py](../labs/config.py)
    - Config loader with TOML + .env: [load_config()](../labs/config.py:193)
    - Effective config dump: [dump_effective_config()](../labs/config.py:238)
    - Internal helpers for merging and env parsing are in the same file.
- CLI:
  - [labs/cli.py](../labs/cli.py)
    - Argument builder: [build_arg_parser()](../labs/cli.py:25)
    - Entry point: [main()](../labs/cli.py:95)
- REST API:
  - [labs/api.py](../labs/api.py)
    - FastAPI app: [app](../labs/api.py:50)
    - Request model: [GenerateRequest](../labs/api.py:15)
    - Response model: [GenerateResponse](../labs/api.py:46)
    - Model cache and builder: [_GenCache](../labs/api.py:56), [_build_generator()](../labs/api.py:74)
    - Endpoints:
      - Health: [/health](../labs/api.py:96)
      - Non-streaming: [/generate](../labs/api.py:101)
      - Streaming (SSE): [/generate/stream](../labs/api.py:133)
- Project metadata and scripts:
  - [pyproject.toml](../pyproject.toml)
    - Console scripts: labs-gen (CLI), labs-api (API)
    - Optional extras for quantization and tests.
- Defaults and docs:
  - [labs.toml](../labs.toml) — default generation and runtime settings
  - [.env.example](../.env.example) — template for environment variables

End-to-end flow

CLI (non-streaming)
1) Parse args in [build_arg_parser()](../labs/cli.py:25) and read config via [load_config()](../labs/config.py:193).
2) Initialize the generator in [main()](../labs/cli.py:95) with a [GenerationConfig](../labs/generate.py:10).
3) Prepare inputs via [HFGenerator._build_inputs()](../labs/generate.py:137), which uses either:
   - tokenizer(...) for raw prompts
   - tokenizer.apply_chat_template(...) for chat messages
4) Align inputs to the model’s device via [HFGenerator._maybe_move_inputs_to_model_device()](../labs/generate.py:171).
5) Generate in [HFGenerator.generate()](../labs/generate.py:196) and print to stdout.

CLI (streaming)
- Same as above, but uses [HFGenerator.stream_generate()](../labs/generate.py:235), which injects a TextIteratorStreamer into model.generate and yields incremental tokens.

API (non-streaming)
1) A request hits [/generate](../labs/api.py:101) (JSON body matching [GenerateRequest](../labs/api.py:15)).
2) The server calls [_build_generator()](../labs/api.py:74) to create or fetch a cached [HFGenerator](../labs/generate.py:47), drawing defaults from [load_config()](../labs/config.py:193).
3) It builds inputs, aligns devices, and calls [HFGenerator.generate()](../labs/generate.py:196).
4) Returns a JSON [GenerateResponse](../labs/api.py:46).

API (streaming)
1) A request hits [/generate/stream](../labs/api.py:133).
2) It builds inputs, aligns devices, and starts a background generation thread with a TextIteratorStreamer.
3) The endpoint returns a StreamingResponse yielding “data: ...” chunks until completion.

Configuration (labs.toml, .env, CLI)

Files and environment
- labs.toml — base defaults under [generation] and [quantization].
- .env — developer-friendly overrides; see [.env.example](../.env.example).
- Environment variables — the loader respects env vars (including those populated from .env).
- CLI flags — highest precedence.

Precedence rules (highest to lowest)
1) CLI arguments (e.g., --model, --temperature)
2) Environment variables (.env included), e.g., LABS_MODEL, LABS_DEVICE_MAP
3) labs.toml keys (e.g., generation.model_name)
4) Built-in defaults in [GenerationConfig](../labs/generate.py:10)

Important keys
- Model selection: LABS_MODEL or generation.model_name
- Generation params: LABS_MAX_NEW_TOKENS, LABS_TEMPERATURE, LABS_TOP_P, LABS_TOP_K, LABS_DO_SAMPLE, LABS_REPETITION_PENALTY
- Runtime: LABS_DEVICE_MAP (auto|cuda|cpu), LABS_TORCH_DTYPE (bf16|fp16|fp32), LABS_TRUST_REMOTE_CODE
- Chat toggles: LABS_USE_CHAT_TEMPLATE, LABS_ADD_GENERATION_PROMPT
- Quantization: LABS_LOAD_IN_4BIT, LABS_LOAD_IN_8BIT (+ bnb settings)

GPU, dtype, and quantization basics

Device placement (device_map)
- auto: Let Accelerate pick devices (GPU/CPU sharding if needed). Best default on GPUs.
- cuda: Force a single CUDA device (if available).
- cpu: Force CPU (slow, but reliable for minimal models).

Dtypes
- bf16: Preferred on modern NVIDIA (Ampere+) for speed and stability.
- fp16: Widely supported on NVIDIA GPUs if bf16 unsupported.
- fp32: CPU-safe default when no CUDA device is available.

Quantization (optional, reduces VRAM)
- 8-bit (load_in_8bit) or 4-bit (load_in_4bit via bitsandbytes).
- 4-bit compute dtype: bf16 (preferred) or fp16.
- Enable via labs.toml, .env, or CLI flags; only one of 4-bit/8-bit may be enabled at once.
- Internals handled in [HFGenerator._build_quantization_config()](../labs/generate.py:101).

Streaming generation

CLI streaming
- Flag: --stream
- Under the hood: [HFGenerator.stream_generate()](../labs/generate.py:235) uses [TextIteratorStreamer](../labs/generate.py:267) to yield tokens progressively.

API streaming (SSE)
- Endpoint: [/generate/stream](../labs/api.py:133)
- Content-Type: text/event-stream
- Each chunk is sent as “data: <text>” lines, followed by an “event: done” marker.
- Works well with curl -N or any SSE-capable client.

Running, testing, and troubleshooting

Install dependencies (uv)
- Base:
  - uv sync
- With extras:
  - Quantization: uv sync --extra quantization
  - Tests: uv sync --extra test

CLI examples
- Non-chat prompt:
  - uv run labs-gen --prompt "Hello! Who are you?" --max-new-tokens 64
- Chat (inline JSON):
  - uv run labs-gen --messages-json '[{"role":"user","content":"Who are you?"}]' --max-new-tokens 64
- Streaming:
  - uv run labs-gen --prompt "Tell me a short GPU joke." --stream
- Override model from shell:
  - LABS_MODEL=mistralai/Mistral-7B-Instruct-v0.3 uv run labs-gen --prompt "Hi!"

API examples
- Start:
  - uv run labs-api
- Health:
  - curl -s http://localhost:8000/health
- Non-streaming:
  - curl -s -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt":"Explain BF16","max_new_tokens":32}'
- Streaming:
  - curl -N -s -X POST http://localhost:8000/generate/stream -H "Content-Type: application/json" -d '{"prompt":"Stream one sentence about CUDA.","max_new_tokens":32}'

Tests
- Smoke tests use a tiny model to keep runtime small:
  - uv run pytest -q
- See tests under tests/ (e.g., non-streaming and streaming paths).

Troubleshooting tips
- “input_ids on cpu, model on cuda” warning:
  - Addressed by automatic input alignment in [HFGenerator._maybe_move_inputs_to_model_device()](../labs/generate.py:171). If you still see it under heavy sharding (device_map=auto), it’s generally safe to ignore.
  - To force single-GPU, try LABS_DEVICE_MAP=cuda.
- OOM (out of memory):
  - Lower max_new_tokens, switch to a smaller model, or enable 4-bit quantization.
- CUDA/driver issues:
  - Ensure driver and CUDA runtime for your PyTorch wheel match. Consider switching from nightly to stable wheels if you need predictability.
- trust_remote_code errors:
  - Some model repos require it; enable via LABS_TRUST_REMOTE_CODE=true or CLI. Only use with trusted sources.
- Hardlink warning during uv install:
  - export UV_LINK_MODE=copy (or use --link-mode=copy).

Security and operational considerations
- trust_remote_code executes repository code — enable only for trusted model repositories.
- Restrict external network access for production if required (model caching via HF cache).
- Validate/limit prompt inputs in production to avoid misuse and control cost/performance.
- Consider request rate limiting and auth if exposing API externally.
- Monitor GPU memory and latency; add basic logging/metrics around [HFGenerator.generate()](../labs/generate.py:196) calls.

Next steps and ideas
- Caching: Add prompt/response caching for common requests.
- Observability: Integrate metrics (latency, tokens/s) and logging.
- Batch inference: Add a batched generation path to improve throughput.
- Model registry: Provide multiple selectable models with a router.
- Safety: Add content filtering or harmlessness classifiers in pre/post processing.
- RAG: Add retrieval augmentation to ground responses in your documents.

Appendix: How generation works (at a glance)
- Tokenization: Input text or chat messages are converted into token IDs.
- Model.generate: Autoregressive decoding predicts the next tokens based on the prompt.
- Sampling controls:
  - temperature: randomness scaling
  - top_p: nucleus sampling
  - top_k: restricts to top-k probable tokens
  - repetition_penalty: reduces repeats
- Streaming: Tokens are yielded as soon as they’re produced via a streamer, improving perceived latency.

With this guide and the code pointers above, you can explore, modify, and extend the system confidently — starting from configuration and model selection, through GPU-aware inference, to exposing stable endpoints for real applications.