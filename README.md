# AI Labs — OpenAI-Compatible Local LLM Inference Server

This repository provides:
- A core generator built on Hugging Face Transformers with sensible NVIDIA GPU defaults
- A CLI tool for single-shot and streaming generation
- A FastAPI server with OpenAI-compatible endpoints (/v1/chat/completions, /v1/models)
- A TOML-based config with environment and CLI overrides
- Optional 4-bit/8-bit quantization support via bitsandbytes

Your hardware (as stated): NVIDIA GPU with ≥16GB VRAM and CUDA available. Defaults are set to prefer BF16 (or FP16) with device_map="auto" and no quantization.

Repository layout
- [labs/generate.py](labs/generate.py) — core generator (HFGenerator)
- [labs/cli.py](labs/cli.py) — CLI entrypoint (labs-gen)
- [labs/api.py](labs/api.py) — FastAPI app and streaming endpoints (labs-api)
- [labs/config.py](labs/config.py) — config loader (labs.toml + env + CLI override)
- [labs.toml](labs.toml) — default configuration
- [pyproject.toml](pyproject.toml) — dependencies, scripts, and optional extras

Requirements
- Python 3.12+
- NVIDIA GPU with a recent CUDA-capable driver (recommended)
- Network access to download models from Hugging Face Hub (first run)

Note on PyTorch nightly
- This project is configured to install PyTorch “nightly” CUDA wheels via uv indexes in [pyproject.toml](pyproject.toml). If you prefer stable wheels, you can change or remove the [tool.uv.sources] section and the [[tool.uv.index]] block pointing at cu130.

Quickstart (uv)
1) Install dependencies:
- Base (no quantization/test extras):
  uv sync
- With quantization and tests:
  uv sync --extra quantization --extra test

2) Run the CLI:
- Non-chat prompt:
  uv run labs-gen --prompt "Hello! Who are you?" --max-new-tokens 64
- Chat messages (inline JSON):
  uv run labs-gen --messages-json '[{"role":"user","content":"Who are you?"}]' --max-new-tokens 64
- Streaming:
  uv run labs-gen --prompt "Tell me a short joke about GPUs." --stream

3) Run the API (FastAPI + uvicorn):
- Start server (accessible on LAN):
  uv run labs-api
- Start server on specific host/port:
  LABS_HOST=0.0.0.0 LABS_PORT=8000 uv run labs-api
- Health check:
  curl -s http://localhost:8000/health

**OpenAI-Compatible API:**
- List models:
  curl -s http://localhost:8000/v1/models
- Chat completion:
  curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Explain BF16 in one sentence"}],"max_tokens":64}'
- Streaming chat completion:
  curl -N -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Tell me a joke"}],"max_tokens":64,"stream":true}'

Configuration
- Defaults live in [labs.toml](labs.toml). You can point to a different file via:
  - CLI: --config /path/to/labs.toml
  - Env var: LABS_CONFIG=/path/to/labs.toml
- .env support: a .env file (if present) is auto-loaded at startup. Copy [.env.example](.env.example) to .env and edit.
  - Example: cp .env.example .env
- Precedence: CLI args override environment vars (including values from .env) override labs.toml.

Key settings in labs.toml:
[generation]
- model_name = "Qwen/Qwen2.5-7B-Instruct"
- device_map = "auto"        # auto-placement across GPU/CPU
- torch_dtype = "bf16"       # prefer bf16 on modern NVIDIA GPUs; or "fp16"/"fp32"
- max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty
- use_chat_template, add_generation_prompt

[quantization]
- load_in_4bit/load_in_8bit are disabled by default for ≥16GB GPUs
- If RAM is tight, enable one (not both)
- 4-bit options: bnb_4bit_quant_type, bnb_4bit_use_double_quant, bnb_4bit_compute_dtype

Environment variable overrides (examples)
- LABS_MODEL=meta-llama/Llama-3-8B-Instruct
- LABS_MAX_NEW_TOKENS=256
- LABS_TEMPERATURE=0.2
- LABS_TOP_P=0.95
- LABS_DEVICE_MAP=auto
- LABS_TORCH_DTYPE=bf16
- LABS_TRUST_REMOTE_CODE=true
- LABS_LOAD_IN_4BIT=true
- LABS_BNB_4BIT_COMPUTE_DTYPE=bf16

**Server Configuration:**
- LABS_HOST=0.0.0.0 (default: binds to all interfaces for LAN access)
- LABS_PORT=8000 (default: server port)

CLI usage examples
- Deterministic decoding (no sampling):
  uv run labs-gen --prompt "Summarize CUDA BF16" --no-sample --max-new-tokens 80
- Use a different model and set temperature:
  uv run labs-gen --prompt "What is device_map=auto?" --model "mistralai/Mistral-7B-Instruct-v0.3" --temperature 0.3
- Chat mode:
  uv run labs-gen --messages-json '[{"role":"user","content":"Write a greeting."}]'
- From file:
  uv run labs-gen --messages-json "@/absolute/or/relative/messages.json"

OpenAI API Compatibility
The API now provides OpenAI-compatible endpoints that can be used as a drop-in replacement with most OpenAI clients:

**Supported Endpoints:**
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Create chat completions (streaming and non-streaming)

**Model Name Mapping:**
- `gpt-3.5-turbo` → `Qwen/Qwen2.5-7B-Instruct`
- `gpt-4` → `Qwen/Qwen2.5-7B-Instruct`
- `qwen2.5-7b-instruct` → `Qwen/Qwen2.5-7B-Instruct`
- Or use the full model name directly: `Qwen/Qwen2.5-7B-Instruct`

**OpenAI Client Usage Example:**
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required for local server
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Maps to your local Qwen model
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

**Supported Parameters:**
- `model` - Model name (with automatic mapping)
- `messages` - Chat messages array
- `max_tokens` - Maximum tokens to generate
- `temperature` - Sampling temperature (0.0-2.0)
- `top_p` - Nucleus sampling probability
- `top_k` - Top-k sampling (non-standard OpenAI param)
- `stream` - Enable streaming responses
- `stop` - Stop sequences
- `frequency_penalty` - Mapped to repetition_penalty
- `presence_penalty` - Frequency penalty variant

API schema and docs
- Swagger UI and OpenAPI are available at:
  http://localhost:8000/docs
  http://localhost:8000/openapi.json

Streaming details
- CLI: pass --stream to progressively print tokens to stdout.
- API: Set `"stream": true` in `/v1/chat/completions` requests for Server-Sent Events (SSE) streaming with proper OpenAI format

Quantization (optional)
- Install extras:
  uv sync --extra quantization
- Enable in labs.toml or env:
  [quantization]
  load_in_4bit = true
- Notes:
  - Only enable one of load_in_4bit or load_in_8bit
  - bitsandbytes requires a compatible CUDA environment; errors will include guidance

Troubleshooting
- CUDA/driver errors:
  - Ensure your NVIDIA driver matches your CUDA runtime and PyTorch wheel.
  - If using nightly, consider switching to stable wheels if you hit regressions.
- Out-of-memory on GPU:
  - Reduce max_new_tokens
  - Enable 4-bit quantization
  - Try a smaller model
- Slow performance on CPU:
  - Use quantized or very small models
  - Consider enabling device_map=auto on a CUDA machine
- trust_remote_code:
  - Only enable for trusted model repos; it executes repository code

**LAN Access Issues:**
- Server not reachable from other devices:
  - Ensure server is bound to 0.0.0.0 (default): `LABS_HOST=0.0.0.0 uv run labs-api`
  - Check firewall settings: `sudo ufw allow 8000` (Ubuntu/Debian) or equivalent
  - Find your server IP: `ip addr show` or `hostname -I`
  - Test from another device: `curl http://<server-ip>:8000/health`
- Port already in use:
  - Use different port: `LABS_PORT=8001 uv run labs-api`
  - Kill existing process: `pkill -f labs-api` or `lsof -ti:8000 | xargs kill`
- Network connectivity:
  - Ensure devices are on same network/subnet
  - Check router/network configuration for device isolation
  - Try disabling VPN if active

Testing (smoke test)
- Install test extras:
  uv sync --extra test
- Run tests:
  uv run pytest -q
- The smoke test will use a tiny model to keep runtime and memory minimal.

Security note
- If you enable trust_remote_code, you are allowing execution of arbitrary code from the model repository. Only enable it for sources you trust.

License
- Add your project’s license here.
