# AI Labs — OpenAI-Compatible Local LLM Inference Server

A high-performance, production-ready local LLM inference server that provides OpenAI-compatible APIs for seamless integration with existing applications. Built on HuggingFace Transformers with optimized defaults for NVIDIA GPUs.

## 🚀 Key Features

- **OpenAI API Compatibility**: Drop-in replacement for OpenAI's API with `/v1/chat/completions` and `/v1/models` endpoints
- **High-Performance Inference**: Optimized for NVIDIA GPUs with automatic BF16/FP16 precision and device mapping
- **Flexible Deployment**: CLI tool for development and FastAPI server for production workloads
- **Advanced Configuration**: TOML-based configuration with environment variable overrides
- **Memory Optimization**: Optional 4-bit/8-bit quantization support via BitsAndBytes
- **Streaming Support**: Real-time token streaming with Server-Sent Events (SSE)

## 📋 System Requirements

- **Python**: 3.12 or higher
- **GPU**: NVIDIA GPU with ≥16GB VRAM (recommended)
- **CUDA**: Recent CUDA-capable driver
- **Network**: Internet access for initial model downloads from HuggingFace Hub

> **Note**: Optimized for NVIDIA GPUs with BF16/FP16 support and `device_map="auto"` for automatic model sharding.

## 📁 Project Structure

```
labs/
├── generate.py     # Core LLM generator with HuggingFace Transformers
├── cli.py          # Command-line interface (labs-gen)
├── api.py          # FastAPI server with OpenAI-compatible endpoints
├── config.py       # Configuration management (TOML + env + CLI)
└── labs.toml       # Default configuration file
```

## ⚡ Quick Start

### Installation

```bash
# Basic installation
uv sync

# With quantization and testing support
uv sync --extra quantization --extra test
```

### CLI Usage

```bash
# Simple text generation
uv run labs-gen --prompt "Hello! Who are you?" --max-new-tokens 64

# Chat-style interaction
uv run labs-gen --messages-json '[{"role":"user","content":"Who are you?"}]' --max-new-tokens 64

# Streaming generation
uv run labs-gen --prompt "Tell me a short joke about GPUs." --stream
```

### API Server

```bash
# Start server (accessible on LAN)
uv run labs-api

# Custom host/port configuration
LABS_HOST=0.0.0.0 LABS_PORT=8000 uv run labs-api

# Health check
curl -s http://localhost:8000/health
```

### API Examples

```bash
# List available models
curl -s http://localhost:8000/v1/models

# Chat completion (use exact model name from .env)
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.1",
    "messages": [{"role": "user", "content": "Explain BF16 in one sentence"}],
    "max_tokens": 64
  }'

# Streaming chat completion
curl -N -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.1",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "max_tokens": 64,
    "stream": true
  }'
```

> **PyTorch Note**: This project uses PyTorch nightly CUDA wheels for optimal performance. To use stable wheels instead, modify the `[tool.uv.sources]` section in `pyproject.toml`.

## ⚙️ Configuration

### Configuration Hierarchy

Configuration follows this precedence order (highest to lowest):
1. **CLI arguments** - Override all other settings
2. **Environment variables** - Override config file settings  
3. **TOML config file** - Default configuration source

### Configuration Files

```bash
# Use custom config file
uv run labs-gen --config /path/to/custom.toml

# Or set via environment variable
export LABS_CONFIG=/path/to/custom.toml

# Environment file support
cp .env.example .env  # Edit as needed
```

### Core Settings (`labs.toml`)

```toml
[generation]
model_name = "openai/gpt-oss-20b"
device_map = "auto"                    # Auto GPU/CPU placement
torch_dtype = "bf16"                   # BF16/FP16/FP32 precision
max_new_tokens = 128
temperature = 0.7
top_p = 0.9
do_sample = true
use_chat_template = true
add_generation_prompt = true

[quantization]
load_in_4bit = false                   # Enable for memory optimization
load_in_8bit = false                   # Alternative to 4-bit
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = true
```

### Environment Variables

```bash
# Model and generation settings
export LABS_MODEL="openai/gpt-oss-20b"
export LABS_MAX_NEW_TOKENS=256
export LABS_TEMPERATURE=0.2
export LABS_TOP_P=0.95

# Hardware optimization
export LABS_DEVICE_MAP=auto
export LABS_TORCH_DTYPE=bf16
export LABS_TRUST_REMOTE_CODE=false

# Memory optimization
export LABS_LOAD_IN_4BIT=true
export LABS_BNB_4BIT_COMPUTE_DTYPE=bf16

# Server configuration
export LABS_HOST=0.0.0.0              # LAN access
export LABS_PORT=8000                 # Server port
```

### Model Switching

The system supports flexible model switching through environment variables:

#### Environment Variables (Recommended)

The simplest way to switch models is by editing the `.env` file:

```bash
# Edit .env file
LABS_MODEL=mistralai/Mistral-7B-v0.1

# Or set temporarily for a single session
export LABS_MODEL="Qwen/Qwen2.5-7B-Instruct"
uv run labs-gen --prompt "Hello!"
```

**Popular Model Examples:**
```bash
# Mistral 7B (good balance of performance and memory usage)
LABS_MODEL=mistralai/Mistral-7B-v0.1

# Qwen 2.5 7B Instruct (excellent instruction following)
LABS_MODEL=Qwen/Qwen2.5-7B-Instruct

# Llama 2 7B Chat (Meta's chat model)
LABS_MODEL=meta-llama/Llama-2-7b-chat-hf

# CodeLlama for code generation
LABS_MODEL=codellama/CodeLlama-7b-Instruct-hf
```

#### Configuration File Fallback

If no environment variable is set, the system will use the default model specified in `labs.toml`:

```toml
[generation]
# Default model (typically overridden by .env file)
model_name = "Qwen/Qwen2.5-7B-Instruct"
```

#### Memory Considerations

- **Qwen/Qwen2.5-7B-Instruct**: Works well with 16GB+ VRAM, no quantization needed
- **openai/gpt-oss-20b**: Requires 4-bit quantization for 16GB VRAM, may need more memory
- **Custom Models**: Adjust quantization settings based on model size and available VRAM

> **Note**: Model profiles automatically apply appropriate quantization settings for each model. Large models (>10B parameters) typically require quantization on consumer GPUs.

### Advanced CLI Examples

```bash
# Deterministic generation (no sampling)
uv run labs-gen --prompt "Summarize CUDA BF16" --no-sample --max-new-tokens 80

# Custom model with specific parameters
uv run labs-gen \
  --prompt "What is device_map=auto?" \
  --model "mistralai/Mistral-7B-Instruct-v0.3" \
  --temperature 0.3

# Chat mode with JSON messages
uv run labs-gen --messages-json '[{"role":"user","content":"Write a greeting."}]'

# Load messages from file
uv run labs-gen --messages-json "@/path/to/messages.json"
```

## 🔌 API Compatibility

The API provides OpenAI-compatible endpoints for seamless integration:

### Supported Endpoints

- **`GET /v1/models`** - List available models
- **`POST /v1/chat/completions`** - Create chat completions (streaming and non-streaming)

### Client Usage Example

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required for local server
)

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",  # Use exact model name from your .env
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

### Supported Parameters

- **`model`** - Exact HuggingFace model name (as configured in .env)
- **`messages`** - Chat messages array
- **`max_tokens`** - Maximum tokens to generate
- **`temperature`** - Sampling temperature (0.0-2.0)
- **`top_p`** - Nucleus sampling probability
- **`top_k`** - Top-k sampling (non-standard OpenAI param)
- **`stream`** - Enable streaming responses
- **`stop`** - Stop sequences
- **`frequency_penalty`** - Mapped to repetition_penalty
- **`presence_penalty`** - Frequency penalty variant

## 📖 API Documentation

Interactive API documentation and schema are available when the server is running:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

## 🌊 Streaming Support

### CLI Streaming
```bash
# Enable streaming with --stream flag
uv run labs-gen --prompt "Tell me a story" --stream
```

### API Streaming
```bash
# Enable Server-Sent Events (SSE) streaming
curl -N -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Mistral-7B-Instruct-v0.1", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

> **Note**: Streaming responses use OpenAI-compatible SSE format with `data:` prefixed JSON chunks.

## ⚡ Memory Optimization (Quantization)

Reduce memory usage with optional 4-bit/8-bit quantization via BitsAndBytes:

### Installation
```bash
# Install quantization dependencies
uv sync --extra quantization
```

### Configuration
```toml
# Enable in labs.toml
[quantization]
load_in_4bit = true                    # 4-bit quantization (recommended)
bnb_4bit_quant_type = "nf4"           # NF4 quantization type
bnb_4bit_use_double_quant = true      # Double quantization for better accuracy
bnb_4bit_compute_dtype = "bf16"       # Compute dtype for quantized layers
```

```bash
# Or via environment variables
export LABS_LOAD_IN_4BIT=true
export LABS_BNB_4BIT_COMPUTE_DTYPE=bf16
```

### Important Notes
- **Mutual Exclusivity**: Only enable one of `load_in_4bit` or `load_in_8bit`
- **CUDA Requirement**: BitsAndBytes requires a compatible CUDA environment
- **Memory Savings**: 4-bit quantization can reduce memory usage by ~75%
- **Performance**: Slight inference speed reduction for significant memory savings

## 🔧 Troubleshooting

### CUDA/Driver Issues
- **Driver Compatibility**: Ensure your NVIDIA driver matches your CUDA runtime and PyTorch wheel
- **Nightly Wheels**: If using PyTorch nightly, consider switching to stable wheels for regressions
- **CUDA Environment**: Verify CUDA installation with `nvidia-smi` and `nvcc --version`

### Memory Issues
- **GPU Out-of-Memory**:
  - Reduce `max_new_tokens` parameter
  - Enable 4-bit quantization (`load_in_4bit = true`)
  - Try a smaller model (e.g., 3B instead of 7B parameters)
- **CPU Performance**:
  - Use quantized models for CPU inference
  - Consider smaller models for CPU-only setups
  - Enable `device_map="auto"` on CUDA machines

### Security Considerations
- **`trust_remote_code`**: Only enable for trusted model repositories
  - ⚠️ **Warning**: This executes arbitrary code from the model repository

### Network & Server Issues

#### LAN Access Problems
- **Server Unreachable**:
  ```bash
  # Ensure server binds to all interfaces (default)
  LABS_HOST=0.0.0.0 uv run labs-api
  
  # Check firewall settings
  sudo ufw allow 8000  # Ubuntu/Debian
  
  # Find server IP address
  ip addr show
  hostname -I
  
  # Test from another device
  curl http://<server-ip>:8000/health
  ```

- **Port Conflicts**:
  ```bash
  # Use different port
  LABS_PORT=8001 uv run labs-api
  
  # Kill existing process
  pkill -f labs-api
  # Or find and kill by port
  lsof -ti:8000 | xargs kill
  ```

- **Network Connectivity**:
  - Ensure devices are on the same network/subnet
  - Check router configuration for device isolation
  - Temporarily disable VPN if active
  - Verify no corporate firewall blocking access

## 🧪 Testing

The project includes smoke tests to verify basic functionality:

### Installation
```bash
# Install test dependencies
uv sync --extra test
```

### Running Tests
```bash
# Run all tests
uv run pytest -q

# Run with verbose output
uv run pytest -v
```

### Test Details
- **Smoke Test**: Uses a tiny model to minimize runtime and memory usage
- **Coverage**: Tests basic CLI and API functionality
- **Performance**: Optimized for CI/CD environments

## 🔒 Security

### Code Execution Warning
- **`trust_remote_code`**: Only enable for trusted model repositories
  - ⚠️ **Critical Warning**: This setting allows execution of arbitrary code from the model repository
  - **Recommendation**: Only use with models from verified, trusted sources
  - **Risk**: Malicious models could execute harmful code on your system

### Best Practices
- **Model Sources**: Prefer official model repositories (HuggingFace, Meta, etc.)
- **Network Security**: Consider firewall rules when exposing the API server
- **Access Control**: The API server does not require authentication by design
- **Environment Isolation**: Consider running in containers for production deployments

## 📄 License

This project is open source. Please add your preferred license here.

### Common License Options
- **MIT License**: Permissive license allowing commercial use
- **Apache 2.0**: Permissive license with patent protection
- **GPL v3**: Copyleft license requiring derivative works to be open source
- **BSD 3-Clause**: Simple permissive license

> **Note**: Choose a license that aligns with your project goals and requirements.
