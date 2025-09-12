# AI Labs — OpenAI-Compatible Local LLM Inference Server

A high-performance, production-ready local LLM inference server that provides OpenAI-compatible APIs for seamless integration with existing applications. Built on HuggingFace Transformers with optimized defaults for NVIDIA GPUs.

## 🚀 Key Features

- **🎨 Beautiful Terminal UI**: Modern, interactive CLI with progress bars, animations, and rich formatting
- **💬 Interactive Chat Mode**: Full conversation management with save/load functionality and session statistics
- **🧠 Hybrid AI Intelligence**: RAG system for transaction questions + full LLM capabilities for general queries
- **🔌 OpenAI API Compatibility**: Drop-in replacement for OpenAI's API with `/v1/chat/completions`, `/v1/embeddings`, `/v1/similarities`, and `/v1/models` endpoints
- **⚡ High-Performance Inference**: Optimized for NVIDIA GPUs with automatic BF16/FP16 precision and device mapping
- **🚀 Flexible Deployment**: Enhanced CLI tool for development and FastAPI server for production workloads
- **⚙️ Advanced Configuration**: TOML-based configuration with environment variable overrides
- **💾 Memory Optimization**: Optional 4-bit/8-bit quantization support via BitsAndBytes
- **🌊 Streaming Support**: Real-time token streaming with Server-Sent Events (SSE)

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
├── embeddings.py   # Text embedding generator with sentence-transformers
├── cli.py          # Enhanced command-line interface (labs-gen)
├── ui.py           # Beautiful terminal UI components
├── interactive.py  # Interactive chat & conversation management
├── api.py          # FastAPI server with OpenAI-compatible endpoints
├── config.py       # Configuration management (TOML + env + CLI)
└── labs.toml       # Default configuration file
```

## ⚡ Quick Start

### Installation

```bash
# Basic installation (includes beautiful terminal UI)
uv sync

# With quantization and testing support
uv sync --extra quantization --extra test

# Dependencies for enhanced CLI:
# - rich: Beautiful terminal formatting and progress bars
# - typer: Enhanced CLI argument parsing  
# - inquirer: Interactive prompts and menus
```

### CLI Usage

```bash
# 🎨 Beautiful Interactive Chat Mode (NEW!)
uv run labs-gen --interactive

# 🧠 Transaction Intelligence - Instant, Perfect Accuracy
uv run labs-gen --prompt "What was my largest expense?"
uv run labs-gen --prompt "Give me a financial summary"
uv run labs-gen --prompt "How much did I spend on software?"

# Simple text generation with enhanced UI
uv run labs-gen --prompt "Hello! Who are you?" --max-new-tokens 64

# Chat-style interaction
uv run labs-gen --messages-json '[{"role":"user","content":"Who are you?"}]' --max-new-tokens 64

# Streaming generation with beautiful terminal UI
uv run labs-gen --prompt "Tell me a short joke about GPUs." --stream

# Show GPU information
uv run labs-gen --show-gpu

# Display configuration
uv run labs-gen --show-config

# Plain output (no UI styling)
uv run labs-gen --prompt "Hello" --no-ui
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

# 🧠 Transaction Intelligence API - Perfect accuracy, instant response
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.1",
    "messages": [{"role": "user", "content": "What was my largest expense?"}],
    "max_tokens": 100
  }'

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

# Embedding configuration
export LABS_EMBEDDING_MODEL="google/embeddinggemma-300m"  # Embedding model
export LABS_EMBEDDING_DEVICE="auto"                       # Auto GPU/CPU selection
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

### 🎨 Enhanced CLI Features

The CLI now includes a beautiful terminal interface with rich formatting, progress bars, and interactive features.

#### **Interactive Chat Mode**
```bash
# Start interactive mode with full conversation management
uv run labs-gen --interactive

# Commands available in interactive mode:
/help               # Show available commands
/save               # Save conversation with custom title
/load               # Load from saved conversations
/list               # List all saved conversations
/clear              # Clear current conversation
/stats              # Show session statistics
/config             # Display model configuration
/exit, /quit        # Exit interactive mode
```

#### **Information Display**
```bash
# Show GPU information with beautiful formatting
uv run labs-gen --show-gpu

# Display current configuration in styled table
uv run labs-gen --show-config

# Disable rich UI for plain text output
uv run labs-gen --prompt "Hello" --no-ui
```

#### **Advanced Generation Examples**
```bash
# Interactive mode with specific model
LABS_MODEL=codellama/CodeLlama-7b-Instruct-hf uv run labs-gen --interactive

# Deterministic generation with enhanced UI
uv run labs-gen --prompt "Summarize CUDA BF16" --no-sample --max-new-tokens 80

# Custom model with beautiful progress display
uv run labs-gen \
  --prompt "What is device_map=auto?" \
  --model "mistralai/Mistral-7B-Instruct-v0.3" \
  --temperature 0.3 \
  --stream

# Chat mode with JSON messages and UI
uv run labs-gen --messages-json '[{"role":"user","content":"Write a greeting."}]'

# Load messages from file with progress indication
uv run labs-gen --messages-json "@/path/to/messages.json"
```

#### **UI Features**
- **🎨 Welcome Banner**: ASCII art with gradient colors
- **📊 Progress Bars**: Animated model loading with stages  
- **📋 Configuration Tables**: Styled display of settings
- **🎯 Generation Stats**: Performance metrics after each response
- **⚡ Real-time Streaming**: Beautiful typing effects
- **💾 Conversation Management**: Save/load chat history
- **🔧 Error Handling**: Helpful error messages with suggestions

#### **Conversation History**

The interactive mode automatically manages conversation history with persistent storage:

```bash
# Conversations are saved to ~/.labs/conversations/
# Each conversation includes:
# - Full message history with timestamps
# - Session statistics (tokens, response times)
# - Custom titles and metadata

# Example conversation management:
You: Hello! Can you help me with Python?
🤖 Assistant: [response]

You: /save
Enter conversation title: Python Help Session
✅ Conversation saved: /home/user/.labs/conversations/Python Help Session.json

# Later, load the conversation:
You: /load
┌─────┬──────────────────────┬────────────┬──────────┐
│ #   │ Title               │ Date       │ Messages │
├─────┼──────────────────────┼────────────┼──────────┤
│ 1   │ Python Help Session │ 2024-01-15 │ 8        │
│ 2   │ Code Review Chat    │ 2024-01-14 │ 12       │
└─────┴──────────────────────┴────────────┴──────────┘
Enter conversation number: 1
✅ Loaded conversation: Python Help Session
```

#### **Example Session Output**

When you run `uv run labs-gen --interactive`, you'll see:

```
╭─────────────────────────────────────────────────────────────╮
│  █████╗ ██╗    ██╗     █████╗ ██████╗ ███████╗              │
│ ██╔══██╗██║    ██║    ██╔══██╗██╔══██╗██╔════╝              │  
│ ███████║██║    ██║    ███████║██████╔╝███████╗              │
│ ██╔══██║██║    ██║    ██╔══██║██╔══██╗╚════██║              │
│ ██║  ██║██║    ██████╗██║  ██║██████╔╝███████║              │
│ ╚═╝  ╚═╝╚═╝    ╚═════╝╚═╝  ╚═╝╚═════╝ ╚══════╝              │
│           🤖 Local LLM Inference Server                      │
│        OpenAI-Compatible • GPU-Optimized • Production-Ready │
╰─────────────────────────────────────────────────────────────╯

🔧 Configuration Summary
┌────────────────────┬──────────────────────────────────────┬────────┐
│ Setting           │ Value                                │ Status │
├────────────────────┼──────────────────────────────────────┼────────┤
│ Model             │ deepseek-ai/DeepSeek-R1-0528-Qwen3-8B │ ✓      │
│ Max Tokens        │ 128                                  │ ✓      │
│ Temperature       │ 0.7                                  │ ✓      │
│ Device Map        │ auto                                 │ ✓      │
│ Precision         │ bfloat16                             │ ✓      │
│ Quantization      │ Disabled                             │ ✓      │
└────────────────────┴──────────────────────────────────────┴────────┘

🤖 Loading Model...
🔍 Resolving model configuration...     ████████████████████ 100%
📥 Downloading tokenizer...              ████████████████████ 100%  
⚙️  Loading model architecture...        ████████████████████ 100%
🧠 Loading model weights...              ████████████████████ 100%
🎯 Optimizing for hardware...            ████████████████████ 100%
✅ Model ready!

🚀 Starting Interactive Chat Mode
Type /help for commands, or just start chatting!

You: 
```

## 🔌 API Compatibility

The API provides OpenAI-compatible endpoints for seamless integration:

### Supported Endpoints

- **`GET /v1/models`** - List available models
- **`POST /v1/chat/completions`** - Create chat completions (streaming and non-streaming)
- **`POST /v1/embeddings`** - Generate text embeddings
- **`POST /v1/similarities`** - Compute pairwise similarities between texts

### Client Usage Example

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required for local server
)

# 🧠 Transaction Intelligence - Perfect accuracy for transaction questions
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    messages=[{"role": "user", "content": "What was my largest expense?"}],
    max_tokens=100
)

# General AI capabilities for everything else
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",  # Use exact model name from your .env
    messages=[{"role": "user", "content": "Explain machine learning"}],
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

## 🧠 Transaction Intelligence (RAG System)

Your AI Labs system includes an advanced **hybrid intelligence architecture** that automatically detects transaction questions and provides instant, perfect accuracy using your transaction data.

### How It Works

```python
# Automatic intelligent routing:
user: "What was my largest expense?"
# → RAG system calculates directly from CSV: 0.05s, 100% accurate

user: "Explain quantum computing" 
# → Full LLM inference: 3-8s, full AI reasoning
```

### Supported Transaction Questions

**Financial Summaries:**
```bash
uv run labs-gen --prompt "Give me a financial summary"
uv run labs-gen --prompt "What are my total expenses?"
uv run labs-gen --prompt "What is my total income?"
```

**Spending Analysis:**
```bash
uv run labs-gen --prompt "What was my largest expense?"
uv run labs-gen --prompt "How much did I spend on software?"
uv run labs-gen --prompt "What are my main spending categories?"
```

**Time-based Queries:**
```bash
uv run labs-gen --prompt "How much did I spend in December 2024?"
uv run labs-gen --prompt "What are my recent transactions?"
```

### Why This is Better Than Pure RAG or Pure LLM

**🎯 Perfect Accuracy**: Transaction questions get mathematically perfect answers calculated directly from your CSV data - no hallucination possible.

**⚡ Instant Response**: Transaction queries respond in ~50ms vs 3-8 seconds for LLM inference.

**🤖 Full AI Capabilities**: Non-transaction questions get complete LLM reasoning and generation.

**🔄 Seamless Integration**: Automatic detection and routing - no manual switching required.

### Transaction Data Setup

Place your transaction CSV file at `data/all_transactions.csv` with columns:
- `Date`: Transaction date
- `Type`: "Expense" or "Income"  
- `Amount`: Transaction amount (negative for expenses)
- `Description`: Transaction description
- `Category`: Spending category (optional)

## 🔗 Text Embeddings

The server provides OpenAI-compatible text embedding functionality using sentence-transformers models, enabling semantic search, similarity computation, and vector database integration.

### Embedding Configuration

Configure embedding models via environment variables:

```bash
# Embedding model configuration
export LABS_EMBEDDING_MODEL="google/embeddinggemma-300m"  # Default embedding model
export LABS_EMBEDDING_DEVICE="auto"                      # Auto GPU/CPU selection
```

### Embedding API Examples

```bash
# Generate embeddings for multiple texts
curl -s -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "That is a happy person",
      "That is a happy dog",
      "Today is a sunny day"
    ],
    "model": "google/embeddinggemma-300m"
  }'

# Compute pairwise similarities
curl -s -X POST http://localhost:8000/v1/similarities \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "That is a happy person",
      "That is a happy dog", 
      "That is a very happy person",
      "Today is a sunny day"
    ],
    "model": "google/embeddinggemma-300m"
  }'
```

### Python Client Usage

```python
import openai
import numpy as np

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Generate embeddings
response = client.embeddings.create(
    model="google/embeddinggemma-300m",
    input=[
        "That is a happy person",
        "That is a happy dog",
        "Today is a sunny day"
    ]
)

embeddings = [data.embedding for data in response.data]
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

# Compute similarities using requests (custom endpoint)
import requests

similarity_response = requests.post(
    "http://localhost:8000/v1/similarities",
    json={
        "texts": [
            "That is a happy person",
            "That is a happy dog",
            "That is a very happy person", 
            "Today is a sunny day"
        ],
        "model": "google/embeddinggemma-300m"
    }
)

similarities = similarity_response.json()["similarities"]
print(f"Similarity matrix shape: {np.array(similarities).shape}")
```

### Direct Module Usage

For advanced use cases, you can use the embedding module directly:

```python
from labs.embeddings import get_embedding_generator

# Get the global embedding generator
embedding_gen = get_embedding_generator()

# Generate embeddings
texts = ["Hello world", "How are you?", "Good morning"]
embeddings = embedding_gen.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")  # [3, 768]

# Compute similarities
similarities = embedding_gen.similarity(embeddings, embeddings)
print(f"Similarities shape: {similarities.shape}")  # [3, 3]

# Combined encode and similarity
embeddings, similarities = embedding_gen.encode_and_similarity(texts)
```

### Supported Embedding Models

Popular sentence-transformers models that work well:

```bash
# Google's EmbeddingGemma (default, optimized for efficiency)
LABS_EMBEDDING_MODEL="google/embeddinggemma-300m"

# All-MiniLM (lightweight, good performance)
LABS_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# All-mpnet (higher quality, larger model)
LABS_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"

# E5 models (excellent multilingual support)
LABS_EMBEDDING_MODEL="intfloat/e5-base-v2"
LABS_EMBEDDING_MODEL="intfloat/e5-large-v2"
```

### Memory Considerations

- **google/embeddinggemma-300m**: ~1.2GB VRAM, 768-dimensional embeddings
- **all-MiniLM-L6-v2**: ~90MB VRAM, 384-dimensional embeddings  
- **all-mpnet-base-v2**: ~420MB VRAM, 768-dimensional embeddings
- **e5-large-v2**: ~1.3GB VRAM, 1024-dimensional embeddings

> **Note**: Embedding models are much smaller than LLMs and can run efficiently on both GPU and CPU.

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

## 🐳 Docker Deployment

The project includes Docker support for containerized deployment with GPU acceleration.

### Building the Image

```bash
# Build the Docker image
docker build -t labs-api .
```

### Running the Container

```bash
# Run with GPU support and .env file configuration
docker run -d --name labs-api \
  --gpus all \
  -p 8000:8000 \
  -v labs_hf_cache:/data \
  -v $(pwd)/.env:/app/.env \
  labs-api:latest

# Check container status
docker ps

# View logs
docker logs labs-api

# Access TUI Chat 
docker exec -it lab-api labs-gen --interactive

```

### Docker Configuration

The container automatically:
- **Loads Configuration**: Reads from mounted `.env` file
- **GPU Access**: Uses `--gpus all` for NVIDIA GPU acceleration
- **Model Caching**: Persists HuggingFace models in `labs_hf_cache` volume
- **Health Checks**: Built-in health monitoring on `/health` endpoint

### Required .env Settings

Ensure your `.env` file includes the HuggingFace token for gated models:

```bash
# Core model configuration
LABS_MODEL=mistralai/Mistral-7B-Instruct-v0.1
LABS_DEVICE_MAP=auto
LABS_TORCH_DTYPE=bf16

# HuggingFace authentication (required for gated models)
HF_TOKEN=your_token_here
```

### Docker Management

```bash
# Stop the container
docker stop labs-api-container

# Remove the container
docker rm labs-api-container

# View container logs in real-time
docker logs -f labs-api-container

# Execute commands inside the container
docker exec labs-api-container labs-gen --prompt "Hello!" --max-new-tokens 32
```

### Production Considerations

- **Resource Limits**: Consider setting memory and CPU limits for production
- **Networking**: Use Docker networks for multi-container deployments
- **Secrets Management**: Use Docker secrets or external secret management for tokens
- **Monitoring**: Integrate with container monitoring solutions

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
