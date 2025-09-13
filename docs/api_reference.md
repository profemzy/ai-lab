# AI Labs API Reference

AI Labs provides OpenAI-compatible REST API endpoints for LLM inference, text embeddings, and text-to-speech generation.

## 🚀 Quick Start

Start the API server:
```bash
uv run labs-api
```

The server will be available at:
- **Local**: http://localhost:8000
- **Network**: http://0.0.0.0:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔌 Endpoints Overview

| Endpoint | Method | Description | OpenAI Compatible |
|----------|--------|-------------|-------------------|
| `/v1/models` | GET | List available models | ✅ Yes |
| `/v1/chat/completions` | POST | Chat completions | ✅ Yes |
| `/v1/embeddings` | POST | Text embeddings | ✅ Yes |
| `/v1/similarities` | POST | Text similarities | ❌ Extension |
| `/v1/tts` | POST | Text-to-speech | ❌ Extension |
| `/health` | GET | Health check | ❌ Utility |

## 📖 Detailed Reference

### Models Endpoint

**GET /v1/models**

List available language models.

```bash
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "object": "model",
      "created": 1703123456,
      "owned_by": "labs"
    }
  ]
}
```

### Chat Completions Endpoint

**POST /v1/chat/completions**

Generate text completions from chat messages.

**Request:**
```json
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

**Parameters:**
- `model` (string, required): Model identifier
- `messages` (array, required): Conversation messages
- `max_tokens` (integer): Maximum tokens to generate (default: 128)
- `temperature` (float): Sampling temperature 0.0-2.0 (default: 0.7)
- `top_p` (float): Nucleus sampling probability (default: 0.9)
- `top_k` (integer): Top-k sampling (non-standard)
- `stream` (boolean): Enable streaming (default: false)
- `stop` (string|array): Stop sequences
- `frequency_penalty` (float): Frequency penalty -2.0 to 2.0
- `presence_penalty` (float): Presence penalty -2.0 to 2.0

**Response (Non-streaming):**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1703123456,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 12,
    "total_tokens": 22
  }
}
```

**Streaming Response:**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1703123456,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1703123456,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"delta":{"content":"Hello!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1703123456,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Embeddings Endpoint

**POST /v1/embeddings**

Generate text embeddings for semantic search and similarity.

**Request:**
```json
{
  "input": [
    "Hello world",
    "How are you?"
  ],
  "model": "google/embeddinggemma-300m",
  "encoding_format": "float"
}
```

**Parameters:**
- `input` (string|array, required): Text(s) to embed
- `model` (string): Embedding model (default: google/embeddinggemma-300m)
- `encoding_format` (string): Format for embeddings (default: float)
- `dimensions` (integer): Embedding dimensions (not supported)

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1234, -0.5678, ...],
      "index": 0
    },
    {
      "object": "embedding", 
      "embedding": [0.9876, -0.4321, ...],
      "index": 1
    }
  ],
  "model": "google/embeddinggemma-300m",
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 6
  }
}
```

### Similarities Endpoint (Extension)

**POST /v1/similarities**

Compute pairwise similarities between texts.

**Request:**
```json
{
  "sentences": [
    "Hello world",
    "Hi there",
    "Good morning",
    "How are you?"
  ],
  "model": "google/embeddinggemma-300m"
}
```

**Response:**
```json
{
  "similarities": [
    [1.0, 0.85, 0.42, 0.23],
    [0.85, 1.0, 0.39, 0.28],
    [0.42, 0.39, 1.0, 0.15],
    [0.23, 0.28, 0.15, 1.0]
  ],
  "model": "google/embeddinggemma-300m",
  "shape": [4, 4]
}
```

### Text-to-Speech Endpoint (Extension)

**POST /v1/tts**

Generate speech audio from text.

**Request:**
```json
{
  "text": "Hello, this is AI Labs text-to-speech!",
  "model": "suno/bark"
}
```

**Parameters:**
- `text` (string, required): Text to convert to speech
- `model` (string): TTS model (default: suno/bark)

**Response:**
- **Content-Type**: `audio/wav`
- **Body**: Binary WAV audio data

**Example:**
```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output speech.wav
```

### Health Check Endpoint

**GET /health**

Check API server health.

**Response:**
```json
{
  "status": "ok"
}
```

## 🔧 Configuration

### Environment Variables

Control the API server behavior with these environment variables:

```bash
# Server Configuration
export LABS_HOST=0.0.0.0          # Server host (default: 0.0.0.0)
export LABS_PORT=8000             # Server port (default: 8000)
export LABS_PRELOAD_ON_START=true # Preload model on startup

# Model Configuration  
export LABS_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export LABS_MAX_NEW_TOKENS=128
export LABS_TEMPERATURE=0.7
export LABS_TOP_P=0.9

# Hardware Optimization
export LABS_DEVICE_MAP=auto
export LABS_TORCH_DTYPE=bf16
export LABS_LOAD_IN_4BIT=false

# Embedding Model
export LABS_EMBEDDING_MODEL="google/embeddinggemma-300m"

# HuggingFace Authentication (for gated models)
export HF_TOKEN=your_token_here
```

## 🧑‍💻 Client Libraries

### Python (OpenAI)

```python
import openai

# Initialize client
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required
)

# Chat completion
response = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

print(response.choices[0].message.content)

# Text embeddings
embeddings = client.embeddings.create(
    model="google/embeddinggemma-300m",
    input=["Hello world", "How are you?"]
)

print(f"Generated {len(embeddings.data)} embeddings")

# Text-to-speech (custom endpoint)
import requests
tts_response = requests.post(
    "http://localhost:8000/v1/tts",
    json={"text": "Hello from Python!"}
)

with open("output.wav", "wb") as f:
    f.write(tts_response.content)
```

### JavaScript/Node.js

```javascript
// Using fetch API
async function chatCompletion(message) {
    const response = await fetch('http://localhost:8000/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            messages: [{ role: 'user', content: message }],
            max_tokens: 50
        })
    });

    const data = await response.json();
    return data.choices[0].message.content;
}

// TTS Generation
async function generateSpeech(text) {
    const response = await fetch('http://localhost:8000/v1/tts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
    });

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    
    // Play or download audio
    const audio = new Audio(audioUrl);
    audio.play();
}

// Usage
chatCompletion("Hello!").then(console.log);
generateSpeech("Hello from JavaScript!");
```

### cURL Examples

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Streaming chat
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 100,
    "stream": true
  }'

# Generate embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "How are you?"],
    "model": "google/embeddinggemma-300m"
  }'

# Text-to-speech
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output speech.wav
```

## 🔒 Security Considerations

### Authentication

The API server does not require authentication by default. For production deployments:

1. **Reverse Proxy**: Use nginx/Apache with authentication
2. **API Gateway**: Implement rate limiting and access control
3. **Network Security**: Restrict access to trusted networks

### Input Validation

All endpoints validate input data:
- Text length limits prevent resource exhaustion
- Model names are validated against available models
- Malformed JSON returns appropriate error codes

### Rate Limiting

Consider implementing rate limiting for production:

```python
# Example rate limiting with slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/chat/completions")
@limiter.limit("10/minute")
async def chat_completions(request: Request, ...):
    # Endpoint implementation
    pass
```

## 📊 Performance Optimization

### Model Loading

- **Preload Models**: Set `LABS_PRELOAD_ON_START=true`
- **GPU Memory**: Use `device_map="auto"` for optimal allocation
- **Quantization**: Enable 4-bit quantization for memory efficiency

### Request Optimization

- **Batch Requests**: Group multiple requests when possible
- **Streaming**: Use streaming for long responses
- **Connection Pooling**: Reuse HTTP connections

### Monitoring

Monitor these metrics in production:
- **Response Time**: Target <2s for chat completions
- **GPU Memory**: Keep usage <90% of available VRAM
- **Request Rate**: Monitor requests per minute
- **Error Rate**: Track failed requests and timeouts

## 🐛 Error Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid JSON, missing required fields |
| 422 | Validation Error | Invalid parameter values |
| 500 | Internal Server Error | Model loading failure, GPU OOM |
| 503 | Service Unavailable | Server overloaded or starting up |

**Example Error Response:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "text"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

## 🔗 Related Documentation

- [README.md](../README.md) - Project overview and setup
- [TTS_USAGE.md](TTS_USAGE.md) - Detailed TTS usage guide
- [Interactive API Docs](http://localhost:8000/docs) - Live API documentation
- [OpenAPI Schema](http://localhost:8000/openapi.json) - Machine-readable API specification

---

For more information or support, please check the project repository or API documentation at http://localhost:8000/docs when the server is running.