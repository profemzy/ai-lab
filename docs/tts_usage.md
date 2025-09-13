# Text-to-Speech (TTS) Usage Guide

AI Labs includes high-quality text-to-speech functionality powered by the suno/bark model, providing natural-sounding speech synthesis with both CLI and API interfaces.

## 🚀 Quick Start

### CLI Usage

```bash
# Generate speech from simple text
uv run labs-gen --prompt "Hello, world!" --tts-output greeting.wav

# Combine LLM generation with TTS
uv run labs-gen --prompt "Tell me a joke" --tts-output joke.wav --max-new-tokens 50

# Use with chat messages  
uv run labs-gen --messages-json '[{"role":"user","content":"How are you?"}]' --tts-output chat.wav
```

### API Usage

```bash
# Generate speech via REST API
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from the API!"}' \
  --output api_speech.wav
```

## 🔧 Technical Details

### Model Information
- **Model**: suno/bark
- **Quality**: High-quality speech synthesis
- **Languages**: English and multiple other languages
- **Sample Rate**: 24 kHz
- **Bit Depth**: 16-bit PCM
- **Channels**: Mono (1 channel)

### System Requirements
- **GPU**: NVIDIA GPU recommended for optimal performance
- **Memory**: ~2-3GB VRAM for TTS model
- **Dependencies**: PyTorch 2.8.0+, transformers, scipy

## 📖 Detailed Examples

### CLI Examples

```bash
# Basic TTS generation
uv run labs-gen --prompt "The weather is beautiful today" --tts-output weather.wav

# Generate longer speech with streaming
uv run labs-gen --prompt "Tell me about machine learning" --tts-output ml_explanation.wav --stream --max-new-tokens 100

# Use specific model settings
uv run labs-gen --prompt "This is a test" --tts-output test.wav --temperature 0.7 --max-new-tokens 30

# Combine with different input formats
echo '{"text": "Hello from file"}' > input.json
uv run labs-gen --messages-json '@input.json' --tts-output from_file.wav
```

### API Examples

#### Basic Usage
```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Welcome to AI Labs TTS system"}' \
  --output welcome.wav
```

#### Python Integration
```python
import requests

def generate_speech(text, output_file):
    """Generate speech from text using AI Labs TTS API"""
    response = requests.post(
        "http://localhost:8000/v1/tts",
        json={"text": text},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Generated {len(response.content)} bytes of audio")
        return True
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return False

# Example usage
generate_speech("Hello, this is a test!", "test_output.wav")
```

#### Batch Processing
```python
import requests
import time
from pathlib import Path

def batch_tts_generation(texts, output_dir="audio_output"):
    """Generate multiple TTS files from a list of texts"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for i, text in enumerate(texts):
        try:
            response = requests.post(
                "http://localhost:8000/v1/tts",
                json={"text": text},
                timeout=30
            )
            
            if response.status_code == 200:
                output_file = f"{output_dir}/audio_{i+1:03d}.wav"
                with open(output_file, "wb") as f:
                    f.write(response.content)
                print(f"✅ Generated: {output_file}")
            else:
                print(f"❌ Failed for text {i+1}: {response.status_code}")
                
        except requests.RequestException as e:
            print(f"❌ Request failed for text {i+1}: {e}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)

# Example usage
texts = [
    "Welcome to our service",
    "Please hold while we connect you",
    "Thank you for calling",
    "Have a great day"
]

batch_tts_generation(texts)
```

### Advanced Usage

#### Custom Audio Processing
```python
import requests
import wave
import numpy as np

def generate_and_process_speech(text):
    """Generate speech and analyze audio properties"""
    response = requests.post(
        "http://localhost:8000/v1/tts",
        json={"text": text}
    )
    
    if response.status_code == 200:
        # Save to temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(response.content)
        
        # Read and analyze
        with wave.open("temp_audio.wav", "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # Convert to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16)
            duration = len(audio_data) / sample_rate
            
            print(f"📊 Audio Analysis:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Sample Rate: {sample_rate} Hz")
            print(f"   Channels: {channels}")
            print(f"   Bit Depth: {sample_width * 8} bits")
            print(f"   Samples: {len(audio_data)}")
            
            return audio_data, sample_rate
    
    return None, None

# Example usage
audio, sr = generate_and_process_speech("This is a technical test of the audio system")
```

## 🛠️ Integration Patterns

### Web Application Integration
```javascript
// Frontend JavaScript example
async function generateSpeech(text) {
    try {
        const response = await fetch('/v1/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (response.ok) {
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Play audio
            const audio = new Audio(audioUrl);
            audio.play();
            
            return audioUrl;
        } else {
            console.error('TTS generation failed:', response.statusText);
        }
    } catch (error) {
        console.error('TTS request failed:', error);
    }
}

// Usage
generateSpeech("Hello from the web application!");
```

### Discord Bot Integration
```python
import discord
import requests
import io

class TTSBot(discord.Client):
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        if message.content.startswith('!tts '):
            text = message.content[5:]  # Remove '!tts ' prefix
            
            try:
                # Generate speech
                response = requests.post(
                    "http://localhost:8000/v1/tts",
                    json={"text": text}
                )
                
                if response.status_code == 200:
                    # Send audio file to Discord
                    audio_file = discord.File(
                        io.BytesIO(response.content),
                        filename="tts_output.wav"
                    )
                    await message.channel.send(file=audio_file)
                else:
                    await message.channel.send(f"❌ TTS failed: {response.status_code}")
                    
            except Exception as e:
                await message.channel.send(f"❌ Error: {str(e)}")

# Usage
# bot = TTSBot()
# bot.run('your_bot_token')
```

## ⚡ Performance Tips

### Optimal Usage Patterns
1. **Batch Processing**: For multiple texts, use small delays between requests
2. **Caching**: Cache frequently used audio to avoid regeneration
3. **Text Length**: Optimal performance with texts under 200 characters
4. **Concurrent Requests**: Limit to 2-3 concurrent TTS requests per GPU

### Memory Management
```python
import gc
import torch

def cleanup_after_tts():
    """Clean up GPU memory after TTS generation"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Use after batch processing
```

## 🔧 Troubleshooting

### Common Issues

#### GPU Out of Memory
```bash
# Error: CUDA out of memory
# Solution: Restart the API server to clear GPU cache
pkill -f "labs-api"
uv run labs-api
```

#### Audio File Corruption
```bash
# Verify audio file integrity
file output.wav
# Should show: "RIFF (little-endian) data, WAVE audio"

# Check file size (should be > 1KB for normal speech)
ls -lh output.wav
```

#### API Connection Issues
```python
# Test API connectivity
import requests

try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"API Status: {response.status_code}")
except requests.RequestException as e:
    print(f"API Connection Failed: {e}")
```

### Performance Debugging
```python
import time
import requests

def benchmark_tts(text, iterations=5):
    """Benchmark TTS performance"""
    times = []
    sizes = []
    
    for i in range(iterations):
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/v1/tts",
            json={"text": text}
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            duration = end_time - start_time
            size = len(response.content)
            
            times.append(duration)
            sizes.append(size)
            
            print(f"Iteration {i+1}: {duration:.2f}s, {size} bytes")
    
    if times:
        avg_time = sum(times) / len(times)
        avg_size = sum(sizes) / len(sizes)
        print(f"\n📊 Average: {avg_time:.2f}s, {avg_size} bytes")

# Example usage
benchmark_tts("This is a performance test of the TTS system")
```

## 🔒 Security Considerations

### Input Validation
```python
import re

def sanitize_tts_input(text):
    """Sanitize text input for TTS"""
    # Remove potentially harmful content
    text = re.sub(r'[<>]', '', text)  # Remove HTML-like tags
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text

# Example usage
safe_text = sanitize_tts_input(user_input)
```

### Rate Limiting
```python
import time
from collections import defaultdict

class TTSRateLimiter:
    def __init__(self, requests_per_minute=10):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip):
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_ip].append(now)
        return True

# Usage in web application
rate_limiter = TTSRateLimiter(requests_per_minute=10)
```

## 📈 Production Deployment

### Docker Configuration
```dockerfile
# Add TTS-specific environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV TTS_MODEL_CACHE="/app/tts_cache"

# Create TTS cache directory
RUN mkdir -p /app/tts_cache

# Pre-download TTS model (optional)
RUN python -c "from transformers import pipeline; pipeline('text-to-speech', 'suno/bark')"
```

### Load Balancing
```nginx
# Nginx configuration for TTS load balancing
upstream tts_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

location /v1/tts {
    proxy_pass http://tts_backend;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    
    # Increase timeout for TTS generation
    proxy_read_timeout 30s;
    proxy_connect_timeout 5s;
}
```

---

For more information, see the main [README.md](README.md) or visit the API documentation at `http://localhost:8000/docs` when the server is running.