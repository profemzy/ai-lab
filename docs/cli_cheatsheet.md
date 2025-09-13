# AI Labs CLI Cheat Sheet

Complete reference for all `labs-gen` command-line options and interactive mode commands.

## 🚀 Quick Start

```bash
# Basic text generation
labs-gen --prompt "Your question here"

# Interactive chat mode
labs-gen --interactive

# Help and information
labs-gen --help
```

## 📝 Command Structure

```bash
labs-gen [INPUT_OPTIONS] [MODEL_OPTIONS] [GENERATION_OPTIONS] [OUTPUT_OPTIONS] [UTILITY_OPTIONS]
```

---

## 🔤 Input Options

### Text Input
```bash
# Simple text prompt
--prompt "Your text here"

# Chat messages as JSON string
--messages-json '[{"role":"user","content":"Hello!"}]'

# Chat messages from file
--messages-json '@/path/to/messages.json'
```

**Examples:**
```bash
labs-gen --prompt "Explain quantum computing"
labs-gen --messages-json '[{"role":"user","content":"What is Python?"}]'
labs-gen --messages-json '@conversation.json'
```

---

## 🤖 Model Options

### Model Selection
```bash
--model MODEL_NAME              # Specify model (overrides config)
--trust-remote-code             # Enable for custom models (security risk)
--config CONFIG_FILE            # Custom config file path
```

**Examples:**
```bash
labs-gen --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --prompt "Hello"
labs-gen --model "codellama/CodeLlama-7b-Instruct-hf" --prompt "Write Python code"
labs-gen --config custom.toml --prompt "Test"
```

---

## ⚙️ Generation Parameters

### Sampling Control
```bash
--max-new-tokens N              # Maximum tokens to generate (default: 128)
--temperature T                 # Sampling temperature 0.0-2.0 (default: 0.7)
--top-p P                      # Nucleus sampling 0.0-1.0 (default: 0.9)
--top-k K                      # Top-k sampling (default: none)
--repetition-penalty F         # Repetition penalty factor (default: 1.0)
--no-sample                    # Deterministic decoding (temperature=0)
```

**Examples:**
```bash
# Creative generation
labs-gen --prompt "Tell a story" --temperature 0.9 --max-new-tokens 200

# Precise/factual generation  
labs-gen --prompt "What is 2+2?" --temperature 0.1 --max-new-tokens 10

# Deterministic output
labs-gen --prompt "Define AI" --no-sample --max-new-tokens 50
```

### Chat Template Control
```bash
--no-chat-template             # Disable chat template formatting
--no-generation-prompt         # Skip generation prompt tokens
```

---

## 🌊 Output Options

### Streaming and UI
```bash
--stream                       # Stream tokens in real-time
--no-ui                       # Disable rich terminal UI (plain text)
```

### Text-to-Speech
```bash
--tts-output FILE.wav          # Generate speech audio from output
```

**Examples:**
```bash
# Streaming with beautiful UI
labs-gen --prompt "Tell me about space" --stream

# Plain text output
labs-gen --prompt "Hello" --no-ui

# Generate speech
labs-gen --prompt "Welcome to AI Labs" --tts-output welcome.wav

# Combine: stream + TTS
labs-gen --prompt "Explain machine learning" --stream --tts-output explanation.wav
```

---

## 🛠️ Utility Options

### Information Display
```bash
--show-config                  # Display current configuration
--show-gpu                     # Show GPU information  
--help, -h                     # Show help message
```

### Interactive Mode
```bash
--interactive, -i              # Start interactive chat session
```

**Examples:**
```bash
labs-gen --show-config
labs-gen --show-gpu
labs-gen --interactive
```

---

## 💬 Interactive Mode Commands

Enter interactive mode with `labs-gen --interactive` or `labs-gen -i`

### Core Commands
| Command | Shortcut | Description |
|---------|----------|-------------|
| `/help` | `/h` | Show help message |
| `/exit` | `/quit`, `/q` | Exit interactive mode |
| `/clear` | `/c` | Clear current conversation |

### Conversation Management
| Command | Shortcut | Description |
|---------|----------|-------------|
| `/save` | `/s` | Save current conversation with custom title |
| `/load` | `/l` | Load a saved conversation |
| `/list` | `/ls` | List all saved conversations |

### Information
| Command | Shortcut | Description |
|---------|----------|-------------|
| `/stats` | `/st` | Show session statistics (tokens, timing) |
| `/config` | `/cfg` | Display current model configuration |
| `/model` | - | Switch model (if supported) |

**Interactive Mode Examples:**
```
You: Hello, how are you?
🤖 Assistant: I'm doing well, thank you for asking!

You: /save
Enter conversation title: First Chat
✅ Conversation saved: /home/user/.labs/conversations/First Chat.json

You: /stats
📊 Session Statistics:
   Messages: 2
   Total Tokens: 25
   Average Response Time: 1.2s

You: /exit
```

---

## 📋 Common Usage Patterns

### Quick Questions
```bash
# Fast answers
labs-gen --prompt "What is Python?" --max-new-tokens 50 --no-ui

# Technical explanations
labs-gen --prompt "Explain Docker containers" --temperature 0.3 --stream
```

### Code Generation
```bash
# Python code
labs-gen --model "codellama/CodeLlama-7b-Instruct-hf" \
         --prompt "Write a function to sort a list" \
         --temperature 0.2

# Multiple languages
labs-gen --prompt "Convert this to JavaScript: print('hello')" --max-new-tokens 100
```

### Creative Writing
```bash
# Stories
labs-gen --prompt "Write a sci-fi short story" \
         --temperature 0.8 \
         --max-new-tokens 300 \
         --stream

# Poetry
labs-gen --prompt "Write a haiku about coding" --temperature 0.6
```

### Chat Conversations
```bash
# Multi-turn conversation from file
cat > conversation.json << 'EOF'
[
  {"role": "user", "content": "Hi! I'm learning Python."},
  {"role": "assistant", "content": "That's great! What would you like to know?"},
  {"role": "user", "content": "How do I create a list?"}
]
EOF

labs-gen --messages-json '@conversation.json' --stream
```

### Voice Generation
```bash
# Generate explanations with speech
labs-gen --prompt "Explain how neural networks work" \
         --max-new-tokens 150 \
         --tts-output neural_networks.wav \
         --stream

# Quick voice responses
labs-gen --prompt "What is the weather like?" \
         --tts-output weather.wav \
         --no-ui
```

---

## 🔧 Configuration Examples

### Environment Variables
```bash
# Set defaults via environment
export LABS_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export LABS_MAX_NEW_TOKENS=256
export LABS_TEMPERATURE=0.7

# Then use CLI normally
labs-gen --prompt "Hello"
```

### Custom Config File
```toml
# custom.toml
[generation]
model_name = "Qwen/Qwen2.5-7B-Instruct"
max_new_tokens = 200
temperature = 0.8
top_p = 0.95
```

```bash
labs-gen --config custom.toml --prompt "Tell me a joke"
```

---

## 🔄 Workflow Examples

### Development Workflow
```bash
# 1. Check system
labs-gen --show-gpu
labs-gen --show-config

# 2. Quick test
labs-gen --prompt "Hello" --no-ui

# 3. Interactive session
labs-gen --interactive
```

### Content Creation Workflow
```bash
# 1. Generate content
labs-gen --prompt "Write an article about AI" \
         --max-new-tokens 500 \
         --temperature 0.7 \
         --stream > article.txt

# 2. Convert to speech
labs-gen --prompt "@article.txt" \
         --tts-output article.wav

# 3. Interactive editing
labs-gen --interactive
```

### Batch Processing
```bash
# Process multiple prompts
while IFS= read -r prompt; do
    echo "Processing: $prompt"
    labs-gen --prompt "$prompt" \
             --max-new-tokens 100 \
             --no-ui > "output_$(echo "$prompt" | tr ' ' '_').txt"
done < prompts.txt
```

---

## 📁 File Formats

### Messages JSON Format
```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user", 
    "content": "Hello!"
  },
  {
    "role": "assistant",
    "content": "Hi there! How can I help you?"
  },
  {
    "role": "user",
    "content": "What is machine learning?"
  }
]
```

### Conversation Save Format
Saved conversations are stored in `~/.labs/conversations/` as JSON files:
```json
{
  "title": "Machine Learning Discussion",
  "created": "2024-01-15T10:30:00Z",
  "messages": [...],
  "stats": {
    "total_tokens": 1250,
    "message_count": 12,
    "average_response_time": 2.3
  }
}
```

---

## 🚨 Important Notes

### Security
- ⚠️ **`--trust-remote-code`**: Only use with trusted models (executes arbitrary code)
- 🔒 **Model Sources**: Prefer official HuggingFace repositories

### Performance Tips
- 🚀 **GPU Memory**: Use `--show-gpu` to monitor VRAM usage
- ⚡ **Streaming**: Use `--stream` for perceived better performance
- 💾 **Token Limits**: Set appropriate `--max-new-tokens` for your use case

### Troubleshooting
```bash
# Check system status
labs-gen --show-gpu
labs-gen --show-config

# Test with minimal settings
labs-gen --prompt "Test" --max-new-tokens 10 --no-ui

# Debug with plain output
labs-gen --prompt "Debug test" --no-ui --temperature 0.1
```

---

## 🔗 Related Documentation

- [README.md](../README.md) - Project overview and setup
- [API_REFERENCE.md](API_REFERENCE.md) - API endpoints
- [TTS_USAGE.md](TTS_USAGE.md) - Text-to-speech guide
- [Interactive Docs](http://localhost:8000/docs) - Live API documentation

---

**Quick Reference Card:**
```bash
# Essential commands
labs-gen --prompt "text"           # Basic generation
labs-gen --interactive             # Chat mode
labs-gen --show-config             # Check settings
labs-gen --help                    # Full help

# Power user
labs-gen --prompt "text" --stream --tts-output audio.wav --max-new-tokens 200
```