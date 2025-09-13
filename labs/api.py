from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from labs import GenerationConfig, HFGenerator
from labs.config import load_config
from labs.embeddings import get_embedding_generator
from labs.tts import get_tts_instance


# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author (system, user, assistant)")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    # Core OpenAI parameters
    model: str = Field(default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages comprising the conversation")
    
    # Generation parameters (OpenAI-compatible names)
    max_tokens: Optional[int] = Field(default=128, ge=1, le=4096, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling (non-standard OpenAI param)")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Streaming
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial progress")
    
    # Additional parameters
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    n: Optional[int] = Field(default=1, ge=1, le=1, description="Number of completions (only 1 supported)")
    
    # Labs-specific parameters (optional)
    trust_remote_code: Optional[bool] = Field(default=False, description="Enable trust_remote_code for custom models")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str  # "stop", "length", "content_filter"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# Streaming response models
class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]  # Can contain "role", "content", etc.
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# Model list response
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "labs"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Embedding request/response models
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field(default="google/embeddinggemma-300m", description="Embedding model to use")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format (float or base64)")
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions (not supported)")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# Similarity request/response models
class SimilarityRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of sentences to compute similarities for")
    model: str = Field(default="google/embeddinggemma-300m", description="Embedding model to use")


class SimilarityResponse(BaseModel):
    similarities: List[List[float]] = Field(..., description="Similarity matrix")
    model: str
    shape: List[int] = Field(..., description="Shape of similarity matrix")


app = FastAPI(
    title="Labs OpenAI-Compatible API",
    description="OpenAI-compatible API for local LLM generation",
    version="1.0.0"
)

# Add CORS middleware with broader origins for OpenAI compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for API clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Model cache (single-process) -------------------------------------


class _GenCache:
    def __init__(self) -> None:
        self._key: Optional[Tuple[str, bool]] = None
        self._gen: Optional[HFGenerator] = None
        self._loaded: bool = False

    def get_with_config(self, cfg: GenerationConfig) -> HFGenerator:
        # Cache keyed by (model_name, trust_remote_code)
        key = (cfg.model_name, cfg.trust_remote_code)
        if self._gen is None or self._key != key:
            self._gen = HFGenerator(cfg)
            self._key = key
            self._loaded = True
        return self._gen

    def preload(self, cfg: GenerationConfig) -> None:
        self.get_with_config(cfg)

    def is_loaded(self) -> bool:
        return self._loaded


_GEN_CACHE = _GenCache()

# Preload model on startup
@app.on_event("startup")
def _startup_preload() -> None:
    try:
        flag = os.getenv("LABS_PRELOAD_ON_START", "true").strip().lower()
        if flag in {"1", "true", "yes", "on"}:
            cfg = load_config(None)
            _GEN_CACHE.preload(cfg)
    except Exception:
        # Fail open: API can still serve requests with lazy load
        pass


def _build_generator(req: ChatCompletionRequest) -> HFGenerator:
    """Build generator from request"""
    cfg = load_config(None)
    
    # Use exact model name from request
    cfg.model_name = req.model
    
    # Apply request parameters
    if req.max_tokens is not None:
        cfg.max_new_tokens = req.max_tokens
    if req.temperature is not None:
        cfg.temperature = req.temperature
    if req.top_p is not None:
        cfg.top_p = req.top_p
    if req.top_k is not None:
        cfg.top_k = req.top_k
    
    # Map frequency_penalty to repetition_penalty (approximate)
    if req.frequency_penalty is not None and req.frequency_penalty != 0.0:
        cfg.repetition_penalty = 1.0 + req.frequency_penalty
    
    # Trust remote code
    if req.trust_remote_code is not None:
        cfg.trust_remote_code = req.trust_remote_code
    
    return _GEN_CACHE.get_with_config(cfg)


def _convert_messages_to_chat_format(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert OpenAI messages to HuggingFace chat format"""
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token average)"""
    return max(1, len(text) // 4)


# OpenAI-compatible endpoints

@app.get("/v1/models")
def list_models() -> ModelListResponse:
    """List available models"""
    cfg = load_config(None)
    current_time = int(time.time())
    
    # Return only the currently configured model
    models = [
        ModelInfo(id=cfg.model_name, created=current_time),
    ]
    
    return ModelListResponse(data=models)


@app.post("/v1/chat/completions", response_model=None)
def create_chat_completion(req: ChatCompletionRequest):
    """Create a chat completion (OpenAI-compatible)"""
    if req.stream:
        return _create_chat_completion_stream(req)
    else:
        return _create_chat_completion_sync(req)


def _create_chat_completion_sync(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """Non-streaming chat completion"""
    try:
        gen = _build_generator(req)
        messages = _convert_messages_to_chat_format(req.messages)
        
        # Generate response
        response_text = gen.generate(
            messages,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            do_sample=req.temperature > 0,
        )
        
        # Create response
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_time = int(time.time())
        
        # Estimate token usage
        prompt_text = " ".join([msg.content for msg in req.messages])
        prompt_tokens = _estimate_tokens(prompt_text)
        completion_tokens = _estimate_tokens(response_text)
        
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response_text),
            finish_reason="stop"
        )
        
        usage = ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        return ChatCompletionResponse(
            id=completion_id,
            created=current_time,
            model=req.model,
            choices=[choice],
            usage=usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e


def _create_chat_completion_stream(req: ChatCompletionRequest) -> StreamingResponse:
    """Streaming chat completion"""
    try:
        gen = _build_generator(req)
        messages = _convert_messages_to_chat_format(req.messages)
        
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_time = int(time.time())
        
        def event_stream() -> Generator[bytes, None, None]:
            try:
                # Send initial chunk with role
                initial_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=current_time,
                    model=req.model,
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta={"role": "assistant"},
                        finish_reason=None
                    )]
                )
                yield f"data: {initial_chunk.model_dump_json()}\n\n".encode("utf-8")
                
                # Stream content
                for chunk in gen.stream_generate(
                    messages,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    do_sample=req.temperature > 0,
                ):
                    stream_chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=current_time,
                        model=req.model,
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": chunk},
                            finish_reason=None
                        )]
                    )
                    yield f"data: {stream_chunk.model_dump_json()}\n\n".encode("utf-8")
                
                # Send final chunk
                final_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=current_time,
                    model=req.model,
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta={},
                        finish_reason="stop"
                    )]
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n".encode("utf-8")
                
                # Send [DONE]
                yield b"data: [DONE]\n\n"
                
            except Exception as e:
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "internal_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {e}") from e


# Health and compatibility endpoints
@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/v1/health")
def health_v1() -> Dict[str, str]:
    """OpenAI-style health check"""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Dict[str, bool]:
    """Readiness probe"""
    return {"loaded": _GEN_CACHE.is_loaded()}


# Embedding endpoints
@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings (OpenAI-compatible)"""
    try:
        embedding_gen = get_embedding_generator()
        
        # Handle single string or list of strings
        if isinstance(req.input, str):
            texts = [req.input]
        else:
            texts = req.input
        
        # Generate embeddings
        embeddings = embedding_gen.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        
        # Convert to list format for JSON response
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append(EmbeddingData(
                object="embedding",
                embedding=embedding.tolist(),
                index=i
            ))
        
        # Estimate token usage
        total_text = " ".join(texts)
        prompt_tokens = _estimate_tokens(total_text)
        
        usage = EmbeddingUsage(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens
        )
        
        return EmbeddingResponse(
            object="list",
            data=embedding_data,
            model=req.model,
            usage=usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}") from e


@app.post("/v1/similarities")
def compute_similarities(req: SimilarityRequest) -> SimilarityResponse:
    """Compute pairwise similarities between sentences"""
    try:
        embedding_gen = get_embedding_generator()
        
        # Generate embeddings and compute similarities
        embeddings, similarities = embedding_gen.encode_and_similarity(
            req.sentences,
            convert_to_numpy=True,
        )
        
        # Convert to list format for JSON response
        similarities_list = similarities.tolist()
        shape = list(similarities.shape)
        
        return SimilarityResponse(
            similarities=similarities_list,
            model=req.model,
            shape=shape
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {e}") from e


# TTS request/response models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    model: Optional[str] = Field(default="suno/bark", description="TTS model to use")

# Add TTS endpoint
@app.post("/v1/tts")
def text_to_speech(request: TTSRequest):
    """Generate speech audio from text input."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        tts = get_tts_instance()
        audio_bytes = tts.synthesize(request.text)
        # Return audio as WAV bytes with appropriate headers
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}") from e


def main() -> int:
    """
    Optional programmatic runner for uvicorn:
      uv run labs-api
    """
    try:
        import uvicorn
    except Exception:
        print("uvicorn is not installed. Install with: uv add uvicorn[standard]", flush=True)
        return 2
    
    # Get host and port from environment variables or use defaults
    host = os.getenv("LABS_HOST", "0.0.0.0")
    port = int(os.getenv("LABS_PORT", "8000"))
    
    print(f"Starting Labs API server on {host}:{port}")
    print(f"Server will be accessible at:")
    print(f"  - Local: http://localhost:{port}")
    print(f"  - Network: http://{host}:{port}")
    if host == "0.0.0.0":
        print(f"  - LAN: http://<your-ip-address>:{port}")
    print(f"  - Health check: http://{host}:{port}/health")
    print(f"  - OpenAI models: http://{host}:{port}/v1/models")
    
    uvicorn.run("labs.api:app", host=host, port=port, reload=True)
    return 0


__all__ = ["app", "main"]
