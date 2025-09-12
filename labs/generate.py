from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .model_loader import load_model_and_tokenizer

try:
    from .rag_qa import TransactionRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    Defaults chosen for NVIDIA GPUs with ≥16GB VRAM:
      - device_map='auto'
      - dtype: BF16 if supported, else FP16; CPU falls back to FP32
      - chat template enabled by default for instruct/chat models
    """
    model_name: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    do_sample: bool = True
    repetition_penalty: Optional[float] = None

    # Loading/runtime options
    device_map: str = "auto"
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = False

    # Chat formatting
    use_chat_template: bool = True
    add_generation_prompt: bool = True

    # Token IDs (fall back to tokenizer defaults if not provided)
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    


class HFGenerator:
    """
    Minimal Hugging Face Transformers generator with sensible defaults for
    CUDA GPUs (BF16/FP16) and device_map='auto'. No quantization by default.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        # Resolve dtype
        self.torch_dtype = self._resolve_dtype(self.config.torch_dtype)

        # Build model loading kwargs
        model_kwargs: Dict[str, Any] = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            # transformers >=4.56 deprecates torch_dtype in favor of dtype
            "dtype": self.torch_dtype
        }

        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            **model_kwargs
        )

        # Cache generation token IDs
        self.eos_token_id = self.config.eos_token_id or self.tokenizer.eos_token_id
        self.pad_token_id = self.config.pad_token_id or self.tokenizer.pad_token_id
        
        # Initialize RAG system for transaction queries
        self.transaction_rag = None
        if RAG_AVAILABLE:
            try:
                self.transaction_rag = TransactionRAG()
                print("✅ Transaction RAG system initialized")
            except Exception as e:
                print(f"⚠️  Transaction RAG not available: {e}")

    def _is_transaction_question(self, text: str) -> bool:
        """Check if a question is about transactions."""
        if not self.transaction_rag:
            return False
            
        transaction_keywords = [
            'expense', 'spend', 'spent', 'cost', 'price', 'pay', 'paid',
            'income', 'earn', 'earned', 'revenue', 'profit',
            'transaction', 'purchase', 'buy', 'bought',
            'total', 'largest', 'biggest', 'most expensive',
            'recent', 'last', 'latest',
            'summary', 'overview', 'breakdown',
            'december', 'november', 'january', 'february', 'march',
            'april', 'may', 'june', 'july', 'august', 'september', 'october',
            'software', 'hardware', 'fuel', 'office', 'vehicle',
            'microsoft', 'adobe', 'ikea', 'starbucks', 'netflix'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in transaction_keywords)
    
    def _extract_user_message(self, prompt_or_messages: Union[str, List[Dict[str, str]]]) -> str:
        """Extract the user message for transaction detection."""
        if isinstance(prompt_or_messages, str):
            return prompt_or_messages
        
        # Find the last user message
        for msg in reversed(prompt_or_messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        
        return ""

    def _resolve_dtype(self, explicit: Optional[torch.dtype]) -> torch.dtype:
        if explicit is not None:
            return explicit
        if torch.cuda.is_available():
            try:
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                # Fall back to FP16 if BF16 probe fails
                pass
            return torch.float16
        return torch.float32


    def _build_inputs(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """
        Prepare model inputs. With device_map='auto', keep tensors on CPU; accelerate/vLLM/TGI
        will handle placement. Do NOT call .to(model.device) in this mode.
        """
        if isinstance(prompt_or_messages, str) or not self.config.use_chat_template:
            # Raw text path
            text = prompt_or_messages if isinstance(prompt_or_messages, str) else str(prompt_or_messages)
            return self.tokenizer(text, return_tensors="pt")

        # Chat messages path
        messages = prompt_or_messages  # type: ignore[assignment]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=self.config.add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _get_model_device(self) -> torch.device:
        """Get the primary device where the model is located."""
        try:
            # Find the first non-meta parameter device
            for param in self.model.parameters():
                if param.device.type != "meta":
                    return param.device
        except Exception:
            pass
        return torch.device("cpu")

    def _maybe_move_inputs_to_model_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move inputs to the same device as the model to avoid device mismatch warnings.
        """
        try:
            model_device = self._get_model_device()
            
            # Check if we need to move inputs
            if "input_ids" in inputs:
                input_device = inputs["input_ids"].device
                if input_device != model_device:
                    # Move all tensor inputs to model device
                    moved: Dict[str, Any] = {}
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            moved[k] = v.to(model_device)
                        else:
                            moved[k] = v
                    return moved
            
            return inputs
        except Exception:
            # If anything fails, return original inputs
            return inputs

    def _build_generation_kwargs(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build generation kwargs from method parameters and config defaults.
        This centralizes the parameter resolution logic used by both generate() and stream_generate().
        """
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": top_p if top_p is not None else self.config.top_p,
            "do_sample": do_sample if do_sample is not None else self.config.do_sample,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        
        # Only add top_k if specified (either in params or config)
        if top_k is not None or self.config.top_k is not None:
            gen_kwargs["top_k"] = top_k if top_k is not None else self.config.top_k
            
        # Only add repetition_penalty if specified (either in params or config)
        if repetition_penalty is not None or self.config.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = (
                repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
            )
            
        return gen_kwargs

    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Single-shot text generation. Returns only newly generated text
        (excludes the prompt portion).
        """
        # Check if this is a transaction question and use RAG if available
        user_message = self._extract_user_message(prompt_or_messages)
        if self._is_transaction_question(user_message):
            try:
                rag_answer = self.transaction_rag.answer_question(user_message)
                # If RAG gives a useful answer (not the fallback message), use it
                if not rag_answer.startswith("I can answer questions"):
                    return rag_answer
            except Exception as e:
                print(f"⚠️  RAG failed, falling back to LLM: {e}")
        
        # Fall back to normal LLM generation
        inputs = self._build_inputs(prompt_or_messages)
        inputs = self._maybe_move_inputs_to_model_device(inputs)

        gen_kwargs = self._build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated tokens (exclude the prompt)
        input_len = inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return text

    def stream_generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
    ):
        """
        Streaming text generation. Yields incremental text chunks as they arrive.
        """
        # Check if this is a transaction question and simulate streaming for RAG
        user_message = self._extract_user_message(prompt_or_messages)
        if self._is_transaction_question(user_message):
            try:
                rag_answer = self.transaction_rag.answer_question(user_message)
                # If RAG gives a useful answer, simulate streaming
                if not rag_answer.startswith("I can answer questions"):
                    words = rag_answer.split()
                    for word in words:
                        yield word + " "
                    return
            except Exception as e:
                print(f"⚠️  RAG failed, falling back to LLM: {e}")
        
        # Fall back to normal streaming generation
        inputs = self._build_inputs(prompt_or_messages)
        inputs = self._maybe_move_inputs_to_model_device(inputs)

        gen_kwargs = self._build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
        thread.start()

        for new_text in streamer:
            yield new_text


__all__ = ["GenerationConfig", "HFGenerator"]
