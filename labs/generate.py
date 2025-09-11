from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig


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

    # Optional quantization (BitsAndBytes). Only one may be True.
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: Optional[torch.dtype] = None
    bnb_4bit_use_double_quant: bool = True


class HFGenerator:
    """
    Minimal Hugging Face Transformers generator with sensible defaults for
    CUDA GPUs (BF16/FP16) and device_map='auto'. No quantization by default.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        # Resolve dtype
        self.torch_dtype = self._resolve_dtype(self.config.torch_dtype)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        # Ensure pad token exists (fallback to EOS)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (accelerate handles placement for device_map='auto')
        quant_config = self._build_quantization_config()
        model_kwargs: Dict[str, Any] = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            # transformers >=4.56 deprecates torch_dtype in favor of dtype
            model_kwargs["dtype"] = self.torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        # Cache generation token IDs
        self.eos_token_id = self.config.eos_token_id or self.tokenizer.eos_token_id
        self.pad_token_id = self.config.pad_token_id or self.tokenizer.pad_token_id

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

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Build BitsAndBytes quantization config if requested via config.
        Raises a RuntimeError with guidance if bitsandbytes is not installed.
        """
        if not (self.config.load_in_4bit or self.config.load_in_8bit):
            return None
        if self.config.load_in_4bit and self.config.load_in_8bit:
            raise ValueError("Only one of load_in_4bit or load_in_8bit may be True.")

        try:
            import bitsandbytes as bnb  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Quantization requested but bitsandbytes is not installed. "
                "Install with: uv add 'bitsandbytes>=0.43.0' or pip install bitsandbytes"
            ) from e

        if self.config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)

        # 4-bit defaults: prefer BF16 compute if supported, else FP16
        compute_dtype = self.config.bnb_4bit_compute_dtype
        if compute_dtype is None:
            if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float16

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

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
        inputs = self._build_inputs(prompt_or_messages)
        inputs = self._maybe_move_inputs_to_model_device(inputs)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": top_p if top_p is not None else self.config.top_p,
            "do_sample": do_sample if do_sample is not None else self.config.do_sample,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        if top_k is not None or self.config.top_k is not None:
            gen_kwargs["top_k"] = top_k if top_k is not None else self.config.top_k
        if repetition_penalty is not None or self.config.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = (
                repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
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
        inputs = self._build_inputs(prompt_or_messages)
        inputs = self._maybe_move_inputs_to_model_device(inputs)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": top_p if top_p is not None else self.config.top_p,
            "do_sample": do_sample if do_sample is not None else self.config.do_sample,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        if top_k is not None or self.config.top_k is not None:
            gen_kwargs["top_k"] = top_k if top_k is not None else self.config.top_k
        if repetition_penalty is not None or self.config.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = (
                repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
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
