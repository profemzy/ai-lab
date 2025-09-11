import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Union

import torch

from labs import GenerationConfig, HFGenerator
from labs.config import load_config


def _parse_messages(raw: str) -> List[Dict[str, str]]:
    """
    Accept either:
      - Inline JSON: '[{"role":"user","content":"Hello"}]'
      - File reference prefixed with '@': '@/path/to/messages.json'
    """
    if raw.startswith("@"):
        path = raw[1:]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="labs-gen",
        description="Generate text using a Hugging Face model with sensible GPU defaults."
    )
    # Input
    p.add_argument("--prompt", type=str, default=None, help="Raw text prompt (non-chat path).")
    p.add_argument(
        "--messages-json",
        type=str,
        default=None,
        help="Chat messages as JSON string or file reference prefixed with '@'. "
             "Example: '[{\"role\":\"user\",\"content\":\"Hello\"}]' or '@/path/to/messages.json'"
    )

    # Model/config
    p.add_argument("--model", type=str, default=None, help="Model name or path (default from config).")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Maximum new tokens to generate.")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    p.add_argument("--top-p", type=float, default=None, help="Nucleus sampling p.")
    p.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    p.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty factor.")
    p.add_argument("--no-sample", action="store_true", help="Disable sampling (deterministic decoding).")

    # Streaming
    p.add_argument("--stream", action="store_true", help="Stream tokens to stdout as they are generated.")

    # Chat/template toggles
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template path even if messages are provided."
    )
    p.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not add generation prompt token(s) in chat template."
    )

    # Trust remote code (for custom models)
    p.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for model/tokenizer.")

    # Quantization (optional)
    p.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization (bitsandbytes).")
    p.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit quantization (bitsandbytes).")
    p.add_argument(
        "--bnb-4bit-compute-dtype",
        type=str,
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        default=None,
        help="Compute dtype for 4-bit quantization."
    )
    p.add_argument(
        "--bnb-4bit-quant-type",
        type=str,
        default=None,
        help='Quantization type for 4-bit (e.g., "nf4").'
    )
    p.add_argument(
        "--bnb-4bit-use-double-quant",
        action="store_true",
        help="Enable double quantization for 4-bit."
    )

    # Config file
    p.add_argument("--config", type=str, default=None, help="Path to labs.toml for defaults (overridden by env/CLI).")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.prompt and not args.messages_json:
        print("Error: Provide either --prompt or --messages-json", file=sys.stderr)
        return 2

    # Prepare config with sensible GPU defaults from core module
    cfg = load_config(args.config)
    if args.model:
        cfg.model_name = args.model
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.repetition_penalty is not None:
        cfg.repetition_penalty = args.repetition_penalty
    if args.no_chat_template:
        cfg.use_chat_template = False
    if args.no_generation_prompt:
        cfg.add_generation_prompt = False
    if args.trust_remote_code:
        cfg.trust_remote_code = True

    # Quantization flags (CLI overrides config)
    if args.load_in_4bit:
        cfg.load_in_4bit = True
    if args.load_in_8bit:
        cfg.load_in_8bit = True
    if args.bnb_4bit_quant_type is not None:
        cfg.bnb_4bit_quant_type = args.bnb_4bit_quant_type
    if args.bnb_4bit_use_double_quant:
        cfg.bnb_4bit_use_double_quant = True
    if args.bnb_4bit_compute_dtype is not None:
        dt_map = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32,
        }
        cfg.bnb_4bit_compute_dtype = dt_map.get(args.bnb_4bit_compute_dtype)

    gen = HFGenerator(cfg)

    # Build input
    prompt_or_messages: Union[str, List[Dict[str, str]]]
    if args.messages_json and not args.no_chat_template:
        try:
            prompt_or_messages = _parse_messages(args.messages_json)
        except Exception as e:
            print(f"Error parsing messages JSON: {e}", file=sys.stderr)
            return 2
    elif args.prompt:
        prompt_or_messages = args.prompt
    else:
        # If messages provided but user disabled chat template, coerce to string
        try:
            parsed = _parse_messages(args.messages_json)  # type: ignore[arg-type]
            prompt_or_messages = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            prompt_or_messages = str(args.messages_json)

    # Generate
    if args.stream:
        for chunk in gen.stream_generate(
            prompt_or_messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=(not args.no_sample),
            repetition_penalty=args.repetition_penalty,
        ):
            print(chunk, end="", flush=True)
        print()
        return 0
    else:
        text = gen.generate(
            prompt_or_messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=(not args.no_sample),
            repetition_penalty=args.repetition_penalty,
        )

        # Output to stdout
        print(text)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
