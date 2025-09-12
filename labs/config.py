from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Python 3.12+ has tomllib in the standard library
import tomllib
from dotenv import load_dotenv

# Import after stdlib to avoid any circular import surprises at module import time
from labs.generate import GenerationConfig


DEFAULT_CONFIG_PATHS = [
    Path.cwd() / "labs.toml",
    Path.cwd() / "config" / "labs.toml",
]


def _coerce_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    name = name.strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    # Unknown string: ignore
    return None


def _bool_env(name: str, default: Optional[bool] = None) -> Optional[bool]:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _int_env(name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default


def _float_env(name: str, default: Optional[float] = None) -> Optional[float]:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default


def _find_config_path(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    env_p = os.getenv("LABS_CONFIG")
    if env_p:
        p = Path(env_p).expanduser().resolve()
        if p.exists():
            return p
    for p in DEFAULT_CONFIG_PATHS:
        if p.exists():
            return p
    return None


def _merge_generation_table(cfg: GenerationConfig, tbl: Dict[str, Any]) -> None:
    # Basic generation and runtime options
    if "model_name" in tbl:
        cfg.model_name = str(tbl["model_name"])
    if "max_new_tokens" in tbl:
        cfg.max_new_tokens = int(tbl["max_new_tokens"])
    if "temperature" in tbl:
        cfg.temperature = float(tbl["temperature"])
    if "top_p" in tbl:
        cfg.top_p = float(tbl["top_p"])
    if "top_k" in tbl and tbl["top_k"] is not None:
        cfg.top_k = int(tbl["top_k"])
    if "do_sample" in tbl:
        cfg.do_sample = bool(tbl["do_sample"])
    if "repetition_penalty" in tbl and tbl["repetition_penalty"] is not None:
        cfg.repetition_penalty = float(tbl["repetition_penalty"])

    if "device_map" in tbl:
        cfg.device_map = str(tbl["device_map"])
    if "torch_dtype" in tbl:
        dt = _coerce_dtype(str(tbl["torch_dtype"]))
        if dt is not None:
            cfg.torch_dtype = dt
    if "trust_remote_code" in tbl:
        cfg.trust_remote_code = bool(tbl["trust_remote_code"])

    if "use_chat_template" in tbl:
        cfg.use_chat_template = bool(tbl["use_chat_template"])
    if "add_generation_prompt" in tbl:
        cfg.add_generation_prompt = bool(tbl["add_generation_prompt"])




def _apply_env_overrides(cfg: GenerationConfig) -> None:
    # Minimal env override set; can be extended as needed
    model = os.getenv("LABS_MODEL")
    if model:
        cfg.model_name = model
    mnt = _int_env("LABS_MAX_NEW_TOKENS")
    if mnt is not None:
        cfg.max_new_tokens = mnt
    temp = _float_env("LABS_TEMPERATURE")
    if temp is not None:
        cfg.temperature = temp
    top_p = _float_env("LABS_TOP_P")
    if top_p is not None:
        cfg.top_p = top_p
    top_k = _int_env("LABS_TOP_K")
    if top_k is not None:
        cfg.top_k = top_k
    do_sample = _bool_env("LABS_DO_SAMPLE")
    if do_sample is not None:
        cfg.do_sample = do_sample
    rep = _float_env("LABS_REPETITION_PENALTY")
    if rep is not None:
        cfg.repetition_penalty = rep

    # Runtime
    device_map = os.getenv("LABS_DEVICE_MAP")
    if device_map:
        cfg.device_map = device_map
    torch_dtype = _coerce_dtype(os.getenv("LABS_TORCH_DTYPE"))
    if torch_dtype is not None:
        cfg.torch_dtype = torch_dtype
    trust_rc = _bool_env("LABS_TRUST_REMOTE_CODE")
    if trust_rc is not None:
        cfg.trust_remote_code = trust_rc
    

    # Chat
    chat_tpl = _bool_env("LABS_USE_CHAT_TEMPLATE")
    if chat_tpl is not None:
        cfg.use_chat_template = chat_tpl
    gen_prompt = _bool_env("LABS_ADD_GENERATION_PROMPT")
    if gen_prompt is not None:
        cfg.add_generation_prompt = gen_prompt



def load_config(path: Optional[str] = None) -> GenerationConfig:
    """
    Load GenerationConfig from TOML and environment variables.

    Priority:
      1) Explicit path argument
      2) LABS_CONFIG env var
      3) Default paths: ./labs.toml or ./config/labs.toml
      4) If no file is found, return defaults overridden by env vars

    Additionally, if a .env file is present in the working directory, it is loaded
    (without overriding existing environment variables).
    """
    # Load .env if present; do not override existing env vars
    try:
        load_dotenv(override=False)
    except Exception:
        pass

    # Default model name - can be overridden by config file or env vars
    default_model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # Check for model name from environment first
    model_name = os.getenv("LABS_MODEL", default_model_name)
    
    p = _find_config_path(path)
    
    # Load config data
    data: Dict[str, Any] = {}
    if p is not None:
        try:
            with open(p, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            pass

    # Check generation table for model name
    gen_tbl = data.get("generation") or data.get("model") or {}
    if isinstance(gen_tbl, dict) and "model_name" in gen_tbl:
        model_name = str(gen_tbl["model_name"])

    # Check environment override (highest priority)
    model_name = os.getenv("LABS_MODEL", model_name)
    
    # Create config with determined model name
    cfg = GenerationConfig(model_name=model_name)

    # Merge tables from config file
    if isinstance(gen_tbl, dict):
        _merge_generation_table(cfg, gen_tbl)


    # Environment overrides last
    _apply_env_overrides(cfg)
    return cfg


def dump_effective_config(cfg: Optional[GenerationConfig] = None) -> Dict[str, Any]:
    """
    Return a JSON-serializable dict of the effective configuration.
    """
    if cfg is None:
        cfg = load_config()
    return asdict(cfg)


__all__ = ["load_config", "dump_effective_config"]
