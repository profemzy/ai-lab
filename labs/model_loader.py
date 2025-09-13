"""
Model loader for base models.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
import logging
import warnings
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress all transformers warnings and progress bars
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable tqdm progress bars  
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Suppress specific loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)



def load_model_and_tokenizer(
    model_name_or_path: str,
    **kwargs
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer.
    
    Args:
        model_name_or_path: Model name or path
        **kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **kwargs
    )
    
    
    return model, tokenizer




def get_available_models() -> dict:
    """
    Get list of available models including base and fine-tuned versions.
    
    Returns:
        Dictionary of model names and their paths/descriptions
    """
    models = {
        "base": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "description": "Base Qwen2.5 7B Instruct model",
            "type": "base"
        }
    }
    
    
    return models