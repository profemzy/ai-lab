"""
Embedding functionality using sentence-transformers.
Provides text embedding and similarity computation capabilities.
"""

from __future__ import annotations

import os
from typing import List, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingConfig:
    """Configuration for embedding models."""
    
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        # Handle "auto" device mapping like transformers
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code


class EmbeddingGenerator:
    """Text embedding generator using sentence-transformers."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model: Optional[SentenceTransformer] = None
        
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is not None:
            return
            
        print(f"Loading embedding model: {self.config.model_name}")
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
            trust_remote_code=self.config.trust_remote_code,
        )
        print(f"Model loaded on device: {self.model.device}")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: Single sentence or list of sentences to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert result to numpy array
            normalize_embeddings: Whether to normalize embeddings to unit length
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        if self.model is None:
            self.load_model()
            
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )
    
    def similarity(
        self,
        embeddings1: Union[np.ndarray, torch.Tensor],
        embeddings2: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        if self.model is None:
            self.load_model()
            
        return self.model.similarity(embeddings1, embeddings2)
    
    def encode_and_similarity(
        self,
        sentences: Union[str, List[str]],
        **encode_kwargs,
    ) -> tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Encode sentences and compute pairwise similarities.
        
        Args:
            sentences: Sentences to encode
            **encode_kwargs: Additional arguments for encoding
            
        Returns:
            Tuple of (embeddings, similarity_matrix)
        """
        embeddings = self.encode(sentences, **encode_kwargs)
        similarities = self.similarity(embeddings, embeddings)
        return embeddings, similarities


def load_embedding_config() -> EmbeddingConfig:
    """Load embedding configuration from environment variables."""
    model_name = os.getenv("LABS_EMBEDDING_MODEL", "google/embeddinggemma-300m")
    device = os.getenv("LABS_EMBEDDING_DEVICE")
    trust_remote_code = os.getenv("LABS_TRUST_REMOTE_CODE", "false").lower() == "true"
    
    return EmbeddingConfig(
        model_name=model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )


# Global embedding generator instance
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        config = load_embedding_config()
        _embedding_generator = EmbeddingGenerator(config)
    return _embedding_generator


__all__ = [
    "EmbeddingConfig",
    "EmbeddingGenerator", 
    "load_embedding_config",
    "get_embedding_generator",
]
