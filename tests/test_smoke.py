import os
import time

import pytest

from labs import HFGenerator, GenerationConfig


@pytest.mark.timeout(120)
def test_generate_tiny_model_non_streaming():
    """
    Smoke test: load a tiny causal LM and generate a short completion.
    Uses a tiny model to minimize download size and memory usage.
    """
    cfg = GenerationConfig(
        model_name="sshleifer/tiny-gpt2",
        max_new_tokens=8,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        # Ensure CPU-safe defaults for CI without GPU
        device_map="auto",
        torch_dtype=None,  # let core select (will be fp32 on CPU)
        use_chat_template=False,  # tiny-gpt2 isn't chat-tuned; use raw prompt
    )
    gen = HFGenerator(cfg)
    out = gen.generate("Hello from a tiny model. Continue:")
    assert isinstance(out, str)
    # Non-empty or at least whitespace separated tokens
    assert len(out.strip()) > 0


@pytest.mark.timeout(120)
def test_generate_tiny_model_streaming():
    """
    Smoke test: streaming generation via the core stream_generate API.
    """
    cfg = GenerationConfig(
        model_name="sshleifer/tiny-gpt2",
        max_new_tokens=8,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        device_map="auto",
        torch_dtype=None,
        use_chat_template=False,
    )
    gen = HFGenerator(cfg)
    chunks = []
    start = time.time()
    for chunk in gen.stream_generate("Streaming test:"):
        # accumulate first few chunks then stop to keep test fast
        chunks.append(chunk)
        if len("".join(chunks)) > 0:
            break
        # Safety to avoid infinite loop if streamer yields nothing
        if time.time() - start > 60:
            break
    combined = "".join(chunks).strip()
    assert isinstance(combined, str)
    assert len(combined) > 0