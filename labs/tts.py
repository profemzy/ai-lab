from __future__ import annotations

import io
import wave
import numpy as np
import torch
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile


class TTS:
    def __init__(self, model_name: str = "suno/bark-small"):
        # Use proper Bark implementation with BarkModel
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = BarkModel.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Default voice preset for consistent quality
        self.voice_preset = "v2/en_speaker_6"
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def synthesize(self, text: str) -> bytes:
        """Generate speech audio as WAV bytes from text using Bark model."""
        if not text or not text.strip():
            raise ValueError("Text for TTS is empty")

        # Process text using Bark processor with voice preset
        inputs = self.processor(text, voice_preset=self.voice_preset)
        
        # Move inputs to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate speech using semantic_max_new_tokens parameter
        with torch.no_grad():
            audio_array = self.model.generate(**inputs, semantic_max_new_tokens=100)

        # Convert to numpy as shown in the example
        audio_array = audio_array.cpu().numpy().squeeze()

        # Bark outputs at 24kHz sample rate
        sample_rate = 24000

        # Normalize audio to [-1, 1] range if needed
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to 16-bit PCM
        pcm16 = (audio_array * 32767.0).astype(np.int16)

        # Write WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()


def get_tts_instance() -> TTS:
    """Singleton TTS instance."""
    global _tts_instance
    if "_tts_instance" not in globals():
        _tts_instance = TTS()
    return _tts_instance


__all__ = ["TTS", "get_tts_instance"]

