from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import soundfile as sf
import io

class TTS:
    def __init__(self, model_name: str = "microsoft/VibeVoice-1.5B", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def synthesize(self, text: str) -> bytes:
        """Generate speech audio bytes from input text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        audio = outputs[0].cpu().numpy()
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate=24000, format="WAV")
        return buffer.getvalue()


def get_tts_instance() -> TTS:
    """Singleton TTS instance."""
    global _tts_instance
    if '_tts_instance' not in globals():
        _tts_instance = TTS()
    return _tts_instance


__all__ = ["TTS", "get_tts_instance"]

