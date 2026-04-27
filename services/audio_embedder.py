import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor

MODEL_NAME = "laion/clap-htsat-unfused"

# Kept as module-level variables so the model loads once and stays in memory
_model = None
_processor = None


def _load_model():
    # Only loads the CLAP model on first call — reuses it for every subsequent embedding
    global _model, _processor
    if _model is None:
        print("[AudioEmbedder] Loading CLAP model...")
        _model = ClapModel.from_pretrained(MODEL_NAME)
        _processor = ClapProcessor.from_pretrained(MODEL_NAME)
        _model.eval()
        print("[AudioEmbedder] CLAP model loaded.")
    return _model, _processor


def _extract_tensor(output):
    # CLAP can return either a raw tensor or a ModelOutput object depending on version
    # This handles both cases safely
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise ValueError(f"Cannot extract tensor from output type: {type(output)}")


def _to_unit_vector(tensor: torch.Tensor) -> np.ndarray:
    # Collapses any shape down to a single 1D vector and normalises it to unit length
    # Normalising means dot product equals cosine similarity during search
    t = tensor
    if t.dim() == 3:
        t = t.mean(dim=1)
    if t.dim() == 2:
        t = t[0]

    vec = t.cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_audio_file(file_path: str) -> np.ndarray | None:
    # Loads up to 30s of audio and returns a 512-d vector representing its content
    # Returns None if the file is too short or fails — worker will skip it
    try:
        model, processor = _load_model()
        audio, _ = librosa.load(file_path, sr=48000, mono=True, duration=30.0)

        if len(audio) < 1000:
            print(f"[AudioEmbedder] File too short, skipping: {file_path}")
            return None

        inputs = processor(audio=audio, return_tensors="pt", sampling_rate=48000)

        with torch.no_grad():
            output = model.get_audio_features(**inputs)

        return _to_unit_vector(_extract_tensor(output))

    except Exception as e:
        print(f"[AudioEmbedder] Failed to embed {file_path}: {e}")
        return None


def embed_text_query(query: str) -> np.ndarray | None:
    # Encodes a text search query into the same 512-d space as the audio embeddings
    # This is what makes text-to-audio comparison possible
    try:
        model, processor = _load_model()
        inputs = processor(text=[query], return_tensors="pt", padding=True)

        with torch.no_grad():
            output = model.get_text_features(**inputs)

        return _to_unit_vector(_extract_tensor(output))

    except Exception as e:
        print(f"[AudioEmbedder] Failed to embed query '{query}': {e}")
        return None