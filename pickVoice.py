import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import ChatTTS

# -------------------------
# CONFIG
# -------------------------
OUT_DIR = Path("voicess")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT = "Yeah, no, this is my backyard. It's never ending so just the way I like it. So social distancing has never been a problem. [lbreak]"
N_VOICES = 100               # <-- choose how many voices to sample
BATCH_SIZE = 8               # speed: generate in small batches
SEED = 1234                  # reproducibility
SR = 24000                   # ChatTTS commonly outputs 24kHz

# Inference style params (tweak if you want)
TEMPERATURE = 0.3
TOP_P = 0.7
TOP_K = 20

# -------------------------
# SET SEEDS (optional but helpful)
# -------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# LOAD ChatTTS
# -------------------------
chat = ChatTTS.Chat()
chat.load(compile=False)

# -------------------------
# Audio saver (robust)
# -------------------------
def save_wav(path: Path, wav: np.ndarray, sr: int = SR):
    """
    wav: float32 numpy array in [-1, 1] ideally
    """
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)

    try:
        import soundfile as sf
        sf.write(str(path), wav, sr)
    except Exception:
        # fallback to scipy
        from scipy.io.wavfile import write as wavwrite
        int16 = (wav * 32767.0).astype(np.int16)
        wavwrite(str(path), sr, int16)

def pad_tail(wav: np.ndarray, sr: int = SR, sec: float = 0.25) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    return np.concatenate([wav, np.zeros(int(sec * sr), dtype=np.float32)])

# -------------------------
# 1) Sample voice embeddings
# -------------------------
speaker_bank = {}
for i in range(N_VOICES):
    vid = f"v{i:04d}"
    speaker_bank[vid] = chat.sample_random_speaker()

# Save the full bank so you can reuse voices later
torch.save(speaker_bank, OUT_DIR / "speaker_bank.pt")

# Save metadata for traceability
meta = {
    "text": TEXT,
    "n_voices": N_VOICES,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "sr": SR,
    "infer_params": {
        "temperature": TEMPERATURE,
        "top_P": TOP_P,
        "top_K": TOP_K
    }
}
(OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# -------------------------
# 2) Generate WAVs in batches
# -------------------------
voice_ids = list(speaker_bank.keys())

for start in tqdm(range(0, len(voice_ids), BATCH_SIZE), desc="Generating voice samples"):
    batch_ids = voice_ids[start:start + BATCH_SIZE]

    for vid in batch_ids:
        params = ChatTTS.Chat.InferCodeParams(
            spk_emb=speaker_bank[vid],
            temperature=TEMPERATURE,
            top_P=TOP_P,
            top_K=TOP_K,
            # Optional: if you still see early-stops, enable one of these if supported by your install:
            # min_new_token=32,
            # ensure_non_empty=True,
        )
        wavs = chat.infer([TEXT], params_infer_code=params)
        wav = wavs[0]

        # Key: add short silence so nothing clips
        wav = pad_tail(wav, SR, 0.25)

        save_wav(OUT_DIR / f"{vid}.wav", wav, SR)

print(f"Done. Wrote {N_VOICES} WAVs to: {OUT_DIR.resolve()}")
print(f"Speaker embeddings saved to: {(OUT_DIR / 'speaker_bank.pt').resolve()}")
