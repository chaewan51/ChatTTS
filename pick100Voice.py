import os
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import soundfile as sf
import ChatTTS

# -------------------------
# CONFIG (Matched to testing3.py)
# -------------------------
OUT_DIR = Path("voicess_audition_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Performance
torch.set_float32_matmul_precision('high')

# Audio Params
SR = 24000
END_MARK = " [uv_break]" # Identical to testing3.py

# Inference Params (Identical to testing3.py)
TEMP = 0.15
TOP_P = 0.70
TOP_K = 20
MAX_NEW = 2048
MIN_NEW_LONG = 48
MIN_NEW_SHORT = 0
REPETITION_PENALTY = 1.35 

# Short text handling
TEMP_SHORT = 0.18
TOP_P_SHORT = 0.60
TOP_K_SHORT = 12
MAX_NEW_SHORT = 256
SHORT_RETRY = 2
MAX_SEC_FOR_SHORT = 1.3 

# -------------------------
# TEST DATA
# -------------------------
N_VOICES = 100
SEED = 5555 # New seed for new voices

# Raw text (Code will add [uv_break] automatically)
TEST_TEXTS = {
    "long": "Yeah, no, this is my backyard. It's never ending so just the way I like it. So social distancing has never been a problem.",
    "question": "Wait, are you actually saying we should go there right now?",
    "short": "Exactly."
}

# -------------------------
# HELPERS (From testing3.py)
# -------------------------
def ensure_end_punct(s: str) -> str:
    s = (s or "").strip()
    if not s: return s
    s = re.sub(r"\[(overlapping|interrupts|slight overlap)\]", "", s, flags=re.I).strip()
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".?!":
        s += "."
    return s

def is_very_short_utterance(text: str) -> bool:
    t = (text or "").strip()
    tokens = re.findall(r"[A-Za-z']+|[0-9]+|[^\s]", t)
    alpha = re.sub(r"[^A-Za-z]", "", t)
    return (len(alpha) <= 6) or (len(tokens) <= 2)

def save_wav(path: Path, wav: np.ndarray, sr: int = SR):
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(str(path), wav, sr)

def pad_tail(wav: np.ndarray, sr: int = SR, sec: float = 0.25) -> np.ndarray:
    return np.concatenate([wav, np.zeros(int(sec * sr), dtype=np.float32)])

# -------------------------
# SETUP MODEL
# -------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Loading ChatTTS...")
chat = ChatTTS.Chat()
chat.load(compile=False) # Windows safe

def make_params(spk_emb, short: bool):
    return ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb,
        temperature=TEMP_SHORT if short else TEMP,
        top_P=TOP_P_SHORT if short else TOP_P,
        top_K=TOP_K_SHORT if short else TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        max_new_token=MAX_NEW_SHORT if short else MAX_NEW,
        min_new_token=MIN_NEW_SHORT if short else MIN_NEW_LONG,
        ensure_non_empty=True,
    )

def synth(text: str, spk_emb) -> np.ndarray:
    """
    Exact replica of the synthesis logic from testing3.py
    """
    raw = (text or "").strip()
    short = is_very_short_utterance(raw)
    tts_text = ensure_end_punct(raw) + END_MARK

    params = make_params(spk_emb, short=short)
    wavs = chat.infer([tts_text], params_infer_code=params)
    wav = np.asarray(wavs[0], dtype=np.float32)

    # Retry logic for short utterances
    if short: 
        # Simple length check to see if it hallucinated excessively
        dur = len(wav) / SR
        if dur > MAX_SEC_FOR_SHORT:
            for _ in range(SHORT_RETRY):
                params = make_params(spk_emb, short=True)
                wavs = chat.infer([tts_text], params_infer_code=params)
                wav2 = np.asarray(wavs[0], dtype=np.float32)
                if (len(wav2) / SR) <= MAX_SEC_FOR_SHORT:
                    wav = wav2
                    break
            # Hard cut if still too long
            if (len(wav) / SR) > MAX_SEC_FOR_SHORT:
                wav = wav[: int(MAX_SEC_FOR_SHORT * SR)]

    wav = pad_tail(wav, SR, 0.25)
    return wav

# -------------------------
# MAIN EXECUTION
# -------------------------
# 1. Generate Bank
print(f"Sampling {N_VOICES} random embeddings...")
speaker_bank = {}
for i in range(N_VOICES):
    vid = f"v{i:04d}"
    speaker_bank[vid] = chat.sample_random_speaker()

# Save Bank immediately
torch.save(speaker_bank, OUT_DIR / "speaker_bank.pt")
print(f"Bank saved to {OUT_DIR / 'speaker_bank.pt'}")

# 2. Generate Auditions
print("Starting Audition Generation...")
for vid, emb in tqdm(speaker_bank.items(), total=N_VOICES):
    
    for label, text in TEST_TEXTS.items():
        try:
            # Use the robust synth function
            wav = synth(text, emb)
            
            # Save: v0001_long.wav, etc.
            filename = f"{vid}_{label}.wav"
            save_wav(OUT_DIR / filename, wav, SR)
            
        except Exception as e:
            print(f"Failed {vid}-{label}: {e}")

print(f"\nDone! Check folder: {OUT_DIR.resolve()}")
print("Tips for selecting voices:")
print("1. 'short' uses different params than 'long'. Make sure the voice sounds the same in both.")
print("2. If 'question' sounds American but 'long' sounds British, discard it.")