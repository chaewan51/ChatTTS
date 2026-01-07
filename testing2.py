from pathlib import Path
import json
import re
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
import ChatTTS

# PERFORMANCE: This enables TF32 on your 3090/A100 for faster matrix math
torch.set_float32_matmul_precision('high')

# -----------------------
# USER CONFIG
# -----------------------
SAMPLE_DIR = Path("sample")
VOICE_DIR  = Path("voices")
OUT_DIR    = Path("sample_output_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOICE_MAP = {"Host_A": "v0022", "Host_B": "v0028"}
SR = 24000

# Hallucination Control
TEMP = 0.12               
REPETITION_PENALTY = 1.25 
MIN_NEW_LONG = 16         
END_MARK = " [uv_break][force_stop]"

def clean_text(s: str) -> str:
    # Keeps alphanumeric and basic punctuation to avoid 'invalid character' warnings
    s = re.sub(r"[^A-Za-z0-9\s\.,']", "", s or "").strip()
    return s + END_MARK

# -----------------------
# LOAD & OPTIMIZE
# -----------------------
print("Initializing ChatTTS with hardware optimizations...")
chat = ChatTTS.Chat()
chat.load(compile=True, device='cuda') 

spk_emb_map = {}
for role, vid in VOICE_MAP.items():
    seed_val = int(re.sub(r"\D", "", vid))
    torch.manual_seed(seed_val)
    
    # Generate profile
    emb = chat.sample_random_speaker()
    
    # Force the embedding to be a flat 1D tensor of [768]
    if isinstance(emb, str):
        emb = torch.randn(768) 
    
    if torch.is_tensor(emb):
        # We use flatten() to ensure there are NO nested dimensions (like [1, 768])
        emb = emb.detach().cpu().flatten() 
        spk_emb_map[role] = emb

# -----------------------
# BATCH PROCESSING
# -----------------------


json_files = sorted(SAMPLE_DIR.glob("*.json"))
pause_wav = np.zeros(int(0.15 * SR), dtype=np.float32)

for jf in tqdm(json_files, desc="Synthesizing Dialogues"):
    data = json.loads(jf.read_text(encoding="utf-8"))
    turns = data.get("dialogue_data", {}).get("dialogue_turns", [])
    
    batch_texts = []
    batch_embs = []
    
    for turn in turns:
        role = turn.get("speaker")
        text = turn.get("tts_text") or turn.get("text")
        if text and role in spk_emb_map:
            batch_texts.append(clean_text(text))
            batch_embs.append(spk_emb_map[role])

    if not batch_texts:
        continue

    # THE CRITICAL FIX:
    # We stack flat 1D tensors to create a 2D matrix: [Number of sentences, 768]
    stacked_embs = torch.stack(batch_embs).to('cuda')

    params = ChatTTS.Chat.InferCodeParams(
        spk_emb=stacked_embs, 
        temperature=TEMP,
        repetition_penalty=REPETITION_PENALTY,
        min_new_token=MIN_NEW_LONG,
    )

    # All turns in the dialogue are now processed in a single GPU pass
    wavs = chat.infer(batch_texts, params_infer_code=params)

    combined_audio = []
    for w in wavs:
        combined_audio.append(np.asarray(w, dtype=np.float32))
        combined_audio.append(pause_wav)

    if combined_audio:
        final_wav = np.concatenate(combined_audio)
        sf.write(OUT_DIR / f"{jf.stem}.wav", final_wav, SR)

print(f"\nSuccess! Processed {len(json_files)} files.")