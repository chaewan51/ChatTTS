from pathlib import Path
import json
import re
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
import ChatTTS

# Fixes the TF32 warning from your logs and speeds up matrix math on A100/3090
torch.set_float32_matmul_precision('high')

# -----------------------
# USER CONFIG
# -----------------------
SAMPLE_DIR = Path("sample")
VOICE_DIR  = Path("voices")
OUT_DIR    = Path("sample_output_reliable")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOICE_MAP = {"Host_A": "v0022", "Host_B": "v0028"}
SR = 24000

# Hallucination Control
TEMP = 0.12               
REPETITION_PENALTY = 1.35 # Programmatically kills the "really really" loops
MIN_NEW_LONG = 16         
END_MARK = " [uv_break][force_stop]"

def clean_text(s: str) -> str:
    # Keeps text clean to avoid 'invalid character' warnings in your logs
    s = re.sub(r"[^A-Za-z0-9\s\.,']", "", s or "").strip()
    return s + END_MARK

# -----------------------
# LOAD MODEL
# -----------------------
print("Initializing ChatTTS...")
chat = ChatTTS.Chat()
# compile=True is the main speedup for A100/3090 in sequential mode
chat.load(compile=True, device='cuda') 

# Pre-generate embeddings to avoid overhead during the loop
spk_emb_map = {}
for role, vid in VOICE_MAP.items():
    seed_val = int(re.sub(r"\D", "", vid))
    torch.manual_seed(seed_val)
    emb = chat.sample_random_speaker()
    
    # Ensure it's a Tensor and move to GPU
    if not torch.is_tensor(emb):
        emb = torch.randn(768) 
    spk_emb_map[role] = emb.to('cuda')

# -----------------------
# PROCESSING LOOP
# -----------------------
json_files = sorted(SAMPLE_DIR.glob("*.json"))
pause_wav = np.zeros(int(0.15 * SR), dtype=np.float32)

for jf in tqdm(json_files, desc="Processing Articles"):
    data = json.loads(jf.read_text(encoding="utf-8"))
    turns = data.get("dialogue_data", {}).get("dialogue_turns", [])
    
    combined_audio = []
    
    # We return to sequential processing to avoid the 'RuntimeError: expand'
    for turn in turns:
        role = turn.get("speaker")
        text = turn.get("tts_text") or turn.get("text")
        
        if not text or role not in spk_emb_map:
            continue
            
        cleaned = clean_text(text)
        
        # Individual inference for 100% stability
        params = ChatTTS.Chat.InferCodeParams(
            spk_emb=spk_emb_map[role], 
            temperature=TEMP,
            repetition_penalty=REPETITION_PENALTY,
            min_new_token=MIN_NEW_LONG,
        )
        
        # On A100, these individual calls are still very fast due to compilation
        wavs = chat.infer([cleaned], params_infer_code=params)
        
        if wavs:
            combined_audio.append(np.asarray(wavs[0], dtype=np.float32))
            combined_audio.append(pause_wav)

    if combined_audio:
        final_wav = np.concatenate(combined_audio)
        out_path = OUT_DIR / f"{jf.stem}.wav"
        sf.write(out_path, final_wav, SR)

print(f"\nSuccess! Processed {len(json_files)} dialogues safely.")