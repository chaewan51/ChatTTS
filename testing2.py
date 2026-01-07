from pathlib import Path
import json
import re
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
import ChatTTS

# PERFORMANCE BOOST: Enables TensorFloat32 for 3090/A100
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
TEMP = 0.12               
REPETITION_PENALTY = 1.25 
MIN_NEW_LONG = 16         
END_MARK = " [uv_break][force_stop]"

def clean_text(s: str) -> str:
    # Remove characters that trigger 'invalid character' warnings in your logs
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
    # Ensure we get the actual Tensor
    emb = chat.sample_random_speaker()
    if isinstance(emb, str):
        emb = chat.get_speaker_emb(emb)
    
    # Critical Fix for the RuntimeError: 
    # Squeeze the tensor to ensure it is 1D [768] before we batch it
    spk_emb_map[role] = emb.detach().cpu().squeeze()

# -----------------------
# BATCH PROCESSING
# -----------------------
json_files = sorted(SAMPLE_DIR.glob("*.json"))
pause_wav = np.zeros(int(0.15 * SR), dtype=np.float32)

for jf in tqdm(json_files, desc="Synthesizing"):
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

    # FIX: Correctly stacking for the infer engine
    # We move to CUDA only at the moment of inference
    stacked_embs = torch.stack(batch_embs).to('cuda')

    params = ChatTTS.Chat.InferCodeParams(
        spk_emb=stacked_embs, 
        temperature=TEMP,
        repetition_penalty=REPETITION_PENALTY,
        min_new_token=MIN_NEW_LONG,
    )

    # Parallel inference on GPU
    # 
    wavs = chat.infer(batch_texts, params_infer_code=params)

    combined_audio = []
    for w in wavs:
        combined_audio.append(np.asarray(w, dtype=np.float32))
        combined_audio.append(pause_wav)

    if combined_audio:
        final_wav = np.concatenate(combined_audio)
        sf.write(OUT_DIR / f"{jf.stem}.wav", final_wav, SR)

print(f"\nSuccess! Processed {len(json_files)} files.")