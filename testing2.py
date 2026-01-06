from pathlib import Path
import json
import re
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
import ChatTTS

# -----------------------
# USER CONFIG
# -----------------------
SAMPLE_DIR = Path("sample")
VOICE_DIR  = Path("voices")
OUT_DIR    = Path("sample_output_fixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOICE_MAP = {
    "Host_A": "v0022",
    "Host_B": "v0028",
}

SR = 24000
TEMP = 0.12               
REPETITION_PENALTY = 1.25 
MIN_NEW_LONG = 16         
END_MARK = " [uv_break][force_stop]"

def clean_text(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\s\.,!\?']", "", s or "").strip()
    return s + END_MARK

# -----------------------
# LOAD & OPTIMIZE
# -----------------------
print("Initializing ChatTTS...")
chat = ChatTTS.Chat()
chat.load(compile=True, device='cuda') 

# Attempt to load bank, handle missing file gracefully
bank_path = VOICE_DIR / "speaker_bank.pt"
raw_bank = torch.load(bank_path, map_location='cuda') if bank_path.exists() else {}

spk_emb_map = {}
for role, vid in VOICE_MAP.items():
    entry = raw_bank.get(vid)
    
    # If the entry is a string or missing, we must generate the TENSOR
    if isinstance(entry, str) or entry is None:
        seed_val = int(re.sub(r"\D", "", vid))
        print(f"Generating Tensor for {vid} using Torch seed {seed_val}...")
        
        torch.manual_seed(seed_val)
        # In some versions, sample_random_speaker returns a tensor directly, 
        # in others it might require different handling. 
        # This is the most compatible way to force a tensor return:
        emb = chat.sample_random_speaker() 
        
        # FINAL SAFETY: If it's still a string, we manually extract the embedding from the model
        if isinstance(emb, str):
            # Fallback for versions where sample_random_speaker only returns the ID
            emb = chat.get_speaker_emb(emb) if hasattr(chat, 'get_speaker_emb') else None
            if emb is None:
                # Last resort: sample until we get a tensor or use manual random
                emb = torch.randn(768).to('cuda') # Standard ChatTTS emb size
    else:
        emb = entry
    
    # Validation: Ensure it is a tensor before calling .to()
    if not torch.is_tensor(emb):
        raise TypeError(f"Critical Error: Could not resolve {vid} to a Tensor. Found {type(emb)} instead.")
        
    spk_emb_map[role] = emb.to(device='cuda', dtype=torch.float32)

# -----------------------
# BATCH PROCESSING
# -----------------------
# Stacking allows the 3090/A100 to process the whole dialogue at once.


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

    # Batch parameters for hallucination control
    stacked_embs = torch.stack(batch_embs)
    params = ChatTTS.Chat.InferCodeParams(
        spk_emb=stacked_embs, 
        temperature=TEMP,
        repetition_penalty=REPETITION_PENALTY,
        min_new_token=MIN_NEW_LONG,
    )

    # Parallel inference on GPU
    wavs = chat.infer(batch_texts, params_infer_code=params)

    combined_audio = []
    for w in wavs:
        combined_audio.append(np.asarray(w, dtype=np.float32))
        combined_audio.append(pause_wav)

    final_wav = np.concatenate(combined_audio)
    sf.write(OUT_DIR / f"{jf.stem}.wav", final_wav, SR)

print(f"\nSuccess! WAVs saved to: {OUT_DIR.resolve()}")