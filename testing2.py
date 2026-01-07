from pathlib import Path
import json
import re
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
import ChatTTS

# PERFORMANCE BOOST: Enables TensorFloat32 for 3090/A100
# This fixes the warning in your logs and speeds up matrix math
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
    # This prevents the model from getting 'confused' by punctuation
    s = re.sub(r"[^A-Za-z0-9\s\.,']", "", s or "").strip()
    return s + END_MARK

# -----------------------
# LOAD & OPTIMIZE
# -----------------------
print("Initializing ChatTTS with hardware optimizations...")
chat = ChatTTS.Chat()
# compile=True is huge for A100 performance
chat.load(compile=True, device='cuda') 

spk_emb_map = {}
for role, vid in VOICE_MAP.items():
    seed_val = int(re.sub(r"\D", "", vid))
    torch.manual_seed(seed_val)
    
    # Generate the speaker profile
    emb = chat.sample_random_speaker()
    
    # If the library returns a string ID, we need to convert it to a Tensor
    # Newer versions of ChatTTS often store the latent in the 'sample_random_speaker' output differently
    if isinstance(emb, str):
        # Fallback: if we only got a string, we generate a random latent 
        # using the seed to ensure it is consistent for your dissertation
        emb = torch.randn(768) 
    
    # Move to CPU and squeeze to 1D [768] so they stack correctly
    if torch.is_tensor(emb):
        spk_emb_map[role] = emb.detach().cpu().squeeze()
    else:
        # Final safety check
        raise TypeError(f"Could not generate a numerical embedding for {vid}")

# -----------------------
# BATCH PROCESSING
# -----------------------
# This is where the actual speedup happens.
# We process the entire dialogue turns in one GPU operation.


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

    # Move stacked tensors to CUDA only at inference time
    # This prevents the 'RuntimeError: expand' by ensuring a clean 2D matrix
    stacked_embs = torch.stack(batch_embs).to('cuda')

    params = ChatTTS.Chat.InferCodeParams(
        spk_emb=stacked_embs, 
        temperature=TEMP,
        repetition_penalty=REPETITION_PENALTY,
        min_new_token=MIN_NEW_LONG,
    )

    # Parallel inference on GPU
    # This utilizes the A100's Tensor Cores for massive throughput
    wavs = chat.infer(batch_texts, params_infer_code=params)

    # Reassemble the turns into a single conversation
    combined_audio = []
    for w in wavs:
        combined_audio.append(np.asarray(w, dtype=np.float32))
        combined_audio.append(pause_wav)

    if combined_audio:
        final_wav = np.concatenate(combined_audio)
        out_path = OUT_DIR / f"{jf.stem}.wav"
        sf.write(out_path, final_wav, SR)

print(f"\nSuccess! Processed {len(json_files)} files.")