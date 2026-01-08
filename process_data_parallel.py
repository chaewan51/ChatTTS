import argparse
from pathlib import Path
import json
import re
import numpy as np
import torch
import random
import csv
import soundfile as sf
import gc 
from tqdm import tqdm
import ChatTTS
import sys

# -----------------------
# ARGUMENT PARSING (THIS IS THE KEY CHANGE)
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Path to input JSONs")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save WAVs")
parser.add_argument("--csv_name", type=str, required=True, help="Name of the CSV file")
parser.add_argument("--voice_dir", type=str, default="voices", help="Path to voices folder")
args = parser.parse_args()

# -----------------------
# CONFIG & PATHS (DYNAMIC NOW)
# -----------------------
SAMPLE_DIR = Path(args.input_dir)
OUT_DIR    = Path(args.output_dir)
VOICE_DIR  = Path(args.voice_dir)
CSV_PATH   = OUT_DIR / args.csv_name

# Ensure output exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = VOICE_DIR / "manifest.json"
BANK_PATH     = VOICE_DIR / "speaker_bank.pt"

# Audio Params (Identical to your code)
SR = 24000
PAUSE_SEC = 0.14
TURN_TAIL_PAD_SEC = 0.18
FILE_TAIL_PAD_SEC = 0.25
END_MARK = " [uv_break]"

# ChatTTS Params (Identical to your code)
TEMP = 0.12
TOP_P = 0.70
TOP_K = 20
MAX_NEW = 2048
MIN_NEW_LONG = 48
MIN_NEW_SHORT = 0
REPETITION_PENALTY = 1.35 

# Performance
torch.set_float32_matmul_precision('high')

# -----------------------
# DATA LOADING
# -----------------------
print(f"Loading Manifest from {MANIFEST_PATH}...")
if not MANIFEST_PATH.exists(): raise FileNotFoundError(f"Missing {MANIFEST_PATH}")
if not BANK_PATH.exists(): raise FileNotFoundError(f"Missing {BANK_PATH}")

manifest_data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
speaker_bank_raw = torch.load(BANK_PATH)

americans = []
british = []
global_emb_map = {} 
meta_lookup = {}    

print("Verifying Voice Data...")

for entry in manifest_data:
    path_obj = Path(entry["path"])
    vid = path_obj.stem 
    
    if vid not in speaker_bank_raw:
        print(f"❌ CRITICAL ERROR: {vid} is in manifest but NOT in speaker_bank.pt!")
        raise KeyError(f"Missing voice: {vid}")

    raw_val = speaker_bank_raw[vid]
    
    meta_lookup[vid] = {
        "manifest_id": entry.get("id", "unknown"),  
        "voice_file": path_obj.name,                
        "accent": entry.get("accent", "unknown"),   
        "gender": entry.get("gender", "unknown")    
    }

    accent = entry.get("accent", "").lower()
    if "american" in accent:
        americans.append(vid)
    elif "british" in accent:
        british.append(vid)
    
    global_emb_map[vid] = raw_val

print(f"✅ Success: Loaded {len(americans)} US and {len(british)} UK voices.")

# -----------------------
# CHECK EXISTING FILES (RESUME LOGIC)
# -----------------------
finished_ids = set()

# 1. Check Output Folder for WAVs (Global check)
for wav_file in OUT_DIR.glob("*.wav"):
    finished_ids.add(wav_file.stem) 

# 2. Check Local CSV
if CSV_PATH.exists():
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "dialogue_id" in row and row["dialogue_id"]:
                    finished_ids.add(row["dialogue_id"])
    except Exception as e:
        print(f"Warning: Could not read existing CSV ({e}). Relying on folder check.")

print(f"Found {len(finished_ids)} already completed dialogues (Global). Skipping them.")

# -----------------------
# HELPERS
# -----------------------
def get_voice_pair_ids(condition):
    if condition == "us_us":
        a, b = random.sample(americans, 2)
    elif condition == "uk_uk":
        a, b = random.sample(british, 2)
    else: # mixed
        us = random.choice(americans)
        uk = random.choice(british)
        pair = [us, uk]
        random.shuffle(pair)
        a, b = pair[0], pair[1]
    return {"Host_A": a, "Host_B": b}

def ensure_end_punct(s: str) -> str:
    s = (s or "").strip()
    if not s: return s
    s = re.sub(r"\[(overlapping|interrupts|slight overlap)\]", "", s, flags=re.I).strip()
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".?!":
        s += "."
    return s

def save_wav(path: Path, wav: np.ndarray, sr: int = SR):
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(str(path), wav, sr)

# -----------------------
# MODEL INIT
# -----------------------
chat = ChatTTS.Chat()
chat.load(compile=False) 

def synth(text: str, spk_emb) -> np.ndarray:
    text = ensure_end_punct(text) + END_MARK
    params = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb,
        temperature=TEMP,
        repetition_penalty=REPETITION_PENALTY,
        min_new_token=MIN_NEW_LONG,
    )
    wavs = chat.infer([text], params_infer_code=params)
    wav = np.asarray(wavs[0], dtype=np.float32)
    return np.concatenate([wav, np.zeros(int(TURN_TAIL_PAD_SEC * SR), dtype=np.float32)])

# -----------------------
# MAIN LOOP
# -----------------------
json_files = sorted(SAMPLE_DIR.glob("*.json"))
random.shuffle(json_files)

n_total = len(json_files)
if n_total == 0:
    print("No JSON files found in input directory.")
    sys.exit(0)

# Calculate ratio based on this shard's files
n_us = n_total // 3
n_uk = n_total // 3
n_mixed = n_total - n_us - n_uk
conditions = ["us_us"] * n_us + ["uk_uk"] * n_uk + ["mixed"] * n_mixed
random.shuffle(conditions)

csv_header = [
    "dialogue_id", "condition", 
    "host_a_file", "host_a_gender", "host_a_accent",
    "host_b_file", "host_b_gender", "host_b_accent"
]

file_exists = CSV_PATH.exists()
mode = "a" if file_exists else "w"

print(f"Opening CSV '{args.csv_name}' in '{mode}' mode...")

with open(CSV_PATH, mode=mode, newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    if not file_exists:
        writer.writerow(csv_header)
        f.flush()

    pause = np.zeros(int(PAUSE_SEC * SR), dtype=np.float32)
    file_tail = np.zeros(int(FILE_TAIL_PAD_SEC * SR), dtype=np.float32)

    for jf, condition in tqdm(zip(json_files, conditions), total=n_total, desc="Synthesizing"):
        
        possible_id = jf.stem 
        if possible_id in finished_ids:
            continue
            
        voice_ids = get_voice_pair_ids(condition)
        current_tensors = {role: global_emb_map[vid] for role, vid in voice_ids.items()}
        
        meta_a = meta_lookup[voice_ids["Host_A"]]
        meta_b = meta_lookup[voice_ids["Host_B"]]

        data = json.loads(jf.read_text(encoding="utf-8"))
        dialogue_id = data.get("id", jf.stem)
        
        if dialogue_id in finished_ids:
            continue

        segments = []
        turns = data["dialogue_data"]["dialogue_turns"]
        
        for turn in turns:
            role = turn.get("speaker", "")
            if role not in current_tensors: continue
            
            text = turn.get("tts_text") or turn.get("text") or ""
            if not text.strip(): continue

            wav = synth(text, current_tensors[role])
            segments.append(wav)
            segments.append(pause)

        if segments:
            full = np.concatenate(segments + [file_tail])
            filename = f"{dialogue_id}.wav"
            
            save_wav(OUT_DIR / filename, full, SR)
            
            writer.writerow([
                dialogue_id, 
                condition,
                meta_a["voice_file"], meta_a["gender"], meta_a["accent"],
                meta_b["voice_file"], meta_b["gender"], meta_b["accent"]
            ])
            f.flush()
            
            del segments, full, wav
            gc.collect()

print(f"Done. Saved to {OUT_DIR}")