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

# -----------------------
# USER CONFIG
# -----------------------
SAMPLE_DIR = Path("input_data/Output_STEP1+2_ENGLISH_TTS")
VOICE_DIR  = Path("voices")
OUT_DIR    = Path("output_data/ENGLISH_chattts_Gem")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = VOICE_DIR / "manifest.json"
BANK_PATH     = VOICE_DIR / "speaker_bank.pt"
CSV_PATH      = OUT_DIR / "metadata.csv"

# Audio Params
SR = 24000
PAUSE_SEC = 0.14
TURN_TAIL_PAD_SEC = 0.18
FILE_TAIL_PAD_SEC = 0.25
END_MARK = " [uv_break]"

# ChatTTS Params
TEMP = 0.12
TOP_P = 0.70
TOP_K = 20
MAX_NEW = 2048
MIN_NEW_LONG = 48
REPETITION_PENALTY = 1.35

# Performance
torch.set_float32_matmul_precision("high")

# -----------------------
# DATA LOADING
# -----------------------
print("Loading Manifest and Speaker Bank...")
if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Missing {MANIFEST_PATH}")
if not BANK_PATH.exists():
    raise FileNotFoundError(f"Missing {BANK_PATH}")

manifest_data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
speaker_bank_raw = torch.load(BANK_PATH, map_location="cpu")

americans = []
british = []
global_emb_map = {}
meta_lookup = {}

print("Verifying Voice Data...")

for entry in manifest_data:
    path_obj = Path(entry["path"])
    vid = path_obj.stem  # e.g., "v0014"

    if vid not in speaker_bank_raw:
        print(f"❌ CRITICAL ERROR: {vid} is in manifest but NOT in speaker_bank.pt!")
        raise KeyError(f"Missing voice: {vid}")

    raw_val = speaker_bank_raw[vid]

    meta_lookup[vid] = {
        "manifest_id": entry.get("id", "unknown"),
        "voice_file": path_obj.name,
        "accent": entry.get("accent", "unknown"),
        "gender": entry.get("gender", "unknown"),
    }

    accent = entry.get("accent", "").lower()
    if "american" in accent:
        americans.append(vid)
    elif "british" in accent:
        british.append(vid)

    # ChatTTS can accept either a tensor embedding or a code string in spk_emb
    global_emb_map[vid] = raw_val

print(f"✅ Success: Loaded {len(americans)} US and {len(british)} UK voices.")

if len(americans) < 2:
    raise ValueError(f"Need at least 2 American voices, got {len(americans)}")
if len(british) < 2:
    raise ValueError(f"Need at least 2 British voices, got {len(british)}")

# -----------------------
# RESUME LOGIC (FIXED)
# -----------------------
# Store finished IDs as STRINGS only, matching output wav stems (e.g., "11198")
finished_ids = set()

for wav_file in OUT_DIR.glob("*.wav"):
    finished_ids.add(str(wav_file.stem))

if CSV_PATH.exists():
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("dialogue_id"):
                    finished_ids.add(str(row["dialogue_id"]))
    except Exception as e:
        print(f"Warning: Could not read existing CSV ({e}). Relying on folder check.")

print(f"Found {len(finished_ids)} already completed dialogues. They will be skipped.")

# -----------------------
# HELPERS
# -----------------------
def get_voice_pair_ids(condition: str):
    if condition == "us_us":
        a, b = random.sample(americans, 2)
    elif condition == "uk_uk":
        a, b = random.sample(british, 2)
    else:  # mixed
        us = random.choice(americans)
        uk = random.choice(british)
        pair = [us, uk]
        random.shuffle(pair)
        a, b = pair[0], pair[1]
    return {"Host_A": a, "Host_B": b}

def ensure_end_punct(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = re.sub(r"\[(overlapping|interrupts|slight overlap)\]", "", s, flags=re.I).strip()
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".?!":
        s += "."
    return s

def save_wav(path: Path, wav: np.ndarray, sr: int = SR):
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(str(path), wav, sr)

def build_speaker_map(turns):
    """
    Ensure turn speakers map onto Host_A/Host_B keys used by current_tensors.
    - If speakers already include Host_A/Host_B, do nothing.
    - Otherwise map the first two unique speaker labels encountered to Host_A/Host_B.
    """
    uniq = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        sp = (t.get("speaker") or "").strip()
        if sp and sp not in uniq:
            uniq.append(sp)

    if "Host_A" in uniq or "Host_B" in uniq:
        return {}

    mapping = {}
    if len(uniq) >= 1:
        mapping[uniq[0]] = "Host_A"
    if len(uniq) >= 2:
        mapping[uniq[1]] = "Host_B"
    return mapping

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
        # If your ChatTTS version supports these and you want them, uncomment:
        # top_p=TOP_P,
        # top_k=TOP_K,
        # max_new_token=MAX_NEW,
    )

    wavs = chat.infer([text], params_infer_code=params)
    wav = np.asarray(wavs[0], dtype=np.float32)

    tail = np.zeros(int(TURN_TAIL_PAD_SEC * SR), dtype=np.float32)
    return np.concatenate([wav, tail])

# -----------------------
# MAIN LOOP (FIXED)
# -----------------------
json_files = sorted(SAMPLE_DIR.glob("*.json"))
random.shuffle(json_files)

n_total = len(json_files)
n_us = n_total // 3
n_uk = n_total // 3
n_mixed = n_total - n_us - n_uk
conditions = ["us_us"] * n_us + ["uk_uk"] * n_uk + ["mixed"] * n_mixed
random.shuffle(conditions)

print(f"Total Experiment Plan: {n_us} US-US, {n_uk} UK-UK, {n_mixed} Mixed.")

csv_header = [
    "dialogue_id", "condition",
    "host_a_file", "host_a_gender", "host_a_accent",
    "host_b_file", "host_b_gender", "host_b_accent",
]

file_exists = CSV_PATH.exists()
mode = "a" if file_exists else "w"
print(f"Opening CSV in '{mode}' mode...")

with open(CSV_PATH, mode=mode, newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(csv_header)
        f.flush()

    pause = np.zeros(int(PAUSE_SEC * SR), dtype=np.float32)
    file_tail = np.zeros(int(FILE_TAIL_PAD_SEC * SR), dtype=np.float32)

    for jf, condition in tqdm(zip(json_files, conditions), total=n_total, desc="Synthesizing"):

        data = json.loads(jf.read_text(encoding="utf-8"))

        # ENFORCE: output filename must match JSON top-level id
        if "id" not in data:
            raise KeyError(f"Missing top-level 'id' in {jf}")

        dialogue_id = str(data["id"]).strip()
        if not dialogue_id.isdigit():
            raise ValueError(f"Non-numeric id '{dialogue_id}' in {jf}")

        # ENFORCE: input filename stem must match id (optional but strongly recommended)
        if jf.stem != dialogue_id:
            raise ValueError(f"Filename/id mismatch: file {jf.stem}.json vs id {dialogue_id}")

        if dialogue_id in finished_ids:
            continue

        voice_ids = get_voice_pair_ids(condition)
        current_tensors = {role: global_emb_map[vid] for role, vid in voice_ids.items()}

        meta_a = meta_lookup[voice_ids["Host_A"]]
        meta_b = meta_lookup[voice_ids["Host_B"]]

        turns = data["dialogue_data"]["dialogue_turns"]

        # FIX: map dataset speaker labels to Host_A/Host_B so segments are not empty
        sp_map = build_speaker_map(turns)

        segments = []

        for turn in turns:
            role_raw = (turn.get("speaker") or "").strip()
            role = sp_map.get(role_raw, role_raw)

            if role not in current_tensors:
                continue

            text = turn.get("tts_text") or turn.get("text") or ""
            if not text.strip():
                continue

            wav = synth(text, current_tensors[role])
            segments.append(wav)
            segments.append(pause)

        if segments:
            full = np.concatenate(segments + [file_tail])

            # Output name EXACTLY matches JSON id (and input stem)
            out_path = OUT_DIR / f"{dialogue_id}.wav"
            save_wav(out_path, full, SR)

            writer.writerow([
                dialogue_id,
                condition,
                meta_a["voice_file"], meta_a["gender"], meta_a["accent"],
                meta_b["voice_file"], meta_b["gender"], meta_b["accent"],
            ])
            f.flush()

            # Update finished_ids during this run
            finished_ids.add(dialogue_id)

            del segments, full, wav
            gc.collect()

print(f"Done. Audio and Metadata saved to: {OUT_DIR.resolve()}")
