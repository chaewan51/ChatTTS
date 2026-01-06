from pathlib import Path
import json
import re
import numpy as np
import torch
from tqdm import tqdm
import ChatTTS

# -----------------------
# USER CONFIG
# -----------------------
SAMPLE_DIR = Path("sample")           # folder with your dialogue JSONs
VOICE_DIR  = Path("voices")           # where speaker_bank.pt lives
OUT_DIR    = Path("sample_output_uvbreak")    # output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick the two voice IDs you liked (from voices/v####.wav)
VOICE_MAP = {
    "Host_A": "v0022",
    "Host_B": "v0028",
}

# Audio / pacing
SR = 24000

# Natural turn-taking: a SHORT pause between turns
PAUSE_SEC = 0.14                      # was 0.22 (often feels sluggish)

# Tiny tail pad AFTER EACH TURN to prevent last-phoneme clipping
TURN_TAIL_PAD_SEC = 0.18              # key fix for "cut off" endings

# Extra tail pad at the very end of the merged file
FILE_TAIL_PAD_SEC = 0.25

# End marker tokens (as STRINGS) — these are supported in ChatTTS README
# Use a light break rather than a big one for naturalness.
END_MARK = " [uv_break]"                # better than huge pauses; reduces cutoff

# -----------------------
# ChatTTS sampling params
# -----------------------
# Defaults for normal sentences
TEMP = 0.15
TOP_P = 0.70
TOP_K = 20
MAX_NEW = 2048

# NOTE: forcing MIN_NEW=64 can cause "wowowow..." babbling for very short texts.
# We'll use MIN_NEW only for longer lines, and 0 for short ones.
MIN_NEW_LONG = 48
MIN_NEW_SHORT = 0

# More conservative params for very short interjections
TEMP_SHORT = 0.18
TOP_P_SHORT = 0.60
TOP_K_SHORT = 12
MAX_NEW_SHORT = 256

# How many times to retry if we detect "degenerate" long babble for short text
SHORT_RETRY = 2
MAX_SEC_FOR_SHORT = 1.3               # if "wow" becomes 4s, we'll retry

# -----------------------
# Helpers
# -----------------------
def ensure_end_punct(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # remove overlap/interrupt markers if any still exist
    s = re.sub(r"\[(overlapping|interrupts|slight overlap)\]", "", s, flags=re.I).strip()
    # normalize repeated spaces
    s = re.sub(r"\s+", " ", s).strip()
    # For very short interjections, don't force punctuation too aggressively
    if s and s[-1] not in ".?!":
        # punctuation helps prevent early-stop and also reduces weird trailing audio
        s += "."
    return s

def save_wav(path: Path, wav: np.ndarray, sr: int = SR):
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    try:
        import soundfile as sf
        sf.write(str(path), wav, sr)
    except Exception:
        from scipy.io.wavfile import write as wavwrite
        wav_int16 = (wav * 32767.0).astype(np.int16)
        wavwrite(str(path), sr, wav_int16)

def pick_tts_text(turn: dict) -> str:
    # Prefer tts_text (already cleaned), fallback to text
    return turn.get("tts_text") or turn.get("text") or ""

def pad_silence(wav: np.ndarray, sec: float, sr: int = SR) -> np.ndarray:
    if sec <= 0:
        return wav
    return np.concatenate([wav, np.zeros(int(sec * sr), dtype=np.float32)])

def is_very_short_utterance(text: str) -> bool:
    t = (text or "").strip()
    # Examples: "wow", "yeah", "mm", "uh", "okay"
    # Use a conservative threshold: <= 6 chars or <= 2 tokens
    tokens = re.findall(r"[A-Za-z']+|[0-9]+|[^\s]", t)
    alpha = re.sub(r"[^A-Za-z]", "", t)
    return (len(alpha) <= 6) or (len(tokens) <= 2)

def looks_degenerate_short(text: str, wav: np.ndarray) -> bool:
    # If input is very short but audio is long, it's likely babble/repetition.
    if not is_very_short_utterance(text):
        return False
    dur = len(wav) / SR
    return dur > MAX_SEC_FOR_SHORT

# -----------------------
# Load ChatTTS + speaker bank
# -----------------------
chat = ChatTTS.Chat()
chat.load(compile=False)

bank_path = VOICE_DIR / "speaker_bank.pt"
if not bank_path.exists():
    raise FileNotFoundError(
        f"Missing {bank_path}. You need speaker_bank.pt from your voice sampling step."
    )

speaker_bank = torch.load(bank_path)

# Convert Host_A/Host_B -> actual embeddings
spk_emb_map = {}
for role, vid in VOICE_MAP.items():
    if vid not in speaker_bank:
        raise KeyError(f"{vid} not found in {bank_path}. Check your voice id.")
    spk_emb_map[role] = speaker_bank[vid]

def make_params(spk_emb, short: bool):
    if short:
        return ChatTTS.Chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=TEMP_SHORT,
            top_P=TOP_P_SHORT,
            top_K=TOP_K_SHORT,
            max_new_token=MAX_NEW_SHORT,
            min_new_token=MIN_NEW_SHORT,   # IMPORTANT: don't force length on "wow"
            ensure_non_empty=True,
        )
    else:
        return ChatTTS.Chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=TEMP,
            top_P=TOP_P,
            top_K=TOP_K,
            max_new_token=MAX_NEW,
            min_new_token=MIN_NEW_LONG,    # helps prevent early-stop on normal lines
            ensure_non_empty=True,
        )

def synth(text: str, spk_emb) -> np.ndarray:
    raw = (text or "").strip()
    short = is_very_short_utterance(raw)

    # Add punctuation + a light end marker for natural endings
    # (END_MARK is a string; your earlier code had END_BREAK as a python name, which would error)
    tts_text = ensure_end_punct(raw) + END_MARK

    # Try once with appropriate params
    params = make_params(spk_emb, short=short)
    wavs = chat.infer([tts_text], params_infer_code=params)
    wav = np.asarray(wavs[0], dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)

    # If it's a short utterance but produced long babble, retry with tighter settings
    if short and looks_degenerate_short(raw, wav):
        for _ in range(SHORT_RETRY):
            params = make_params(spk_emb, short=True)
            wavs = chat.infer([tts_text], params_infer_code=params)
            wav2 = np.asarray(wavs[0], dtype=np.float32)
            wav2 = np.clip(wav2, -1.0, 1.0)
            if not looks_degenerate_short(raw, wav2):
                wav = wav2
                break

        # As a last resort, hard-trim very short utterances to keep turn-taking natural
        if looks_degenerate_short(raw, wav):
            wav = wav[: int(MAX_SEC_FOR_SHORT * SR)]

    # Add a tiny tail pad to prevent end clipping (more reliable than huge break tokens)
    wav = pad_silence(wav, TURN_TAIL_PAD_SEC, SR)
    return wav

# -----------------------
# Main: loop JSONs -> output wav per file
# -----------------------
json_files = sorted(SAMPLE_DIR.glob("*.json"))
if not json_files:
    raise FileNotFoundError(f"No JSON files found in: {SAMPLE_DIR.resolve()}")

pause = np.zeros(int(PAUSE_SEC * SR), dtype=np.float32)
file_tail = np.zeros(int(FILE_TAIL_PAD_SEC * SR), dtype=np.float32)

for jf in tqdm(json_files, desc="Synthesizing dialogues"):
    data = json.loads(jf.read_text(encoding="utf-8"))

    dialogue_id = data.get("id", jf.stem)
    turns = data["dialogue_data"]["dialogue_turns"]

    segments = []
    for turn in turns:
        role = turn.get("speaker", "")
        if role not in spk_emb_map:
            raise KeyError(f"Unknown speaker '{role}' in {jf.name}. Add it to VOICE_MAP.")

        text = pick_tts_text(turn)
        if not (text or "").strip():
            continue

        wav = synth(text, spk_emb_map[role])
        segments.append(wav)
        segments.append(pause)

    full = np.concatenate(segments) if segments else np.zeros(1, dtype=np.float32)
    full = np.concatenate([full, file_tail])

    out_path = OUT_DIR / f"{dialogue_id}.wav"
    save_wav(out_path, full, SR)

print(f"Done. WAVs saved to: {OUT_DIR.resolve()}")
