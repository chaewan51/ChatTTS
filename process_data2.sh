#!/usr/bin/env bash
set -euo pipefail

# ==============================
# 1. CONFIGURATION
# ==============================
SCRIPT_NAME="process_data2.py"  # <-- your updated python filename

# Original Input Folder (where the 1499 JSONs live)
ORIGINAL_INPUT="input_data/Output_STEP1+2Deep_ENGLISH_TTS"

# Output folder (shared across shards) - your python also writes here
FINAL_OUTPUT="output_data/ENGLISH_chattts_Deep"

# GPUs to use
# IMPORTANT:
# - If you're in Slurm, GPUs are often remapped and you should use (0 1 2 3)
# - If this node truly exposes 0-7 and you want 3/5/6/7, keep it.
GPUS=(3 5 6 7)

# Directories
WORK_DIR="workspace_shards"
LOG_DIR="logs_chattts2"

# Absolute project root (repo root)
PROJECT_ROOT="/gpuhome/czc5884/work/ChatTTS"

mkdir -p "${LOG_DIR}"
mkdir -p "${FINAL_OUTPUT}"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

# ==============================
# 2. SPLIT INPUTS
# ==============================
echo "🔹 Gathering and splitting files..."

find "${ORIGINAL_INPUT}" -name "*.json" | sort > "${WORK_DIR}/all_files.txt"
NUM_GPUS=${#GPUS[@]}

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "❌ GPUS array is empty."
  exit 1
fi

split -n l/${NUM_GPUS} -d "${WORK_DIR}/all_files.txt" "${WORK_DIR}/chunk_"

echo "✅ Total JSONs: $(wc -l < "${WORK_DIR}/all_files.txt")"
echo "✅ Shards: ${NUM_GPUS}"
wc -l "${WORK_DIR}"/chunk_* || true

# ==============================
# 3. SETUP & LAUNCH
# ==============================
for i in "${!GPUS[@]}"; do
  GPU_ID=${GPUS[$i]}
  CHUNK_FILE="${WORK_DIR}/chunk_0${i}"

  SHARD_ROOT="${WORK_DIR}/shard_${i}"
  SHARD_INPUT="${SHARD_ROOT}/input"
  SHARD_SCRIPT="${SHARD_ROOT}/run.py"

  mkdir -p "${SHARD_INPUT}"

  # A) Link/copy input JSONs into shard input
  # Prefer hardlinks (fast), fall back to normal copy if hardlink fails.
  xargs -a "${CHUNK_FILE}" cp -l -t "${SHARD_INPUT}" 2>/dev/null || xargs -a "${CHUNK_FILE}" cp -t "${SHARD_INPUT}"

  # B) Copy Python script into the shard
  cp "${SCRIPT_NAME}" "${SHARD_SCRIPT}"

  # C) Inject absolute sys.path so `import ChatTTS` works reliably
  sed -i "1i import sys; sys.path.insert(0, '${PROJECT_ROOT}')" "${SHARD_SCRIPT}"

  # D) Force shard-local SAMPLE_DIR and shared OUT_DIR / VOICE_DIR (absolute)
  #    (Your Python enforces filename stem == JSON id, so keeping original filenames via cp is important.)
  sed -i "s|^SAMPLE_DIR *= *Path(.*)|SAMPLE_DIR = Path(\"${SHARD_INPUT}\")|g" "${SHARD_SCRIPT}"
  sed -i "s|^OUT_DIR *= *Path(.*)|OUT_DIR = Path(\"${PROJECT_ROOT}/${FINAL_OUTPUT}\")|g" "${SHARD_SCRIPT}"
  sed -i "s|^VOICE_DIR *= *Path(.*)|VOICE_DIR = Path(\"${PROJECT_ROOT}/voices\")|g" "${SHARD_SCRIPT}"

  # E) Make each GPU write to a different CSV to avoid file-write collisions
  #    This assumes your python defines: CSV_PATH = OUT_DIR / "metadata.csv"
  sed -i "s|CSV_PATH *= *OUT_DIR */ *\"metadata\.csv\"|CSV_PATH = OUT_DIR / \"metadata_gpu${GPU_ID}.csv\"|g" "${SHARD_SCRIPT}"
  sed -i "s|CSV_PATH *= *OUT_DIR */ *'metadata\.csv'|CSV_PATH = OUT_DIR / 'metadata_gpu${GPU_ID}.csv'|g" "${SHARD_SCRIPT}"

  # F) Launch
  LOG_FILE="${LOG_DIR}/gpu_${GPU_ID}.log"
  echo "🚀 Launching shard ${i} on GPU ${GPU_ID} (log: ${LOG_FILE})..."

  nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python -u "${SHARD_SCRIPT}" > "${LOG_FILE}" 2>&1 &

done

echo ""
echo "✅ All jobs launched."
echo "   Monitor logs: tail -f ${LOG_DIR}/gpu_*.log"
echo "   Check running: ps -u \$USER -o pid,cmd | grep -E \"python -u .*${WORK_DIR}/shard_\" | grep -v grep"
