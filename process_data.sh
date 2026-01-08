#!/usr/bin/env bash
set -e

# ==============================
# 1. CONFIGURATION
# ==============================
SCRIPT_NAME="process_data.py"

ORIGINAL_INPUT="input_data/Output_STEP1+2_ENGLISH_TTS"
FINAL_OUTPUT="output_data/ENGLISH_chattts_Gem"

# Your GPUs
GPUS=(3 5 6 7) 

# Directories
WORK_DIR="workspace_shards"
LOG_DIR="logs_chattts"

# ABSOLUTE PATH TO PROJECT ROOT
PROJECT_ROOT=$(pwd)

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
split -n l/${NUM_GPUS} -d "${WORK_DIR}/all_files.txt" "${WORK_DIR}/chunk_"

# ==============================
# 3. SETUP & LAUNCH
# ==============================
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    CHUNK_FILE="${WORK_DIR}/chunk_0${i}"
    
    # Define Paths
    SHARD_ROOT="${WORK_DIR}/shard_${i}"
    SHARD_INPUT="${SHARD_ROOT}/input"
    SHARD_SCRIPT="${SHARD_ROOT}/run.py"
    
    mkdir -p "${SHARD_INPUT}"
    
    # A. Link input files
    xargs -a "${CHUNK_FILE}" cp -l -t "${SHARD_INPUT}" 2>/dev/null || xargs -a "${CHUNK_FILE}" cp -t "${SHARD_INPUT}"

    # B. Copy Script
    cp "${SCRIPT_NAME}" "${SHARD_SCRIPT}"
    
    # C. *** KEY FIX ***: SYMBOLIC LINK TO CHATTTS
    # We put a "shortcut" to the ChatTTS folder inside the shard folder.
    # Python will find it immediately.
    ln -s "${PROJECT_ROOT}/ChatTTS" "${SHARD_ROOT}/ChatTTS"
    # Also link 'voices' folder just in case your script looks for relative path 'voices/'
    ln -s "${PROJECT_ROOT}/voices" "${SHARD_ROOT}/voices"

    # D. Modify Paths in Python Script
    sed -i "s|SAMPLE_DIR.*=.*Path(.*)|SAMPLE_DIR = Path(\"${SHARD_INPUT}\")|g" "${SHARD_SCRIPT}"
    sed -i "s|OUT_DIR.*=.*Path(.*)|OUT_DIR = Path(\"${FINAL_OUTPUT}\")|g" "${SHARD_SCRIPT}"
    sed -i "s|\"metadata.csv\"|\"metadata_gpu${GPU_ID}.csv\"|g" "${SHARD_SCRIPT}"
    
    # Also fix VOICE_DIR to use absolute path or the local link
    # (Optional safety measure: force VOICE_DIR to be the linked one)
    sed -i "s|VOICE_DIR.*=.*Path(.*)|VOICE_DIR = Path(\"${SHARD_ROOT}/voices\")|g" "${SHARD_SCRIPT}"

    # E. Launch (No PYTHONPATH needed anymore!)
    LOG_FILE="${LOG_DIR}/gpu_${GPU_ID}.log"
    echo "🚀 Launching GPU ${GPU_ID}..."
    
    nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python "${SHARD_SCRIPT}" > "${LOG_FILE}" 2>&1 &
        
done

echo ""
echo "✅ All jobs running."
echo "   Monitor: tail -f ${LOG_DIR}/*.log"