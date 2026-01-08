#!/usr/bin/env bash
set -e

# ==============================
# 1. CONFIGURATION
# ==============================
SCRIPT_NAME="process_data.py"  # Ensure this matches your python filename

# Original Input Folder
ORIGINAL_INPUT="input_data/Output_STEP1+2_ENGLISH_TTS"
FINAL_OUTPUT="output_data/ENGLISH_chattts_Gem"

# Your GPUs
GPUS=(3 5 6 7) 

# Directories
WORK_DIR="workspace_shards"
LOG_DIR="logs_chattts"

# *** KEY FIX: HARDCODED ABSOLUTE PATH FROM YOUR ERROR LOG ***
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
split -n l/${NUM_GPUS} -d "${WORK_DIR}/all_files.txt" "${WORK_DIR}/chunk_"

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
    
    # A. Link input files
    xargs -a "${CHUNK_FILE}" cp -l -t "${SHARD_INPUT}" 2>/dev/null || xargs -a "${CHUNK_FILE}" cp -t "${SHARD_INPUT}"

    # B. Copy Script
    cp "${SCRIPT_NAME}" "${SHARD_SCRIPT}"
    
    # C. *** ABSOLUTE PATH INJECTION (The Fix) ***
    
    # 1. Insert code at line 1 to force Python to look in your root folder
    #    This fixes "ModuleNotFoundError: No module named 'ChatTTS'"
    sed -i "1i import sys; sys.path.insert(0, '${PROJECT_ROOT}')" "${SHARD_SCRIPT}"

    # 2. Update Data Paths to be Absolute
    sed -i "s|SAMPLE_DIR.*=.*Path(.*)|SAMPLE_DIR = Path(\"${SHARD_INPUT}\")|g" "${SHARD_SCRIPT}"
    sed -i "s|OUT_DIR.*=.*Path(.*)|OUT_DIR = Path(\"${PROJECT_ROOT}/${FINAL_OUTPUT}\")|g" "${SHARD_SCRIPT}"
    
    # 3. Update Voice Dir to Absolute (Fixes "Missing manifest.json" errors)
    sed -i "s|VOICE_DIR.*=.*Path(.*)|VOICE_DIR = Path(\"${PROJECT_ROOT}/voices\")|g" "${SHARD_SCRIPT}"

    # 4. CSV Renaming
    sed -i "s|\"metadata.csv\"|\"metadata_gpu${GPU_ID}.csv\"|g" "${SHARD_SCRIPT}"

    # D. Launch
    LOG_FILE="${LOG_DIR}/gpu_${GPU_ID}.log"
    echo "🚀 Launching GPU ${GPU_ID}..."
    
    nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python "${SHARD_SCRIPT}" > "${LOG_FILE}" 2>&1 &
        
done

echo ""
echo "✅ All jobs running."
echo "   Monitor logs: tail -f ${LOG_DIR}/*.log"