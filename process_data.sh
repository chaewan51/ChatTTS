#!/usr/bin/env bash
set -e

# ==============================
# 1. CONFIGURATION
# ==============================
SCRIPT_NAME="testing3_final.py"  # Make sure this matches your python filename

# Original Input Folder
ORIGINAL_INPUT="input_data/Output_STEP1+2_ENGLISH_TTS"

# ONE COMMON OUTPUT FOLDER
FINAL_OUTPUT="output_data/ENGLISH_chattts_Gem"

# Your GPUs
GPUS=(3 5 6 7) 

# Temporary workspace for splitting inputs
WORK_DIR="workspace_shards"
LOG_DIR="logs_chattts"
# KEY FIX: Capture the absolute path of your project root
PROJECT_ROOT=$(pwd)

mkdir -p "${LOG_DIR}"
mkdir -p "${FINAL_OUTPUT}"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

# ==============================
# 2. SPLIT INPUTS
# ==============================
echo "🔹 Gathering and splitting files..."

# Find all JSONs
find "${ORIGINAL_INPUT}" -name "*.json" | sort > "${WORK_DIR}/all_files.txt"
NUM_GPUS=${#GPUS[@]}

# Split list into N chunks
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
    
    # A. Link input files to a temp folder (keeps them isolated)
    xargs -a "${CHUNK_FILE}" cp -l -t "${SHARD_INPUT}" 2>/dev/null || xargs -a "${CHUNK_FILE}" cp -t "${SHARD_INPUT}"

    # B. Create a modified copy of the Python script
    cp "${SCRIPT_NAME}" "${SHARD_SCRIPT}"
    
    # C. SED MAGIC: Modify paths in the script copy
    
    # 1. Point INPUT to the specific shard folder
    sed -i "s|SAMPLE_DIR.*=.*Path(.*)|SAMPLE_DIR = Path(\"${SHARD_INPUT}\")|g" "${SHARD_SCRIPT}"
    
    # 2. Point OUTPUT to the COMMON folder
    sed -i "s|OUT_DIR.*=.*Path(.*)|OUT_DIR = Path(\"${FINAL_OUTPUT}\")|g" "${SHARD_SCRIPT}"
    
    # 3. RENAME THE CSV to avoid conflicts (metadata_gpu0.csv, metadata_gpu1.csv...)
    sed -i "s|\"metadata.csv\"|\"metadata_gpu${GPU_ID}.csv\"|g" "${SHARD_SCRIPT}"

    # D. Launch
    LOG_FILE="${LOG_DIR}/gpu_${GPU_ID}.log"
    echo "🚀 Launching GPU ${GPU_ID}..."
    
    # --- FIX IS HERE ---
    # We add PYTHONPATH="${PROJECT_ROOT}" so python knows where to find 'ChatTTS' folder
    nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${PROJECT_ROOT}" \
        python "${SHARD_SCRIPT}" > "${LOG_FILE}" 2>&1 &
        
done

echo ""
echo "✅ All jobs running."
echo "   Audio files will appear in: ${FINAL_OUTPUT}"
echo "   Monitor errors in: ${LOG_DIR}/gpu_*.log"