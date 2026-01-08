#!/usr/bin/env bash
set -e

# ==============================
# CONFIG
# ==============================
SCRIPT_NAME="process_data_parallel.py"  # The NEW python script above

ORIGINAL_INPUT="input_data/Output_STEP1+2_ENGLISH_TTS"
FINAL_OUTPUT="output_data/ENGLISH_chattts_Gem"

GPUS=(3 5 6 7) 

WORK_DIR="workspace_shards"
LOG_DIR="logs_chattts"

mkdir -p "${LOG_DIR}"
mkdir -p "${FINAL_OUTPUT}"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

# ==============================
# SPLIT INPUTS
# ==============================
echo "🔹 Splitting files..."
find "${ORIGINAL_INPUT}" -name "*.json" | sort > "${WORK_DIR}/all_files.txt"
NUM_GPUS=${#GPUS[@]}
split -n l/${NUM_GPUS} -d "${WORK_DIR}/all_files.txt" "${WORK_DIR}/chunk_"

# ==============================
# LAUNCH
# ==============================
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    CHUNK_FILE="${WORK_DIR}/chunk_0${i}"
    
    # We create a temporary input folder for this GPU
    SHARD_INPUT="${WORK_DIR}/shard_${i}_input"
    mkdir -p "${SHARD_INPUT}"
    
    # Link files (Symlink for speed)
    xargs -a "${CHUNK_FILE}" cp -l -t "${SHARD_INPUT}" 2>/dev/null || xargs -a "${CHUNK_FILE}" cp -t "${SHARD_INPUT}"

    LOG_FILE="${LOG_DIR}/gpu_${GPU_ID}.log"
    echo "🚀 Launching GPU ${GPU_ID} on $(wc -l < ${CHUNK_FILE}) files..."
    
    # RUN THE PYTHON SCRIPT DIRECTLY (No copying/moving the script!)
    # We pass the paths as arguments.
    nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python "${SCRIPT_NAME}" \
        --input_dir "${SHARD_INPUT}" \
        --output_dir "${FINAL_OUTPUT}" \
        --csv_name "metadata_gpu${GPU_ID}.csv" \
        > "${LOG_FILE}" 2>&1 &
        
done

echo ""
echo "✅ All jobs running."
echo "   Monitor logs: tail -f ${LOG_DIR}/*.log"