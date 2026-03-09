#!/usr/bin/env bash
# =============================================================================
# run.sh — Wrapper script đọc .env rồi chạy run.py
# Dùng: bash run.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Load .env (bỏ qua comment và dòng trống)
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    # shellcheck source=.env
    source "$ENV_FILE"
    set +o allexport
else
    echo "[ERROR] .env file không tìm thấy tại $ENV_FILE"
    exit 1
fi

# --- Chuyển boolean flags thành argparse flags ---
EXTRA_FLAGS=""

[[ "${THINK:-false}"                 == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --think"
[[ "${LATENT_SPACE_REALIGN:-false}"  == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --latent_space_realign"
[[ "${ACCUMULATE_LATENT:-false}"     == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --accumulate_latent"
[[ "${USE_VLLM:-false}"              == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --use_vllm"
[[ "${ENABLE_PREFIX_CACHING:-false}" == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --enable_prefix_caching"
[[ "${USE_SECOND_HF_MODEL:-false}"   == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --use_second_HF_model"
[[ "${DEVICE_MAP_AUTO:-false}"       == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --device_map_auto"
[[ "${SAVE_KV:-false}"               == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --save_kv"

echo "=================================================="
echo " LastLatentMAS Experiment Runner"
echo "=================================================="
echo " METHOD              : ${METHOD}"
echo " MODEL_NAME          : ${MODEL_NAME}"
echo " TASK                : ${TASK}"
echo " PROMPT              : ${PROMPT}"
echo " MAX_SAMPLES         : ${MAX_SAMPLES}"
echo " LATENT_STEPS        : ${LATENT_STEPS}"
echo " MAX_NEW_TOKENS      : ${MAX_NEW_TOKENS}"
echo " TEMPERATURE         : ${TEMPERATURE}"
echo " TOP_P               : ${TOP_P}"
echo " GENERATE_BS         : ${GENERATE_BS}"
echo " DEVICE              : ${DEVICE}"
echo " DEVICE2             : ${DEVICE2}"
echo " EXTRA_FLAGS         : ${EXTRA_FLAGS}"
echo "=================================================="

python3 "$SCRIPT_DIR/run.py" \
    --method                  "${METHOD}" \
    --model_name              "${MODEL_NAME}" \
    --task                    "${TASK}" \
    --prompt                  "${PROMPT}" \
    --max_samples             "${MAX_SAMPLES}" \
    --split                   "${SPLIT}" \
    --max_new_tokens          "${MAX_NEW_TOKENS}" \
    --latent_steps            "${LATENT_STEPS}" \
    --temperature             "${TEMPERATURE}" \
    --top_p                   "${TOP_P}" \
    --generate_bs             "${GENERATE_BS}" \
    --text_mas_context_length "${TEXT_MAS_CONTEXT_LENGTH}" \
    --device                  "${DEVICE}" \
    --device2                 "${DEVICE2}" \
    --tensor_parallel_size    "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization  "${GPU_MEMORY_UTILIZATION}" \
    --kv_flush_interval       "${KV_FLUSH_INTERVAL}" \
    --seed                    "${SEED}" \
    $EXTRA_FLAGS
