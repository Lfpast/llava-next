# ============================================================
# LLaVA-Video 评估脚本
# ============================================================

set -euo pipefail
WORK_DIR="/home/dduab/jiayusheng/LLaVA-NeXT"
EVAL_DIR="/home/dduab/jiayusheng/LLaVA-NeXT/data/eval_local"
VIDEO_DIR="/home/dduab/jiayusheng/LLaVA-NeXT/data/test/video_media"
IMAGE_DIR="/home/dduab/jiayusheng/LLaVA-NeXT/data/test/image_media"

# ============================================================
# 【用户配置】
#   RUN_NAME   : work_dirs/{encoder_type}/ 下的子目录名（即训练时的 RUN_NAME）
#   MODEL_PATH : 直接指定模型路径（设置后 RUN_NAME 不再使用）
# ============================================================

RUN_NAME="v1_lr1e5_16_4096"
MODEL_PATH=""

# ── 路径解析 ────────────────────────────────────────────────
BASE="${WORK_DIR}/work_dirs/siglip"

if [[ -n "${MODEL_PATH}" ]]; then
  FINAL_MODEL_PATH="${MODEL_PATH}"
elif [[ -n "${RUN_NAME}" ]]; then
  FINAL_MODEL_PATH="${BASE}/${RUN_NAME}"
else
  # 自动选最新
  FINAL_MODEL_PATH=$(ls -td "${BASE}/"*/ 2>/dev/null | head -1)
  if [[ -z "${FINAL_MODEL_PATH}" ]]; then
    echo "[ERROR] No model found in ${BASE}/"
    exit 1
  fi
  echo "Auto-selected model: ${FINAL_MODEL_PATH}"
fi

OUTPUT_DIR="${FINAL_MODEL_PATH}/eval"
LOG_PATH="${OUTPUT_DIR}/eval.log"
mkdir -p "${OUTPUT_DIR}"

# 验证模型目录
if [[ ! -d "${FINAL_MODEL_PATH}" ]]; then
  echo "[ERROR] model directory not found: ${FINAL_MODEL_PATH}"
  echo "Available runs in ${BASE}/:" 
  ls -1 "${BASE}/" 2>/dev/null || echo "(empty)"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED=1

echo "============================================================" | tee "${LOG_PATH}"
echo "Evaluation" | tee -a "${LOG_PATH}"
echo "Model path    : ${FINAL_MODEL_PATH}" | tee -a "${LOG_PATH}"
echo "Output dir    : ${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
echo "============================================================" | tee -a "${LOG_PATH}"

python "${WORK_DIR}/scripts/run_local_eval.py" \
  --model_path "${FINAL_MODEL_PATH}" \
  --eval_dir "${EVAL_DIR}" \
  --video_dir "${VIDEO_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_frames 16 \
  --device cuda \
  --tasks all \
  2>&1 | tee -a "${LOG_PATH}"

echo "Evaluation complete. Results in: ${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
