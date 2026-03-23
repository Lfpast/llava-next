#!/usr/bin/env bash
# Local StageB-only debug script for video training.
# It reuses an existing StageA checkpoint and skips StageA execution.

set -euo pipefail

REPO_ROOT="/home/dduab/jiayusheng/LLaVA-NeXT"
IMAGE_FOLDER="${REPO_ROOT}/data/train/image_media"
VIDEO_FOLDER="${REPO_ROOT}/data/train/video_media"

DATA_YAML_VIDEO_ONLY="${REPO_ROOT}/scripts/video/train/local_data_video.yaml"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="disabled"
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/project/peilab/jys/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
PROMPT_VERSION="qwen_1_5"

ENABLE_VFR="true"
ENABLE_VFA="false"

VFR_WEIGHT="0.5"
VFA_WEIGHT="0.0"
VFA_LAYER="16"

# User knobs
RUN_NAME="vfr_temp2_stageB_test"
OUTPUT_DIR="${REPO_ROOT}/work_dirs/siglip"

# IMPORTANT: set this to an existing StageA output directory.
# Example from your log:
#   /home/dduab/jiayusheng/LLaVA-NeXT/work_dirs/siglip/vfr_temp2_stageA_tmp
STAGE_A_DIR="${REPO_ROOT}/work_dirs/siglip/vfr_temp2_stageA_tmp"

MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-local_video_v100"
[[ -n "$RUN_NAME" ]] && MID_RUN_NAME="$RUN_NAME"
[[ -n "$OUTPUT_DIR" ]] && FINAL_OUTPUT_DIR="$OUTPUT_DIR/$MID_RUN_NAME" || FINAL_OUTPUT_DIR="./work_dirs/$MID_RUN_NAME"

COMMON_ARGS=(
  --deepspeed "${REPO_ROOT}/scripts/zero3.json"
  --version "${PROMPT_VERSION}"
  --image_folder "${IMAGE_FOLDER}"
  --video_folder "${VIDEO_FOLDER}"
  --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model"
  --mm_vision_tower_lr 2e-6
  --vision_tower "${VISION_MODEL_VERSION}"
  --mm_projector_type mlp2x_gelu
  --mm_vision_select_layer -2
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --group_by_modality_length True
  --attn_implementation sdpa
  --num_train_epochs 1
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps 2
  --evaluation_strategy no
  --save_strategy no
  --learning_rate 1e-5
  --weight_decay 0.
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --logging_steps 1
  --fp16 False
  --bf16 True
  --tf32 False
  --model_max_length 4096
  --gradient_checkpointing True
  --dataloader_num_workers 2
  --lazy_preprocess True
  --report_to none
  --torch_compile False
  --dataloader_drop_last True
  --mm_spatial_pool_stride 2
  --save_only_model True
)

EXTRA_ARGS=()
if [[ "${ENABLE_VFR}" == "true" ]]; then
  EXTRA_ARGS+=(
    --vfr_enabled True
    --vfr_weight "${VFR_WEIGHT}"
    --vfr_gt_model_name facebook/dinov2-base
    --vfr_use_dinov2_transformers True
    --vfr_log_every 50
    --vfr_tokens_mode video
  )
fi

if [[ "${ENABLE_VFA}" == "true" ]]; then
  EXTRA_ARGS+=(
    --vfa_enabled True
    --vfa_weight "${VFA_WEIGHT}"
    --vfa_layer "${VFA_LAYER}"
  )
fi

if [[ ! -d "${STAGE_A_DIR}" ]]; then
  echo "ERROR: STAGE_A_DIR does not exist: ${STAGE_A_DIR}"
  exit 1
fi

mkdir -p "${FINAL_OUTPUT_DIR}"

echo "------------------------------------------------------------"
echo "Stage: stageB_video_only"
echo "model_name_or_path: ${STAGE_A_DIR}"
echo "data_path:          ${DATA_YAML_VIDEO_ONLY}"
echo "output_dir:         ${FINAL_OUTPUT_DIR}"
echo "------------------------------------------------------------"

# StageA is intentionally skipped in this script.
# It directly starts StageB from an existing StageA checkpoint.
torchrun --nproc_per_node=3 --master_port 30000 \
  -m llava.train.train_mem \
  "${COMMON_ARGS[@]}" \
  --model_name_or_path "${STAGE_A_DIR}" \
  --data_path "${DATA_YAML_VIDEO_ONLY}" \
  --run_name "${MID_RUN_NAME}_stageB_video" \
  --output_dir "${FINAL_OUTPUT_DIR}" \
  --image_aspect_ratio anyres_max_8 \
  --mm_patch_merge_type spatial_unpad \
  --mm_newline_position grid \
  --frames_upbound 16 \
  --add_time_instruction True \
  --force_sample True \
  --image_grid_pinpoints "(1x1),...,(6x6)" \
  "${EXTRA_ARGS[@]}"

echo "StageB-only run finished. Outputs: ${FINAL_OUTPUT_DIR}"
