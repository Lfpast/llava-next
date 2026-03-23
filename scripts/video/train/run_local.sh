#!/usr/bin/env bash
# ============================================================
# LLaVA-Video 训练脚本
# - 普通: 单阶段混合训练
# - VFR: 两阶段 image-only -> video-only 串联训练
# - VFA:
# ============================================================

set -euo pipefail

REPO_ROOT="/home/dduab/jiayusheng/LLaVA-NeXT"
IMAGE_FOLDER="${REPO_ROOT}/data/train/image_media"
VIDEO_FOLDER="${REPO_ROOT}/data/train/video_media"

DATA_YAML_MIXED="${REPO_ROOT}/scripts/video/train/local_data.yaml"
DATA_YAML_IMAGE_ONLY="${REPO_ROOT}/scripts/video/train/local_data_image.yaml"
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
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si"

# ============================================================
# Switch
# ============================================================
ENABLE_VFR="true"
ENABLE_VFA="false"

VFR_WEIGHT="0.5"
VFA_WEIGHT="0.0"
VFA_LAYER="16"

# ============================================================
# 【用户配置】输出路径与运行名称
#   RUN_NAME   : 本次运行的标识名，用于日志显示
#   OUTPUT_DIR : 模型 checkpoint 的保存目录（绝对路径或相对路径均可）
#                留空则自动生成（基于模型版本字符串）
# ============================================================

RUN_NAME="vfr_temp2"
OUTPUT_DIR="${REPO_ROOT}/work_dirs/siglip"

# 若未手动指定，则自动生成
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-local_video_v100"
[[ -n "$RUN_NAME" ]]   && MID_RUN_NAME="$RUN_NAME"
[[ -n "$OUTPUT_DIR" ]] && FINAL_OUTPUT_DIR="$OUTPUT_DIR/$MID_RUN_NAME" || FINAL_OUTPUT_DIR="./work_dirs/$MID_RUN_NAME"

STAGE_A_DIR="${FINAL_OUTPUT_DIR}_stageA_tmp"
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
  )
fi

if [[ "${ENABLE_VFA}" == "true" ]]; then
  EXTRA_ARGS+=(
    --vfa_enabled True
    --vfa_weight "${VFA_WEIGHT}"
    --vfa_layer "${VFA_LAYER}"
  )
fi

run_stage() {
  local stage_name="$1"
  local model_path="$2"
  local data_yaml="$3"
  local out_dir="$4"
  local aspect_ratio="$5"
  local patch_merge="$6"
  local newline_pos="$7"
  local frames_upbound="$8"
  local add_time_instruction="$9"
  local force_sample="${10}"
  local tokens_mode="${11}"

  local stage_args=(
    --model_name_or_path "${model_path}"
    --data_path "${data_yaml}"
    --run_name "${stage_name}"
    --output_dir "${out_dir}"
    --image_aspect_ratio "${aspect_ratio}"
    --mm_patch_merge_type "${patch_merge}"
    --mm_newline_position "${newline_pos}"
    --frames_upbound "${frames_upbound}"
    --add_time_instruction "${add_time_instruction}"
    --force_sample "${force_sample}"
  )

  if [[ "${aspect_ratio}" == "anyres_max_8" ]]; then
    stage_args+=(--image_grid_pinpoints "(1x1),...,(6x6)")
  fi

  if [[ "${ENABLE_VFR}" == "true" ]]; then
    stage_args+=(--vfr_tokens_mode "${tokens_mode}")
  fi

  echo "------------------------------------------------------------"
  if [[ "${ENABLE_VFR}" == "true" ]]; then
    echo "Stage: ${stage_name}"
  else
    echo "Stage: mixed"
  fi
  echo "model_name_or_path: ${model_path}"
  echo "data_path:          ${data_yaml}"
  echo "output_dir:         ${out_dir}"
  echo "------------------------------------------------------------"

  torchrun --nproc_per_node=3 --master_port 30000 \
    -m llava.train.train_mem \
    "${COMMON_ARGS[@]}" \
    "${stage_args[@]}" \
    "${EXTRA_ARGS[@]}"
}

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME:               ${MID_RUN_NAME}"
echo "OUTPUT_DIR:             ${FINAL_OUTPUT_DIR}"
echo "ENABLE_VFR:             ${ENABLE_VFR}"
echo "ENABLE_VFA:             ${ENABLE_VFA}"

if [[ "${ENABLE_VFR}" == "true" ]]; then
  rm -rf "${STAGE_A_DIR}"
  mkdir -p "${FINAL_OUTPUT_DIR}"

  run_stage \
    "${MID_RUN_NAME}_stageA_image" \
    "${PREV_STAGE_CHECKPOINT}" \
    "${DATA_YAML_IMAGE_ONLY}" \
    "${STAGE_A_DIR}" \
    "pad" \
    "flat" \
    "no_token" \
    "1" \
    "False" \
    "False" \
    "image"

  run_stage \
    "${MID_RUN_NAME}_stageB_video" \
    "${STAGE_A_DIR}" \
    "${DATA_YAML_VIDEO_ONLY}" \
    "${FINAL_OUTPUT_DIR}" \
    "anyres_max_8" \
    "spatial_unpad" \
    "grid" \
    "16" \
    "True" \
    "True" \
    "video"

  rm -rf "${STAGE_A_DIR}"
  echo "VFR two-stage done. Final artifacts kept in: ${FINAL_OUTPUT_DIR}"
else
  run_stage \
    "${MID_RUN_NAME}" \
    "${PREV_STAGE_CHECKPOINT}" \
    "${DATA_YAML_MIXED}" \
    "${FINAL_OUTPUT_DIR}" \
    "anyres_max_8" \
    "spatial_unpad" \
    "grid" \
    "16" \
    "True" \
    "True" \
    "both"
fi