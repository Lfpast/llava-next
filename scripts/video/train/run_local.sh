#!/bin/bash
set -o pipefail
# ============================================================
# LLaVA-Video V100 训练脚本 (FP16 + Zero3 Offload)
# 针对V100优化：使用原生FP16，Zero3参数+优化器CPU卸载
# ============================================================

IMAGE_FOLDER="/home/user/liujian/jiayusheng/LLaVA-NeXT/data/train/image_media"
VIDEO_FOLDER="/home/user/liujian/jiayusheng/LLaVA-NeXT/data/train/video_media"
DATA_YAML="/home/user/liujian/jiayusheng/LLaVA-NeXT/scripts/video/train/local_data.yaml"

# ── Conda 环境 ──────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate llava
  set -u
fi

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONUNBUFFERED=1
export WANDB_MODE="disabled"
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# V100显存优化 - 不使用expandable_segments（PyTorch版本兼容性问题）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# V100 NCCL 修复 (无 NVLink, 禁用 P2P)
export NCCL_P2P_DISABLE=1
# DeepSpeed: 跳过系统 CUDA 与 PyTorch 编译 CUDA 版本不一致的检查
export DS_SKIP_CUDA_CHECK=1

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION="qwen_1_5"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si"

# ============================================================
# 【用户配置】输出路径与运行名称
#   RUN_NAME   : 本次运行的标识名，用于日志显示
#   OUTPUT_DIR : 模型 checkpoint 的保存目录（绝对路径或相对路径均可）
#                留空则自动生成（基于模型版本字符串）
# ============================================================
RUN_NAME="v1_lr1e5_16_4096"
OUTPUT_DIR="/home/user/liujian/jiayusheng/LLaVA-NeXT/work_dirs/siglip"

# 若未手动指定，则自动生成
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-local_video_v100"
[[ -n "$RUN_NAME" ]]   && MID_RUN_NAME="$RUN_NAME"
[[ -n "$OUTPUT_DIR" ]] && FINAL_OUTPUT_DIR="$OUTPUT_DIR/$MID_RUN_NAME" || FINAL_OUTPUT_DIR="./work_dirs/$MID_RUN_NAME"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME:               ${MID_RUN_NAME}"
echo "OUTPUT_DIR:             ${FINAL_OUTPUT_DIR}"

torchrun --nproc_per_node=8 --master_port 30000 \
    LLaVA-NeXT/llava/train/train_mem.py \
    --deepspeed LLaVA-NeXT/scripts/zero3_offload_v100.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_8 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --attn_implementation sdpa \
    --run_name $MID_RUN_NAME \
    --output_dir $FINAL_OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile False \
    --dataloader_drop_last True \
    --frames_upbound 16 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --save_only_model True \
2>&1 | awk '
  index($0, "DeepSpeed Op Builder: Installed CUDA version") > 0 { next }
  index($0, "TORCH_EXTENSION_NAME=cpu_adam") > 0 { next }
  $0 ~ /^\[[0-9]+\/[0-9]+\] c\+\+ .*cpu_adam/ { next }
  { print; fflush() }
'

exit 0