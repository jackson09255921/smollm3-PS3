#!/bin/bash
set -x -e


# --- 1. 設定環境變數 ---
# (Optional) Set up Hugging Face cache directory
# export HF_HOME=/path/to/user/cache


# Activate your conda environment
source /home/fireblue/miniconda3/etc/profile.d/conda.sh
conda activate smolvlm

# --- 2. 設定專案路徑 ---
PROJECT_ROOT=/mnt/c/Users/HankWang.DESKTOP-VFUI6TF/Desktop/smollm3-PS3/vision/smolvlm2

# --- 3. 設定資料路徑 ---
# yaml 設定檔案路徑
DATA_PATH="scripts/mixtures/align.yaml" # <--- 檢查你的資料路徑

# 圖片/影片資料夾路徑
DATA_FOLDER="/mnt/c/Users/HankWang.DESKTOP-VFUI6TF/Desktop/smollm3-PS3/dataset"

# 實驗名稱
RUN_NAME="ps3-500m-pretrain-projector-only"

# --- 4. 進入專案目錄並設定 PYTHONPATH ---
cd $PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Data config: $DATA_PATH"
echo "Data folder: $DATA_FOLDER"

# User-defined variables
MODEL_NAME="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" # <--- 基礎 LLM (500M)


# 啟用單機訓練
python smolvlm/train/train_mem.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --model_max_length 8192 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 7 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --connector_lr 1e-4 \
    --language_model_lr 2e-5 \
    --num_train_epochs 1 \
    --connector_lr 1e-4 \
    --language_model_lr 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --peft_enable False \
    --logging_steps 1 \
    --data_mixture $DATA_PATH \
    --data_folder $DATA_FOLDER \
    --dataloader_drop_last True \
    --dataloader_num_workers 4 \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1 \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name $RUN_NAME \
    \
    # --- MODIFICATION: PS3 學習率設定 --- \
    # (SmolVLMTrainer 會自動使用這些) \
    
    \
    # --- (保持不變) Scheduler 設定 --- \
    
    --vision_tower_lr 0.0 \
    --conntector_lr 1e-4 \
    --language_model_lr 0.0 \
    

echo "訓練完成！"