import logging
import torch
from dataclasses import dataclass

from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoProcessor,
    TrainingArguments,
    set_seed
)

# 1. 匯入卻謝函式與類別
from train import(
    prepare_model,
    set_trainable_params,
    get_nb_trainable_parameters
)

# 從 args 檔案匯入參數類別
from args import DataArguments, ModelArguments, TrainingArguments

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_architecture():
    logger.info("開始檢查模型架構...")
    

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    set_seed(training_args.seed)
    # # 模擬 TrainingArguments
    # training_args = TrainingArguments(
    #     output_dir="./temp_model_check", # 隨便給一個路徑
        
    #     # --- 模擬您的 .sh 設定 ---
    #     bf16=True,
    #     gradient_checkpointing=True,
        
    #     # --- 傳入 .sh 腳本中的「學習率」參數 ---
    #     # (這些參數是在 args.py 中定義的)
    #     language_model_lr=0.0,
    #     connector_lr=1e-4,      # <-- 只訓練這個
    #     vision_tower_lr=0.0
    # )
    
    # ！！！修正點 (2)！！！
    # 仿造 train_mem.py，在物件建立「之後」，才動態新增 tune_... 屬性
    logger.info("正在動態新增 tune_... 旗標...")
    training_args.tune_language_model = (training_args.language_model_lr > 1e-9)      # --> False
    training_args.tune_mm_connector = (training_args.connector_lr > 1e-9)     # --> True
    training_args.tune_vision_tower = (training_args.vision_tower_lr > 1e-9)       # --> False


    logger.info(f"模擬設定: tune_language_model={training_args.tune_language_model}")
    logger.info(f"模擬設定: tune_mm_connector={training_args.tune_mm_connector}")
    logger.info(f"模擬設定: tune_vision_tower={training_args.tune_vision_tower}")
    
    
    # 3. 執行自己的函式來準備模型
    logger.info("正在呼叫 prepare_model()...")
    model = prepare_model(model_args, training_args)
    logger.info("模型準備完成。")
    
    logger.info("正在呼叫 set_trainable_params()...")
    set_trainable_params(model, training_args)
    
    trainable_params, total_params = get_nb_trainable_parameters(model)
    pct = 100.0 * trainable_params / max(total_params, 1)
    
    logger.info("---最終檢查結果---")
    logger.info(f"可訓練參數數量: {trainable_params:,d} / {total_params:,} ({pct:.2f}%)")
    
    if training_args.tune_mm_connector and not training_args.tune_language_model and not training_args.tune_vision_tower:
        if trainable_params > 0 and trainable_params < total_params:
            logger.info("✅ 檢查通過！模型已正確設定為「僅訓練 Projector」。")
        else:
            logger.error("❌ 檢查失敗！可訓練參數數量不符合預期。")
    else:
        logger.warn("警告：訓練設定不是「僅訓練 Projector」，請自行確認參數是否正確。")

if __name__ == "__main__":
    import os
    import sys

    # 獲取當前腳本的目錄 (.../smolvlm/train)
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    # 往上一層到套件目錄 (.../smolvlm)
    package_root = os.path.dirname(current_dir)
    
    # 再往上一層到專案根目錄 (.../smolvlm2)
    # 這是 import smolvlm.train.train 需要的根目錄
    project_root = os.path.dirname(package_root)
    
    # 將 *專案根目錄* 加入到 Python 搜尋路徑的最前面
    sys.path.insert(0, project_root)
    
    logger.info(f"已將專案根目錄 {project_root} 加入 PYTHONPATH")

    # 執行檢查
    check_model_architecture()